import itertools
import random
import time
import sys

# Global cache
_CACHE = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    start_time = time.time()
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {tri: i for i, tri in enumerate(triples)}
    n_triples = len(triples)
    
    # block_masks[i] is a bitmask of triples covered by block i
    block_masks = [0] * n_blocks
    # blocks_containing[j] is a list of block indices containing triple j
    blocks_containing = [[] for _ in range(n_triples)]
    # block_triples[i] is a list of triple indices in block i
    block_triples = [[] for _ in range(n_blocks)]
    
    for i, block in enumerate(all_blocks):
        mask = 0
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            mask |= (1 << idx)
            blocks_containing[idx].append(i)
            block_triples[i].append(idx)
        block_masks[i] = mask
        
    full_mask = (1 << n_triples) - 1
    
    _CACHE[key] = (all_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples)
    return _CACHE[key]

def _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, blocks_containing, n_blocks, seed, block_triples, alpha=1.0):
    """Greedy with diversity parameter alpha. Higher alpha favors blocks that cover more uncovered triples relative to their total coverage."""
    rng = random.Random(seed)
    uncovered = full_mask
    blocks_used = []
    used_mask = 0
    active_blocks = set(range(n_blocks))
    
    while uncovered:
        best_score = -1
        best_block_idx = -1
        best_candidates = []
        uncovered_count = uncovered.bit_count()
        
        for idx in active_blocks:
            gain = (block_masks[idx] & uncovered).bit_count()
            if gain == 0:
                continue
            # Score: gain / (total coverage of block)^alpha, or just gain with random tie-breaking
            total = block_masks[idx].bit_count()
            score = gain / (total ** alpha)
            # Add small random perturbation for diversity
            score += rng.random() * 0.01 * gain
            
            if score > best_score:
                best_score = score
                best_block_idx = idx
                best_candidates = [idx]
                if gain == uncovered_count:
                    break
            elif abs(score - best_score) < 1e-9:
                best_candidates.append(idx)
            
        if best_block_idx == -1:
            break
            
        selected_idx = rng.choice(best_candidates)
        
        blocks_used.append(selected_idx)
        used_mask |= block_masks[selected_idx]
        uncovered &= ~used_mask
        active_blocks.remove(selected_idx)
        
    return blocks_used

def _local_search(v, k, t, initial_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples, seed, deadline):
    """Enhanced local search with:
    1. Redundant block removal
    2. Single swaps (replace one block with another that covers all its unique triples)
    3. Two-for-one swaps (remove two blocks, add one block that covers all their unique triples)
    4. Hill-climbing with restarts within the time budget
    """
    rng = random.Random(seed)
    
    def compute_coverage(selected):
        coverage_count = [0] * n_triples
        for b in selected:
            for tri_idx in block_triples[b]:
                coverage_count[tri_idx] += 1
        return coverage_count
    
    def is_valid(selected, coverage_count):
        for i in range(n_triples):
            if coverage_count[i] == 0:
                return False
        return True
    
    def find_redundant(selected, coverage_count):
        for b in selected:
            redundant = True
            for tri_idx in block_triples[b]:
                if coverage_count[tri_idx] == 1:
                    redundant = False
                    break
            if redundant:
                return b
        return None
    
    def try_single_swap(selected, coverage_count, rng):
        """Try to replace one block with another that covers all its unique triples."""
        sel_list = list(selected)
        rng.shuffle(sel_list)
        
        for b in sel_list:
            # Find unique triples for b
            unique_tris = []
            for tri_idx in block_triples[b]:
                if coverage_count[tri_idx] == 1:
                    unique_tris.append(tri_idx)
            
            if not unique_tris:
                # If no unique triples, block is redundant (should have been caught)
                continue
                
            # Pick a random unique triple to find candidates
            tri_idx = rng.choice(unique_tris)
            candidates = blocks_containing[tri_idx]
            
            if len(candidates) <= 1:
                continue
                
            # Shuffle candidates and try a few
            cand_list = list(candidates)
            rng.shuffle(cand_list)
            
            for b_prime in cand_list[:100]:  # Limit candidates checked
                if b_prime in selected:
                    continue
                
                # Check if b_prime covers all unique triples of b
                covers_all = True
                for ut in unique_tris:
                    if not (block_masks[b_prime] & (1 << ut)):
                        covers_all = False
                        break
                
                if not covers_all:
                    continue
                
                # Perform swap
                for tri_idx2 in block_triples[b]:
                    coverage_count[tri_idx2] -= 1
                selected.remove(b)
                
                for tri_idx2 in block_triples[b_prime]:
                    coverage_count[tri_idx2] += 1
                selected.add(b_prime)
                
                return True
            
        return False
    
    def try_two_for_one_swap(selected, coverage_count, rng):
        """Try to remove two blocks and add one block that covers all their unique triples."""
        if len(selected) < 2:
            return False
        
        sel_list = list(selected)
        rng.shuffle(sel_list)
        
        # Try a limited number of pairs
        max_pairs = min(50, len(sel_list) * (len(sel_list) - 1) // 2)
        pairs_tried = 0
        
        for i in range(len(sel_list)):
            if pairs_tried >= max_pairs:
                break
            for j in range(i + 1, len(sel_list)):
                if pairs_tried >= max_pairs:
                    break
                pairs_tried += 1
                
                b1 = sel_list[i]
                b2 = sel_list[j]
                
                # Find unique triples for b1 and b2 (excluding overlap with each other)
                unique_tris = set()
                for tri_idx in block_triples[b1]:
                    if coverage_count[tri_idx] == 1 or (coverage_count[tri_idx] == 2 and tri_idx in block_triples[b2]):
                        # This triple is only covered by b1 and/or b2
                        # Check if it's in b2
                        if tri_idx in block_triples[b2]:
                            if coverage_count[tri_idx] == 2:
                                unique_tris.add(tri_idx)
                        else:
                            if coverage_count[tri_idx] == 1:
                                unique_tris.add(tri_idx)
                
                for tri_idx in block_triples[b2]:
                    if coverage_count[tri_idx] == 1:
                        unique_tris.add(tri_idx)
                    elif coverage_count[tri_idx] == 2:
                        # Check if it's in b1
                        if tri_idx in block_triples[b1]:
                            unique_tris.add(tri_idx)
                
                if not unique_tris:
                    continue
                
                # Find candidate blocks that cover all unique triples
                # Use the first unique triple to get candidates
                first_tri = next(iter(unique_tris))
                candidates = blocks_containing[first_tri]
                
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                for b_prime in cand_list[:50]:
                    if b_prime in selected:
                        continue
                    
                    # Check if b_prime covers all unique triples
                    covers_all = True
                    for ut in unique_tris:
                        if not (block_masks[b_prime] & (1 << ut)):
                            covers_all = False
                            break
                    
                    if not covers_all:
                        continue
                    
                    # Perform swap: remove b1, b2, add b_prime
                    for tri_idx2 in block_triples[b1]:
                        coverage_count[tri_idx2] -= 1
                    for tri_idx2 in block_triples[b2]:
                        coverage_count[tri_idx2] -= 1
                    selected.discard(b1)
                    selected.discard(b2)
                    
                    for tri_idx2 in block_triples[b_prime]:
                        coverage_count[tri_idx2] += 1
                    selected.add(b_prime)
                    
                    return True
        
        return False
    
    # Main local search loop
    selected = set(initial_blocks)
    coverage_count = compute_coverage(selected)
    
    best_size = len(selected)
    best_selected = set(selected)
    
    no_improve_count = 0
    max_no_improve = 20
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        
        # 1. Try to remove redundant blocks (multiple passes)
        while True:
            redundant_b = find_redundant(selected, coverage_count)
            if redundant_b is None:
                break
            for tri_idx in block_triples[redundant_b]:
                coverage_count[tri_idx] -= 1
            selected.remove(redundant_b)
            improved = True
            if len(selected) < best_size:
                best_size = len(selected)
                best_selected = set(selected)
        
        if not improved:
            # 2. Try single swaps
            if try_single_swap(selected, coverage_count, rng):
                improved = True
                # Recompute coverage after swap
                coverage_count = compute_coverage(selected)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = set(selected)
        
        if not improved:
            # 3. Try two-for-one swaps
            if try_two_for_one_swap(selected, coverage_count, rng):
                improved = True
                # Recompute coverage after swap
                coverage_count = compute_coverage(selected)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = set(selected)
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
    
    return list(best_selected)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Precompute
    all_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples = _precompute(v, k, t)
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    alpha_values = [1.0, 0.5, 0.8, 1.2, 0.3, 1.5]
    
    # Run multiple restarts with different alpha values and seeds
    while time.time() < deadline - 10:
        alpha = alpha_values[seed % len(alpha_values)]
        
        # 1. Greedy with diversity
        initial_blocks = _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, blocks_containing, n_blocks, seed, block_triples, alpha=alpha)
        
        if not initial_blocks:
            seed += 1
            continue
        
        # 2. Local Search with sub-deadline
        remaining = deadline - time.time()
        # Allocate more time to early runs, less to later ones
        if seed < 3:
            ls_time = min(25.0, remaining - 5.0)
        elif seed < 8:
            ls_time = min(15.0, remaining - 5.0)
        else:
            ls_time = min(8.0, remaining - 5.0)
        
        ls_deadline = time.time() + max(1.0, ls_time)
        
        if ls_deadline > deadline - 3.0:
            ls_deadline = deadline - 3.0
            
        if ls_deadline <= time.time():
            break
            
        blocks = _local_search(v, k, t, initial_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples, seed, ls_deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            print(f"New best: {best_len} blocks at t={time.time()-start_time:.1f}s (seed={seed}, alpha={alpha})", file=sys.stderr)
        
        seed += 1
    
    if best_blocks is None:
        blocks = _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, blocks_containing, n_blocks, 999, block_triples, alpha=1.0)
        best_blocks = blocks
    
    result = [list(all_blocks[idx]) for idx in best_blocks]
    
    return result
