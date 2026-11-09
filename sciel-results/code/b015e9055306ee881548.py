import itertools
import random
import time
import sys
from collections import defaultdict

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
    
    # Precompute overlap matrix for pairs of blocks (how many triples they share)
    # This helps in local search to evaluate swap quality
    # For memory efficiency, compute on-demand with caching
    overlap_cache = {}
    
    _CACHE[key] = (all_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples, overlap_cache)
    return _CACHE[key]

def _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, blocks_containing, n_blocks, seed, block_triples):
    rng = random.Random(seed)
    uncovered = full_mask
    blocks_used = []
    used_mask = 0
    active_blocks = set(range(n_blocks))
    
    while uncovered:
        uncovered_count = uncovered.bit_count()
        
        # Find block with max gain
        best_gain = 0
        best_candidates = []
        for idx in active_blocks:
            gain = (block_masks[idx] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_candidates = [idx]
                if gain == uncovered_count:
                    break
            elif gain == best_gain and gain > 0:
                best_candidates.append(idx)
        
        if not best_candidates or best_gain == 0:
            break
        
        # Random choice among best candidates for diversity
        selected_idx = rng.choice(best_candidates)
        
        blocks_used.append(selected_idx)
        used_mask |= block_masks[selected_idx]
        uncovered &= ~used_mask
        active_blocks.discard(selected_idx)
        
    return blocks_used

def _walksat(v, k, t, initial_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples, seed, deadline):
    """
    Max-SAT formulation: each block is a Boolean variable.
    For each triple, a clause requires at least one containing block to be true.
    WalkSAT minimizes unsatisfied clauses while penalizing block count.
    """
    rng = random.Random(seed)
    
    # Build clause structure: for each triple, list of blocks containing it
    # We already have blocks_containing[tri_idx] = list of block indices
    
    # Initial assignment: True for blocks in initial_blocks, False otherwise
    assignment = [False] * n_blocks
    for b in initial_blocks:
        assignment[b] = True
    
    best_size = sum(assignment)
    best_assignment = assignment[:]
    
    # Penalty weight for keeping blocks off
    # We want to minimize: num_unsatisfied_clauses * W_CLAUSE + num_true_blocks * W_BLOCK
    # W_CLAUSE should be large enough that satisfying all clauses is preferred
    # W_BLOCK controls the trade-off
    W_CLAUSE = 10
    W_BLOCK = 1
    
    def _evaluate(assignment):
        """Return (num_unsatisfied, num_true)"""
        num_true = sum(assignment)
        num_unsat = 0
        for tri_idx in range(n_triples):
            covered = False
            for b in blocks_containing[tri_idx]:
                if assignment[b]:
                    covered = True
                    break
            if not covered:
                num_unsat += 1
        return num_unsat, num_true
    
    def _score(assignment):
        num_unsat, num_true = _evaluate(assignment)
        return num_unsat * W_CLAUSE + num_true * W_BLOCK
    
    current_score = _score(assignment)
    current_unsat, current_true = _evaluate(assignment)
    
    if current_unsat == 0 and current_true < best_size:
        best_size = current_true
        best_assignment = assignment[:]
    
    no_improve = 0
    max_no_improve = 5000  # iterations
    
    while time.time() < deadline:
        # If all clauses satisfied, try to reduce block count
        if current_unsat == 0:
            # Try to turn off a random block
            true_blocks = [i for i in range(n_blocks) if assignment[i]]
            if not true_blocks:
                break
            
            # Try turning off blocks one at a time, prefer those with most redundancy
            rng.shuffle(true_blocks)
            improved = False
            for b in true_blocks:
                assignment[b] = False
                new_unsat, new_true = _evaluate(assignment)
                if new_unsat == 0:
                    # Good, keep it off
                    current_unsat = new_unsat
                    current_true = new_true
                    current_score = _score(assignment)
                    if current_true < best_size:
                        best_size = current_true
                        best_assignment = assignment[:]
                    improved = True
                    break
                else:
                    assignment[b] = True
            
            if not improved:
                no_improve += 1
                if no_improve > 20:
                    break
                continue
        
        # WalkSAT step: pick an unsatisfied clause, flip a variable in it
        # Find an unsatisfied clause
        unsat_clauses = []
        for tri_idx in range(n_triples):
            covered = False
            for b in blocks_containing[tri_idx]:
                if assignment[b]:
                    covered = True
                    break
            if not covered:
                unsat_clauses.append(tri_idx)
        
        if not unsat_clauses:
            current_unsat = 0
            current_true = sum(assignment)
            if current_true < best_size:
                best_size = current_true
                best_assignment = assignment[:]
            continue
        
        # Pick a random unsatisfied clause
        tri_idx = rng.choice(unsat_clauses)
        clause_vars = blocks_containing[tri_idx]
        
        if not clause_vars:
            break
        
        # Score each variable flip in the clause
        best_var = -1
        best_score = float('inf')
        
        for b in clause_vars:
            old_val = assignment[b]
            assignment[b] = not old_val
            s = _score(assignment)
            if s < best_score:
                best_score = s
                best_var = b
            assignment[b] = old_val
        
        if best_var == -1:
            break
        
        # With some probability, pick a random variable instead (for diversity)
        if rng.random() < 0.3:
            chosen = rng.choice(clause_vars)
        else:
            chosen = best_var
        
        assignment[chosen] = not assignment[chosen]
        
        new_unsat, new_true = _evaluate(assignment)
        new_score = _score(assignment)
        
        if new_score < current_score:
            current_score = new_score
            current_unsat = new_unsat
            current_true = new_true
            no_improve = 0
            if new_unsat == 0 and new_true < best_size:
                best_size = new_true
                best_assignment = assignment[:]
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                # Random restart within current search
                no_improve = 0
                # Perturb: flip a few random variables
                for _ in range(3):
                    b = rng.randint(0, n_blocks - 1)
                    assignment[b] = not assignment[b]
                current_unsat, current_true = _evaluate(assignment)
                current_score = _score(assignment)
    
    return best_assignment, best_size


def _local_search(v, k, t, initial_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples, seed, deadline, overlap_cache):
    rng = random.Random(seed)
    
    selected = set(initial_blocks)
    coverage_count = [0] * n_triples
    
    for b in selected:
        for tri_idx in block_triples[b]:
            coverage_count[tri_idx] += 1
    
    best_size = len(selected)
    best_selected = set(selected)
    
    no_improve_time = time.time()
    max_no_improve = 1.5  # Even shorter patience
    
    def _get_overlap(b1, b2):
        key = (min(b1, b2), max(b1, b2))
        if key not in overlap_cache:
            overlap_cache[key] = (block_masks[b1] & block_masks[b2]).bit_count()
        return overlap_cache[key]
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to remove a redundant block
        if selected:
            items = list(selected)
            rng.shuffle(items)
            for b in items:
                redundant = True
                for tri_idx in block_triples[b]:
                    if coverage_count[tri_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for tri_idx in block_triples[b]:
                        coverage_count[tri_idx] -= 1
                    selected.remove(b)
                    improved = True
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_selected = set(selected)
                    no_improve_time = time.time()
                    break
        
        if not improved:
            # 2. Try swaps with overlap-aware scoring
            swap_done = False
            sel_list = list(selected)
            rng.shuffle(sel_list)
            
            for b in sel_list:
                if time.time() >= deadline:
                    break
                    
                # Find unique triples for b
                unique_tris = []
                for tri_idx in block_triples[b]:
                    if coverage_count[tri_idx] == 1:
                        unique_tris.append(tri_idx)
                
                if not unique_tris:
                    continue
                
                # Try multiple unique triples as starting points
                for tri_idx in rng.sample(unique_tris, min(4, len(unique_tris))):
                    candidates = blocks_containing[tri_idx]
                    
                    if len(candidates) <= 1:
                        continue
                    
                    cand_list = list(candidates)
                    rng.shuffle(cand_list)
                    
                    best_b_prime = -1
                    best_score = -1
                    
                    for b_prime in cand_list[:50]:
                        if b_prime in selected:
                            continue
                        
                        # Check safety: b_prime must cover all unique triples of b
                        safe = True
                        for tri_idx2 in block_triples[b]:
                            if coverage_count[tri_idx2] == 1 and not (block_masks[b_prime] & (1 << tri_idx2)):
                                safe = False
                                break
                        
                        if not safe:
                            continue
                        
                        # Score: prefer blocks with less overlap with existing selected blocks
                        # (more diverse coverage)
                        overlap_sum = 0
                        for b_other in selected:
                            if b_other == b:
                                continue
                            overlap_sum += _get_overlap(b_prime, b_other)
                        
                        # Negative overlap = better (less redundancy)
                        score = -overlap_sum
                        
                        if score > best_score:
                            best_score = score
                            best_b_prime = b_prime
                    
                    if best_b_prime != -1:
                        # Perform swap
                        for tri_idx2 in block_triples[b]:
                            coverage_count[tri_idx2] -= 1
                        selected.remove(b)
                        
                        for tri_idx2 in block_triples[best_b_prime]:
                            coverage_count[tri_idx2] += 1
                        selected.add(best_b_prime)
                        
                        swap_done = True
                        improved = True
                        if len(selected) < best_size:
                            best_size = len(selected)
                            best_selected = set(selected)
                        no_improve_time = time.time()
                        break
                
                if swap_done:
                    break
        
        if not improved:
            # 3. Try 2-opt: remove 2 blocks, add 1 block
            if len(selected) >= 3:
                sel_list2 = list(selected)
                rng.shuffle(sel_list2)
                two_opt_done = False
                
                for i in range(min(12, len(sel_list2))):
                    if time.time() >= deadline:
                        break
                    b1 = sel_list2[i]
                    
                    unique_tris_1 = [tri_idx for tri_idx in block_triples[b1] if coverage_count[tri_idx] == 1]
                    if not unique_tris_1:
                        continue
                    
                    for j in range(i + 1, min(i + 6, len(sel_list2))):
                        if time.time() >= deadline:
                            break
                        b2 = sel_list2[j]
                        
                        unique_tris_2 = [tri_idx for tri_idx in block_triples[b2] if coverage_count[tri_idx] == 1]
                        
                        all_unique = set(unique_tris_1) | set(unique_tris_2)
                        if not all_unique:
                            continue
                        
                        first_tri = next(iter(all_unique))
                        candidates = blocks_containing[first_tri]
                        
                        if len(candidates) <= 1:
                            continue
                        
                        cand_list2 = list(candidates)
                        rng.shuffle(cand_list2)
                        
                        best_b_prime = -1
                        best_score = -1
                        
                        for b_prime in cand_list2[:30]:
                            if b_prime in selected:
                                continue
                            
                            # Safety check
                            safe = True
                            for tri_idx2 in block_triples[b1]:
                                if coverage_count[tri_idx2] == 1 and not (block_masks[b_prime] & (1 << tri_idx2)):
                                    safe = False
                                    break
                            if safe:
                                for tri_idx2 in block_triples[b2]:
                                    if coverage_count[tri_idx2] == 1 and not (block_masks[b_prime] & (1 << tri_idx2)):
                                        safe = False
                                        break
                            
                            if not safe:
                                continue
                            
                            # Score by overlap
                            overlap_sum = 0
                            for b_other in selected:
                                if b_other == b1 or b_other == b2:
                                    continue
                                overlap_sum += _get_overlap(b_prime, b_other)
                            
                            score = -overlap_sum
                            
                            if score > best_score:
                                best_score = score
                                best_b_prime = b_prime
                        
                        if best_b_prime != -1:
                            # Perform 2-opt
                            for tri_idx2 in block_triples[b1]:
                                coverage_count[tri_idx2] -= 1
                            selected.remove(b1)
                            
                            for tri_idx2 in block_triples[b2]:
                                coverage_count[tri_idx2] -= 1
                            selected.remove(b2)
                            
                            for tri_idx2 in block_triples[best_b_prime]:
                                coverage_count[tri_idx2] += 1
                            selected.add(best_b_prime)
                            
                            two_opt_done = True
                            improved = True
                            if len(selected) < best_size:
                                best_size = len(selected)
                                best_selected = set(selected)
                            no_improve_time = time.time()
                            break
                        
                        if two_opt_done:
                            break
                    
                    if two_opt_done:
                        break
        
        if not improved:
            if time.time() - no_improve_time > max_no_improve:
                break
    
    # Final fill: add blocks to cover any remaining uncovered triples
    uncovered_mask = 0
    for tri_idx in range(n_triples):
        if coverage_count[tri_idx] == 0:
            uncovered_mask |= (1 << tri_idx)
    
    if uncovered_mask:
        # Greedily add blocks to cover uncovered triples
        active = set(range(n_blocks))
        while uncovered_mask and time.time() < deadline:
            best_idx = -1
            max_gain = 0
            for idx in active:
                if idx in selected:
                    continue
                gain = (block_masks[idx] & uncovered_mask).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_idx = idx
                if gain == uncovered_mask.bit_count():
                    break
            if best_idx == -1 or max_gain == 0:
                break
            for tri_idx2 in block_triples[best_idx]:
                coverage_count[tri_idx2] += 1
            selected.add(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
            if len(selected) < best_size:
                best_size = len(selected)
                best_selected = set(selected)
    
    return list(best_selected)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Precompute
    all_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples, overlap_cache = _precompute(v, k, t)
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    
    # Run multiple restarts with shorter local search to allow more diversity
    while time.time() < deadline - 10:
        # 1. Greedy (fast)
        initial_blocks = _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, blocks_containing, n_blocks, seed, block_triples)
        
        if not initial_blocks:
            seed += 1
            continue
        
        # 2. Local Search with shorter sub-deadline for more restarts
        remaining = deadline - time.time()
        # Shorter local search: 8s max, but at least 2s if we have time
        ls_time = min(8.0, max(2.0, remaining - 15.0))
        ls_deadline = time.time() + ls_time
        
        if ls_deadline > deadline - 12.0:
            ls_deadline = deadline - 12.0
        
        if ls_deadline <= time.time() + 1.0:
            break
            
        blocks = _local_search(v, k, t, initial_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples, seed, ls_deadline, overlap_cache)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            print(f"New best (LS): {best_len} blocks at t={time.time()-start_time:.1f}s")
        
        # 3. WalkSAT refinement on the best solution found so far
        ws_remaining = deadline - time.time()
        if ws_remaining > 3.0:
            ws_deadline = time.time() + min(5.0, ws_remaining - 2.0)
            ws_init = best_blocks if best_blocks else blocks
            ws_assignment, ws_size = _walksat(v, k, t, ws_init, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples, seed + 1000, ws_deadline)
            ws_blocks = [i for i in range(n_blocks) if ws_assignment[i]]
            
            if len(ws_blocks) < best_len:
                best_len = len(ws_blocks)
                best_blocks = ws_blocks
                print(f"New best (WS): {best_len} blocks at t={time.time()-start_time:.1f}s")
        
        seed += 1
    
    if best_blocks is None:
        blocks = _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, blocks_containing, n_blocks, 999, block_triples)
        best_blocks = blocks
    
    # Final WalkSAT polish pass with remaining time
    remaining = deadline - time.time()
    if remaining > 2.0 and best_blocks:
        ws_deadline = time.time() + min(remaining - 1.0, 8.0)
        ws_assignment, ws_size = _walksat(v, k, t, best_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples, seed + 2000, ws_deadline)
        ws_blocks = [i for i in range(n_blocks) if ws_assignment[i]]
        
        if len(ws_blocks) < len(best_blocks):
            best_blocks = ws_blocks
            print(f"Final WS polish: {len(best_blocks)} blocks")
    
    result = [list(all_blocks[idx]) for idx in best_blocks]
    
    return result
