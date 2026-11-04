import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit

    # Pre-computation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, the set of triples it covers (as a set of indices)
    block_triples = []
    for b in all_blocks:
        tris = frozenset(triple_to_idx[tri] for tri in itertools.combinations(b, t))
        block_triples.append(tris)
        
    # For each triple, the list of blocks that cover it
    triple_to_blocks = defaultdict(list)
    for b_idx, tris in enumerate(block_triples):
        for tri_idx in tris:
            triple_to_blocks[tri_idx].append(b_idx)

    def greedy_construct(randomize=False):
        """
        Strong greedy: at each step pick the block covering the most uncovered triples.
        If randomize=True, add small random noise to tie-break / perturb.
        """
        # uncovered_counts[tri] = 1 if uncovered, 0 if covered
        uncovered = np.ones(n_triples, dtype=np.int8)
        blocks = []
        
        while np.any(uncovered):
            # For each block, compute gain = number of currently uncovered triples it covers
            # To speed up, we can iterate blocks and count uncovered
            # n_blocks can be large (C(24,7) = 346104), so we need efficiency
            
            best_gain = -1
            best_b = -1
            
            # Optimization: only consider blocks that cover at least one uncovered triple
            # We can check if a block has any uncovered triple
            for b_idx in range(n_blocks):
                tris = block_triples[b_idx]
                gain = 0
                for tri in tris:
                    if uncovered[tri]:
                        gain += 1
                if randomize and gain > 0:
                    gain += random.randint(0, 2)  # small perturbation
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
                    if best_gain == len(tris):
                        break
            
            if best_b < 0:
                break
                
            blocks.append(best_b)
            for tri in block_triples[best_b]:
                if uncovered[tri]:
                    uncovered[tri] = 0
                    
        return blocks

    def optimize(blocks, time_budget=10):
        """
        Local search: try to remove redundant blocks, then try swaps (remove one, add one better).
        Returns optimized list of block indices.
        """
        if not blocks:
            return []
            
        # counts[tri] = number of blocks covering tri
        counts = np.zeros(n_triples, dtype=np.int8)
        for b_idx in blocks:
            for tri in block_triples[b_idx]:
                counts[tri] += 1
        
        active = [True] * len(blocks)
        current_blocks = blocks[:]
        
        def try_remove_redundant():
            nonlocal active, current_blocks
            removed = False
            indices = list(range(len(current_blocks)))
            random.shuffle(indices)
            for i in indices:
                if not active[i]:
                    continue
                b_idx = current_blocks[i]
                redundant = True
                for tri in block_triples[b_idx]:
                    if counts[tri] <= 1:
                        redundant = False
                        break
                if redundant:
                    active[i] = False
                    for tri in block_triples[b_idx]:
                        counts[tri] -= 1
                    removed = True
            return removed

        # Initial removal pass
        try_remove_redundant()
        
        end_time = time.time() + time_budget
        no_improve = 0
        
        while time.time() < end_time:
            # Try removal again
            if try_remove_redundant():
                no_improve = 0
                continue
            
            # Try a swap
            active_indices = [i for i in range(len(current_blocks)) if active[i]]
            if not active_indices:
                break
                
            i_out = random.choice(active_indices)
            b_out = current_blocks[i_out]
            
            # Temporarily remove
            active[i_out] = False
            for tri in block_triples[b_out]:
                counts[tri] -= 1
            
            # Find uncovered triples
            uncovered_mask = (counts == 0)
            uncovered_tris = np.where(uncovered_mask)[0]
            
            if len(uncovered_tris) == 0:
                # b_out was redundant, keep it removed
                no_improve = 0
                # Try more removals
                try_remove_redundant()
                continue
            
            if len(uncovered_tris) > 5:
                # Too many uncovered, revert
                active[i_out] = True
                for tri in block_triples[b_out]:
                    counts[tri] += 1
                no_improve += 1
                if no_improve > 50:
                    break
                continue
            
            # Find candidate that covers all uncovered_tris
            candidates = set(triple_to_blocks[uncovered_tris[0]])
            for tri in uncovered_tris[1:]:
                candidates.intersection_update(triple_to_blocks[tri])
                if not candidates:
                    break
            
            if candidates:
                current_active_set = set(current_blocks[i] for i in range(len(current_blocks)) if active[i])
                valid_candidates = [b for b in candidates if b != b_out and b not in current_active_set]
                
                if valid_candidates:
                    # Pick candidate with max gain (covers most other uncovered, though all uncovered must be covered)
                    # Actually, since all uncovered_tris must be covered, we just need one.
                    # Prefer one that also covers more already-covered triples? No, we want to minimize total.
                    # A swap that removes one and adds one keeps size same.
                    # But we want to remove one and add one, so size is same.
                    # However, the new block might allow further removals.
                    
                    b_in = random.choice(valid_candidates)
                    
                    # Add b_in
                    current_blocks.append(b_in)
                    active.append(True)
                    for tri in block_triples[b_in]:
                        counts[tri] += 1
                    
                    no_improve = 0
                    # Try removals again
                    try_remove_redundant()
                else:
                    # Revert
                    active[i_out] = True
                    for tri in block_triples[b_out]:
                        counts[tri] += 1
                    no_improve += 1
                    if no_improve > 50:
                        break
            else:
                # Revert
                active[i_out] = True
                for tri in block_triples[b_out]:
                    counts[tri] += 1
                no_improve += 1
                if no_improve > 50:
                    break
        
        final_blocks = [current_blocks[i] for i in range(len(current_blocks)) if active[i]]
        return final_blocks

    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Greedy with various randomizations
    for i in range(10):
        if time.time() > deadline:
            break
        cand = greedy_construct(randomize=(i > 0))
        opt_cand = optimize(cand, time_budget=5)
        
        if len(opt_cand) < best_len:
            best_len = len(opt_cand)
            best_blocks = opt_cand
            # If we hit lower bound, stop
            if best_len <= 69:
                break
    
    # Phase 2: More aggressive optimization on best solution
    if best_blocks and time.time() < deadline - 10:
        time_budget = deadline - time.time() - 5
        # Try to further optimize
        improved = optimize(best_blocks, time_budget=time_budget)
        if len(improved) < best_len:
            best_len = len(improved)
            best_blocks = improved
    
    return [list(all_blocks[b_idx]) for b_idx in best_blocks]
