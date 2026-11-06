import time
import numpy as np
from itertools import combinations

def construct():
    """
    Constructs the Steiner System S(5, 6, 12) using Mixed Integer Linear Programming (MILP).
    Falls back to backtracking if MILP is unavailable or fails.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n_points = 12
    block_size = 6
    sub_size = 5
    target_blocks = 132
    total_subs = 792 # C(12, 5)
    
    # 1. Precomputation
    all_blocks = list(combinations(range(n_points), block_size))
    n_blocks = len(all_blocks)
    
    # Map each block to the set of 5-subsets it covers
    block_to_subs = {}
    for b in all_blocks:
        subs = tuple(combinations(b, sub_size))
        block_to_subs[b] = subs
        
    # Inverted index: for each 5-subset, which blocks contain it?
    subs_to_blocks = {sub: [] for sub in combinations(range(n_points), sub_size)}
    for b, subs in block_to_subs.items():
        for s in subs:
            subs_to_blocks[s].append(b)

    # --- Attempt MILP Solution ---
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import lil_matrix
        
        all_subs = list(combinations(range(n_points), sub_size))
        n_subs = len(all_subs)
        sub_to_idx = {s: i for i, s in enumerate(all_subs)}
        
        A = lil_matrix((n_subs, n_blocks), dtype=int)
        for j, b in enumerate(all_blocks):
            for s in combinations(b, sub_size):
                i = sub_to_idx[s]
                A[i, j] = 1
        
        A_csr = A.tocsr()
        
        c = -np.ones(n_blocks)
        constraints = LinearConstraint(A_csr, -np.inf, 1)
        integrality = np.ones(n_blocks)
        bounds = Bounds(0, 1)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            x = res.x
            selected_indices = [i for i in range(n_blocks) if x[i] > 0.5]
            if len(selected_indices) == target_blocks:
                selected_blocks = [all_blocks[i] for i in selected_indices]
                # Verify
                covered = set()
                valid = True
                for b in selected_blocks:
                    subs = set(block_to_subs[b])
                    if covered & subs:
                        valid = False
                        break
                    covered.update(subs)
                if valid:
                    return selected_blocks
        # If MILP failed or didn't return optimal, fall through to backtracking
    except Exception:
        pass

    # --- Fallback: Backtracking ---
    # (Same as original code)
    first_block = (0, 1, 2, 3, 4, 5)
    covered = set(block_to_subs[first_block])
    chosen_blocks = [first_block]
    last_block = first_block
    
    def backtrack(covered, chosen_blocks, last_block, depth):
        if time.time() > deadline:
            return None
        if len(chosen_blocks) == target_blocks:
            if len(covered) == total_subs:
                return list(chosen_blocks)
            else:
                return None
        
        uncovered_subs = [s for s in subs_to_blocks if s not in covered]
        if not uncovered_subs:
            return None
            
        best_sub = None
        best_candidates = None
        min_count = float('inf')
        
        for s in uncovered_subs:
            candidates = []
            for b in subs_to_blocks[s]:
                if b <= last_block:
                    continue
                conflict = False
                for sub in block_to_subs[b]:
                    if sub in covered:
                        conflict = True
                        break
                if not conflict:
                    candidates.append(b)
            
            count = len(candidates)
            if count == 0:
                return None
            if count < min_count:
                min_count = count
                best_sub = s
                best_candidates = candidates
                if count == 1:
                    break
        
        if best_candidates is None:
            return None
            
        for b in best_candidates:
            new_covered = set(covered)
            for sub in block_to_subs[b]:
                new_covered.add(sub)
            result = backtrack(new_covered, chosen_blocks + [b], b, depth + 1)
            if result is not None:
                return result
                
        return None

    solution = backtrack(covered, chosen_blocks, last_block, 1)
    
    if solution:
        return solution
    else:
        # Greedy fallback
        covered = set()
        greedy_blocks = []
        for b in all_blocks:
            subs = block_to_subs[b]
            if not any(s in covered for s in subs):
                greedy_blocks.append(b)
                covered.update(subs)
                if len(greedy_blocks) >= target_blocks:
                    break
        return greedy_blocks
