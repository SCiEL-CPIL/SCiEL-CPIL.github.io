import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit
    
    def generate_candidates(weight):
        """Generate all integer vectors of given Hamming weight with entries in {-1, 1}."""
        cands = []
        for idxs in itertools.combinations(range(D), weight):
            for signs in itertools.product([1, -1], repeat=weight):
                v = [0] * D
                for i, s in zip(idxs, signs):
                    v[i] = s
                cands.append(v)
        return np.array(cands, dtype=np.int8)

    def find_max_packing(candidates, max_dot):
        """
        Find a large subset of candidates such that pairwise dot product <= max_dot.
        Uses Greedy + Local Search (1-for-2 swaps).
        """
        n = len(candidates)
        if n == 0:
            return np.array([], dtype=np.int64)
        
        # Compute Gram matrix
        # candidates is (n, D) int8
        # G = candidates @ candidates.T -> (n, n)
        # Dot products are small integers.
        G = candidates @ candidates.T
        
        # Conflict matrix: True if dot > max_dot
        # We need i != j, but diagonal is weight (which is > max_dot usually, e.g. 4 > 2).
        # So diagonal is naturally conflicting.
        conflict = (G > max_dot)
        
        # Greedy Selection
        # We want to pick indices such that conflict[i, j] is False for all selected i, j.
        selected_mask = np.zeros(n, dtype=bool)
        
        # Order candidates to improve greedy? 
        # Maybe sort by number of conflicts? (Degree in conflict graph)
        # Lower degree first is often good for Independent Set.
        degrees = np.sum(conflict, axis=1)
        order = np.argsort(degues) # This is wrong, degrees is int, argsort works.
        
        # Actually, just iterating 0..n-1 is fast. Let's try that first.
        # If time permits, we can do multiple passes with different orders.
        
        for i in range(n):
            if selected_mask[i]:
                continue
            # Check if i conflicts with any selected
            # conflict[i, selected_mask] is a boolean array
            # If any is True, conflict exists.
            if not np.any(conflict[i, selected_mask]):
                selected_mask[i] = True
                
        # Local Search: 1-for-2 swaps
        # Try to remove one selected node and add two non-selected nodes.
        # This is O(N^3) if done naively.
        # We can limit the number of iterations or swaps.
        
        selected_indices = np.where(selected_mask)[0]
        n_selected = len(selected_indices)
        
        # To speed up local search, we can just do a few passes.
        # Since N is ~5000, N^2 is 25M.
        # Checking all pairs of non-selected is too slow.
        
        # Alternative Local Search:
        # Try to add any unselected node. (Already done in greedy, but maybe after swaps?)
        # Actually, the greedy is maximal (no single node can be added).
        # So we must swap.
        
        # Let's do a simple heuristic: 
        # Pick a random selected node, remove it, then try to add as many non-selected nodes as possible greedily.
        # If we get more than 1, keep the swap.
        
        # To keep it fast, we limit the number of swap attempts.
        max_swap_attempts = 200
        attempts = 0
        improved = True
        while improved and attempts < max_swap_attempts and time.time() < deadline - 5:
            improved = False
            attempts += 1
            
            # Pick a random selected node to remove
            if n_selected == 0:
                break
            rem_idx = selected_indices[np.random.randint(n_selected)]
            
            # Temporarily remove it
            selected_mask[rem_idx] = False
            n_selected -= 1
            
            # Try to add non-selected nodes greedily
            # We need to check against current selected_mask
            added = []
            # Iterate through all non-selected?
            # To be fast, we can just iterate through a random subset or all.
            # N is 5000. Iterating 5000 times is fast.
            for j in range(n):
                if selected_mask[j]:
                    continue
                # Check if j can be added
                if not np.any(conflict[j, selected_mask]):
                    selected_mask[j] = True
                    added.append(j)
                    n_selected += 1
            
            # If we added more than 1, it's an improvement (net gain >= 1)
            # We removed 1, added len(added).
            if len(added) > 1:
                improved = True
                selected_indices = np.where(selected_mask)[0]
                n_selected = len(selected_indices)
            else:
                # Revert
                selected_mask[rem_idx] = True
                for j in added:
                    selected_mask[j] = False
                # n_selected unchanged (back to original)
                
        return candidates[selected_mask]

    best_vectors = []
    best_score = 0

    # Try Weight 4
    if time.time() < deadline - 10:
        cands_4 = generate_candidates(4)
        # Max dot for weight 4 is 4/2 = 2.
        res_4 = find_max_packing(cands_4, 2)
        if len(res_4) > best_score:
            best_score = len(res_4)
            best_vectors = res_4

    # Try Weight 3
    if time.time() < deadline - 5:
        cands_3 = generate_candidates(3)
        # Max dot for weight 3 is 3/2 = 1.5 -> 1.
        res_3 = find_max_packing(cands_3, 1)
        if len(res_3) > best_score:
            best_score = len(res_3)
            best_vectors = res_3

    # Try Weight 5?
    # Weight 5: dot <= 2.5 -> 2.
    # Candidates: C(11,5)*32 = 462 * 32 = 14784.
    # G is 14784^2 ~ 218M. Too big for memory/time?
    # 218M * 1 byte = 218MB. Might be tight but possible.
    # However, the condition is tighter relative to the "density"?
    # Let's stick to 3 and 4 first. If time permits, try 5.
    
    if time.time() < deadline - 15:
        cands_5 = generate_candidates(5)
        # Max dot 2.
        res_5 = find_max_packing(cands_5, 2)
        if len(res_5) > best_score:
            best_score = len(res_5)
            best_vectors = res_5

    if not best_vectors:
        # Fallback: just return some valid small set
        # e.g. e1, -e1, e2, -e2 ...
        vecs = []
        for i in range(D):
            v = [0]*D
            v[i] = 1
            vecs.append(v)
            v[i] = -1
            vecs.append(v)
        best_vectors = np.array(vecs, dtype=np.int64)

    return np.array(best_vectors, dtype=np.int64)
