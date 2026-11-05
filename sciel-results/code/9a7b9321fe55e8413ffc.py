import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 150s kill
    
    # Problem parameters
    N = q ** n  # 2187 for q=3, n=7
    
    # Helper to convert index to list of digits
    def idx_to_word(idx):
        word = []
        for _ in range(n):
            word.append(idx % q)
            idx //= q
        return word
    
    # Precompute all words
    words = [idx_to_word(i) for i in range(N)]
    
    # Build word to index mapping
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Compute neighbors and store as list of numpy arrays for fast operations
    neighbors = []
    for i in range(N):
        w_i = words[i]
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val == w_i[pos]:
                    continue
                new_w = list(w_i)
                new_w[pos] = val
                neigh.append(word_to_idx[tuple(new_w)])
        neighbors.append(np.array(neigh))
        
    # Precompute neighbor matrix for vectorized operations
    # All neighbors have the same size: 1 + n*(q-1)
    neighbor_matrix = np.array(neighbors)
    
    best_code = None
    best_size = float('inf')
    
    def greedy_cover(uncovered_bool, code, time_limit):
        """
        Greedy set cover.
        uncovered_bool: numpy boolean array of size N, True if uncovered.
        code: list of indices.
        """
        code = list(code)
        # Work on a copy of uncovered state
        unc = uncovered_bool.copy()
        
        while unc.any() and time.time() < time_limit:
            # Find candidate that covers most uncovered points
            # covs[c] = number of uncovered points in neighbors[c]
            covs = unc[neighbor_matrix].sum(axis=1)
            
            max_cov = covs.max()
            if max_cov == 0:
                break
                
            # Get all candidates with max cov
            candidates = np.where(covs == max_cov)[0]
            
            if len(candidates) == 0:
                break
            
            chosen = random.choice(candidates)
            code.append(chosen)
            
            # Update uncovered
            unc[neighbors[chosen]] = False
            
        return code
    
    def prune_redundant(code, time_limit):
        """Remove redundant codewords."""
        if not code:
            return code
        
        code = list(code)
        # Use numpy for coverage counts
        cover_count = np.zeros(N, dtype=np.int32)
        
        for c in code:
            cover_count[neighbors[c]] += 1
            
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            to_remove = []
            
            # A codeword c is redundant if all u in neighbors[c] have cover_count[u] > 1
            for idx, c in enumerate(code):
                if np.min(cover_count[neighbors[c]]) > 1:
                    to_remove.append(idx)
            
            if to_remove:
                for idx in to_remove:
                    c = code[idx]
                    cover_count[neighbors[c]] -= 1
                remove_set = set(to_remove)
                code = [c for i, c in enumerate(code) if i not in remove_set]
                changed = True
                
        return code
    
    def kick_and_repair(code, num_kicks, time_limit):
        """Remove random codewords and repair with greedy."""
        if len(code) <= num_kicks:
            return code
        
        code = list(code)
        # Remove random codewords
        removed = random.sample(range(len(code)), num_kicks)
        for idx in sorted(removed, reverse=True):
            code.pop(idx)
        
        # Find uncovered using numpy
        covered = np.zeros(N, dtype=bool)
        for c in code:
            covered[neighbors[c]] = True
            
        uncovered = ~covered
        
        # Repair
        code = greedy_cover(uncovered, code, time_limit)
        code = prune_redundant(code, time_limit)
        
        return code
    
    def swap_improve(code, time_limit):
        """
        Try to replace one codeword with another to reduce size or improve coverage.
        This is a local search move.
        """
        if len(code) < 2:
            return code
        
        code = list(code)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Compute current coverage
            cover_count = np.zeros(N, dtype=np.int32)
            for c in code:
                cover_count[neighbors[c]] += 1
            
            # For each codeword, try removing it and see if we can add one back
            for idx, c in enumerate(code):
                if time.time() > time_limit:
                    break
                    
                # Temporarily remove c
                cover_count[neighbors[c]] -= 1
                
                # Find uncovered points
                uncovered = cover_count == 0
                
                if not uncovered.any():
                    # c was redundant, remove it
                    code.pop(idx)
                    improved = True
                    break
                
                # Find best replacement
                # A candidate d is good if it covers many uncovered points
                unc_count = uncovered.sum()
                covs = uncovered[neighbor_matrix].sum(axis=1)
                
                in_code = set(code)
                # Mask out in_code candidates
                covs_masked = covs.copy()
                for d in in_code:
                    covs_masked[d] = -1
                
                max_cov = covs_masked.max()
                
                if max_cov >= unc_count:
                    # We can cover all uncovered points with one codeword
                    candidates = np.where(covs_masked == max_cov)[0]
                    if len(candidates) > 0:
                        chosen = random.choice(candidates)
                        code.append(chosen)
                        # Update cover_count
                        cover_count[neighbors[chosen]] += 1
                        improved = True
                        break
                else:
                    # Can't cover all uncovered with one codeword.
                    # Revert
                    cover_count[neighbors[c]] += 1
            
        return code
    
    def two_for_one_improve(code, time_limit):
        """
        Try to remove two codewords and add one codeword that covers their unique points.
        This is a 2->1 reduction move.
        """
        if len(code) < 3:
            return code
        
        code = list(code)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Compute current coverage
            cover_count = np.zeros(N, dtype=np.int32)
            for c in code:
                cover_count[neighbors[c]] += 1
            
            # Try all pairs
            n_code = len(code)
            for i in range(n_code):
                if time.time() > time_limit:
                    break
                for j in range(i+1, n_code):
                    if time.time() > time_limit:
                        break
                    
                    # Temporarily remove i and j
                    c1 = code[i]
                    c2 = code[j]
                    cover_count[neighbors[c1]] -= 1
                    cover_count[neighbors[c2]] -= 1
                    
                    # Find uncovered points
                    uncovered = cover_count == 0
                    
                    if not uncovered.any():
                        # Both were redundant together, remove both
                        # Remove in reverse order to keep indices valid
                        code.pop(j)
                        code.pop(i)
                        improved = True
                        # Revert cover_count since we removed both
                        cover_count[neighbors[c1]] += 1
                        cover_count[neighbors[c2]] += 1
                        break
                    
                    unc_count = uncovered.sum()
                    covs = uncovered[neighbor_matrix].sum(axis=1)
                    
                    in_code = set(code)
                    covs_masked = covs.copy()
                    for d in in_code:
                        covs_masked[d] = -1
                    
                    max_cov = covs_masked.max()
                    
                    if max_cov >= unc_count:
                        # We can cover all uncovered points with one codeword
                        # This reduces size by 1 (remove 2, add 1)
                        candidates = np.where(covs_masked == max_cov)[0]
                        if len(candidates) > 0:
                            chosen = random.choice(candidates)
                            code.append(chosen)
                            cover_count[neighbors[chosen]] += 1
                            # Remove i and j
                            # Need to be careful with indices
                            # Since we appended, indices i and j are still valid if i<j
                            # But we need to remove from the list
                            # Let's rebuild the list excluding c1 and c2
                            code = [c for c in code if c != c1 and c != c2]
                            code.append(chosen)
                            improved = True
                            # Revert cover_count for the removed ones
                            cover_count[neighbors[c1]] += 1
                            cover_count[neighbors[c2]] += 1
                            break
                    else:
                        # Revert
                        cover_count[neighbors[c1]] += 1
                        cover_count[neighbors[c2]] += 1
                
                if improved:
                    break
            
        return code
    
    # Run multiple iterations with different strategies
    max_iterations = 1000 # Let time limit control this
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 5:
            break
        
        # Start with greedy
        initial_unc = np.ones(N, dtype=bool)
        code = greedy_cover(initial_unc, [], deadline - 2)
        code = prune_redundant(code, deadline - 2)
        
        # Try kicks
        for num_kicks in [1, 2, 3, 4, 5]:
            if time.time() > deadline - 2:
                break
            code = kick_and_repair(code, num_kicks, deadline - 2)
            code = swap_improve(code, deadline - 2)
            code = two_for_one_improve(code, deadline - 2)
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
                if best_size <= 186:
                    break
        
        if best_size <= 186:
            break
            
    # Convert internal indices back to lists of integers
    if best_code is None:
        # Fallback
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(words[c]) for c in best_code]
