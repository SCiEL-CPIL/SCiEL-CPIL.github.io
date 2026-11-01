"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # For the specific case q=3, n=6, radius=1
    total_words = q ** n
    
    def word_to_idx(word, q, n):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx, q, n):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    # Precompute neighbors for each word (for radius=1)
    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        w = idx_to_word(idx, q, n)
        neighbors[idx].append(idx)  # self
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        w[pos] = new_val
                        nb_idx = word_to_idx(w, q, n)
                        neighbors[idx].append(nb_idx)
                w[pos] = orig
    
    # Convert neighbors to numpy arrays for faster operations
    neighbors_np = [np.array(nb, dtype=np.int32) for nb in neighbors]
    
    # Precompute neighbor matrix once (shared across all functions)
    nm = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        nm[i, neighbors_np[i]] = True
    
    # Convert nm to int32 for faster dot products
    nm_int = nm.astype(np.int32)
    
    def greedy_cover(restart_seed=0, initial_codewords=None, symmetry_tie_break=False):
        """Greedy set cover with coverage counting and optional symmetry-guided tie-breaking."""
        if time.time() > deadline:
            return None
        
        rng = random.Random(restart_seed)
        
        # Coverage count: how many codewords cover each word
        coverage = np.zeros(total_words, dtype=np.int32)
        codewords = []
        covered_mask = np.zeros(total_words, dtype=bool)
        
        # If we have initial codewords, add them
        if initial_codewords:
            for cw in initial_codewords:
                if cw not in codewords:
                    codewords.append(cw)
                    for nb in neighbors_np[cw]:
                        coverage[nb] += 1
                        covered_mask[nb] = True
        
        # Uncovered words as a boolean array (True = uncovered)
        uncovered_arr = np.ones(total_words, dtype=np.int32)
        for cw in codewords:
            uncovered_arr[neighbors_np[cw]] = 0
        
        # Greedy: repeatedly pick the codeword that covers the most uncovered words
        while time.time() < deadline:
            # Check if all covered
            if uncovered_arr.sum() == 0:
                break
            
            # Count uncovered neighbors for each candidate
            counts = nm_int.dot(uncovered_arr)
            
            # Find the candidate with max count
            best_count = counts.max()
            if best_count == 0:
                break
            
            # Get all candidates with max count
            candidates = np.where(counts == best_count)[0]
            
            # Tie-breaking: Choose the candidate that maximizes overlap with existing codewords
            if len(candidates) > 1:
                if len(candidates) > 5:
                    existing_covered_mask = (coverage > 0).astype(np.int32)
                    scores = nm_int[candidates, :].dot(existing_covered_mask)
                    best_idx_in_cands = int(np.argmax(scores))
                    best_cw = int(candidates[best_idx_in_cands])
                else:
                    best_cw = int(candidates[0])
            else:
                best_cw = int(candidates[0])
            
            # Add best_cw
            codewords.append(best_cw)
            uncovered_arr[neighbors_np[best_cw]] = 0
            coverage[neighbors_np[best_cw]] += 1
            covered_mask[neighbors_np[best_cw]] = True
        
        return codewords
    
    def remove_redundant(codewords):
        """Remove redundant codewords iteratively using coverage counts."""
        if time.time() > deadline:
            return codewords
        
        if not codewords:
            return codewords
        
        # Build coverage count: for each word, how many codewords cover it
        coverage = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            for nb in neighbors_np[cw]:
                coverage[nb] += 1
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                
                cw = codewords[i]
                
                # Check if all words in ball(cw) are covered by at least 2 codewords
                if coverage[neighbors_np[cw]].min() >= 2:
                    coverage[neighbors_np[cw]] -= 1
                    codewords.pop(i)
                    changed = True
                    break
        
        return codewords
    
    def local_search(codewords, max_iters=200):
        """
        Active local search: Try to replace codewords to enable further removals.
        Strategy: 
        1. Remove redundancies.
        2. For each codeword, try replacing it with a neighbor.
        3. If replacement allows removing another codeword (net -1), accept.
        """
        if time.time() > deadline:
            return codewords
        
        codewords = list(codewords)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            codewords = remove_redundant(codewords)
            
            if len(codewords) <= 73:
                break
                
            # Compute coverage
            coverage = np.zeros(total_words, dtype=np.int32)
            for cw in codewords:
                coverage[neighbors_np[cw]] += 1
            
            # Find codewords with highest slack (most redundant coverage)
            candidates_to_move = []
            for cw in codewords:
                slack = (coverage[neighbors_np[cw]] >= 2).sum()
                candidates_to_move.append((slack, cw))
            
            candidates_to_move.sort(reverse=True)
            
            for slack_val, cw in candidates_to_move:
                if time.time() > deadline:
                    return codewords
                
                neighbors_of_cw = neighbors_np[cw]
                
                # Find unique words covered only by cw
                unique_mask = (coverage == 1) & nm[cw, :]
                unique_count = unique_mask.sum()
                
                # Find best replacement neighbor
                best_new_cw = -1
                best_score = -1
                
                for new_cw in neighbors_of_cw:
                    if new_cw in codewords:
                        continue
                    
                    # Check validity: new_cw must cover all unique words of cw
                    if unique_count > 0:
                        if not np.all(nm[new_cw, :][unique_mask]):
                            continue
                    
                    # Score: How many words does new_cw cover that are already covered by >=2?
                    score = (nm[new_cw, :] & (coverage >= 2)).sum()
                    if score > best_score:
                        best_score = score
                        best_new_cw = new_cw
                
                if best_new_cw != -1:
                    # Execute move
                    codewords.remove(cw)
                    codewords.append(best_new_cw)
                    
                    # Remove redundancies
                    new_codewords = remove_redundant(codewords)
                    
                    if len(new_codewords) < len(codewords):
                        codewords = new_codewords
                        improved = True
                        break
                    else:
                        # Rollback
                        codewords.remove(best_new_cw)
                        codewords.append(cw)
        
        return remove_redundant(codewords)
    
    def final_polish(codewords):
        """
        Final high-precision polish: try to remove any single codeword by testing
        if the remaining set still covers everything. This is O(N * M) where N is
        the number of codewords and M is the total number of words.
        """
        if time.time() > deadline:
            return codewords
        
        codewords = list(codewords)
        codewords = remove_redundant(codewords)
        
        # Try removing each codeword one at a time
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                
                cw = codewords[i]
                
                # Temporarily remove cw and check if all words are still covered
                # Compute coverage without cw
                coverage = np.zeros(total_words, dtype=np.int32)
                for j, other_cw in enumerate(codewords):
                    if j != i:
                        coverage[neighbors_np[other_cw]] += 1
                
                # Check if any word has coverage 0
                if coverage.min() >= 1:
                    # cw is redundant even without the local check
                    codewords.pop(i)
                    changed = True
                    break
        
        return codewords
    
    def lagrangian_relaxation(codewords, max_iters=50):
        """
        Lagrangian relaxation approach: use subgradient method to dynamically weight
        the marginal benefit of adding each codeword, helping escape local minima.
        """
        if time.time() > deadline:
            return codewords
        
        codewords = list(codewords)
        codewords = remove_redundant(codewords)
        
        # Compute coverage
        coverage = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            coverage[neighbors_np[cw]] += 1
        
        # Dual variables: weight for each word indicating how "expensive" it is to leave uncovered
        # Initialize with uniform weights
        dual_vars = np.ones(total_words, dtype=np.float64)
        
        # Subgradient step size
        step_size = 1.0
        best_codewords = codewords.copy()
        best_count = len(codewords)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute weighted marginal benefit for each potential codeword
            # For each word not in codewords, compute how many uncovered/under-covered words it would help
            uncovered_weight = np.where(coverage < 1, dual_vars, 0.0)
            
            # Score for each candidate: sum of dual weights of words it covers
            scores = nm_int.dot(uncovered_weight.astype(np.int32))
            
            # Find candidates not in current codewords
            in_codewords = np.zeros(total_words, dtype=bool)
            for cw in codewords:
                in_codewords[cw] = True
            
            # Mask out current codewords
            scores[in_codewords] = -1
            
            # Find best candidate
            best_score = scores.max()
            if best_score <= 0:
                break
            
            best_cw = int(np.argmax(scores))
            
            # Add the best candidate
            codewords.append(best_cw)
            coverage[neighbors_np[best_cw]] += 1
            
            # Now try to remove redundancies
            codewords = remove_redundant(codewords)
            
            # Update coverage
            coverage = np.zeros(total_words, dtype=np.int32)
            for cw in codewords:
                coverage[neighbors_np[cw]] += 1
            
            # If we improved, update best
            if len(codewords) < best_count:
                best_count = len(codewords)
                best_codewords = codewords.copy()
            
            # Update dual variables using subgradient method
            # Gradient of Lagrangian with respect to dual vars
            # For each word, gradient is 1 if uncovered, 0 otherwise (simplified)
            gradient = np.where(coverage < 1, 1.0, 0.0)
            
            # Update dual variables
            dual_vars += step_size * gradient
            
            # Decay step size
            step_size *= 0.95
        
        return best_codewords
    
    # Strategy:
    # 1. Start with the tetracode-based construction (81 words)
    # 2. Aggressively remove redundant codewords
    # 3. Try multiple random restarts of greedy to find better solutions
    # 4. Use Lagrangian relaxation to escape local minima
    # 5. Combine best results
    
    best_result = None
    best_count = float('inf')
    
    # First, try the tetracode construction with redundancy removal
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for tail_a in range(3):
        for tail_b in range(3):
            for h in tetracode:
                word = h + [tail_a, tail_b]
                idx = word_to_idx(word, q, n)
                initial_codewords.append(idx)
    
    # Remove redundant from initial
    reduced = remove_redundant(initial_codewords.copy())
    if len(reduced) < best_count:
        best_count = len(reduced)
        best_result = reduced
    
    # Try greedy from scratch with different seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        
        result = greedy_cover(restart_seed=seed, initial_codewords=None, symmetry_tie_break=False)
        if result is not None:
            result = remove_redundant(result)
            if len(result) < best_count:
                best_count = len(result)
                best_result = result
    
    # Try hybrid: start with partial tetracode, then greedy
    for seed in range(20):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        initial = rng.sample(initial_codewords, k=min(len(initial_codewords), rng.randint(30, 60)))
        result = greedy_cover(restart_seed=seed, initial_codewords=initial, symmetry_tie_break=False)
        if result is not None:
            result = remove_redundant(result)
            if len(result) < best_count:
                best_count = len(result)
                best_result = result
    
    # Try additional greedy restarts with different initial random seeds
    for seed in range(50, 100):
        if time.time() > deadline:
            break
        
        result = greedy_cover(restart_seed=seed, initial_codewords=None, symmetry_tie_break=False)
        if result is not None:
            result = remove_redundant(result)
            if len(result) < best_count:
                best_count = len(result)
                best_result = result
    
    # Apply Lagrangian relaxation to best results found so far
    if best_result is not None:
        lag_result = lagrangian_relaxation(best_result, max_iters=30)
        lag_result = remove_redundant(lag_result)
        if len(lag_result) < best_count:
            best_count = len(lag_result)
            best_result = lag_result
    
    # Final local search on best result
    if best_result is not None:
        best_result = local_search(best_result)
        best_result = remove_redundant(best_result)
        
        # Apply final polish
        best_result = final_polish(best_result)
    
    # Convert back to lists
    if best_result is None:
        return [idx_to_word(idx, q, n) for idx in initial_codewords]
    
    return [idx_to_word(idx, q, n) for idx in best_result]
# EVOLVE-BLOCK-END
