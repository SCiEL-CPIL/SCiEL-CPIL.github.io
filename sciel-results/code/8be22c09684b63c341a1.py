import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Generate all words
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # 2. Precompute neighbors for radius 1
    # neighbors[i] is a list of indices j such that dist(words[i], words[j]) <= 1
    neighbors = []
    for i, w in enumerate(words):
        neigh = [i]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                neigh.append(idx)
                w_list[pos] = orig_val # Restore
        neighbors.append(neigh)
        
    # Convert neighbors to sets for faster intersection/membership if needed, 
    # but lists are fine for iteration.
    # For the repair phase, we need to know which words are uncovered.
    # We will maintain a 'counts' array.
    
    # Helper: Calculate coverage counts for a given code list
    def get_counts(code_list):
        counts = np.zeros(N, dtype=np.int32)
        for idx in code_list:
            counts[neighbors[idx]] += 1
        return counts

    # Helper: Greedy Set Cover on currently uncovered points
    # Returns a new code list that covers everything
    def greedy_repair(code_list):
        counts = get_counts(code_list)
        uncovered_mask = counts == 0
        if not np.any(uncovered_mask):
            return list(code_list)
        
        new_code = list(code_list)
        uncovered_indices = np.where(uncovered_mask)[0]
        
        while len(uncovered_indices) > 0:
            if time.time() > deadline - 10:
                break
            
            # Find the candidate that covers the most uncovered points
            # Optimization: Only consider candidates that cover at least 1 uncovered point
            # Since N is small (2187), we can iterate all candidates if needed, 
            # but we can restrict to those in the "ball" of uncovered points?
            # Actually, just iterate all N.
            
            best_idx = -1
            max_gain = 0
            
            # We can speed this up by only checking candidates that are neighbors of some uncovered point?
            # Or just check all. 2187 * 15 (avg neighbors) is ~30k ops. Very fast.
            
            for i in range(N):
                # Count how many of neighbors[i] are in uncovered_indices
                # Using numpy sum on boolean mask
                # neighbors[i] is a list of ints
                # We need to check if counts[neighbors[i]] == 0
                # gain = sum(1 for nb in neighbors[i] if counts[nb] == 0)
                
                # Vectorized:
                # nb_indices = np.array(neighbors[i])
                # gain = np.sum(counts[nb_indices] == 0)
                
                # To avoid array creation overhead in tight loop, manual loop might be faster for small lists
                gain = 0
                for nb in neighbors[i]:
                    if counts[nb] == 0:
                        gain += 1
                        if gain > max_gain:
                            # Early exit? No, we need max. But if gain > max_gain, we can update.
                            pass
                
                if gain > max_gain:
                    max_gain = gain
                    best_idx = i
            
            if max_gain == 0:
                # Should not happen if problem is solvable, but break to avoid infinite loop
                break
                
            # Add best_idx to code
            new_code.append(best_idx)
            
            # Update counts
            for nb in neighbors[best_idx]:
                counts[nb] += 1
            
            # Update uncovered_indices (remove those now covered)
            # This is O(N) if we rebuild, or we can just check at the end of loop
            # Let's just rebuild the list of uncovered indices
            uncovered_indices = np.where(counts == 0)[0]
            
        return new_code

    # Helper: Remove redundant words
    def prune_code(code_list):
        counts = get_counts(code_list)
        changed = True
        while changed:
            changed = False
            for c in list(code_list):
                # Check if all neighbors have count > 1
                # If min count in neighbors[c] > 1, c is redundant
                min_cnt = np.min(counts[neighbors[c]])
                if min_cnt > 1:
                    code_list.remove(c)
                    for nb in neighbors[c]:
                        counts[nb] -= 1
                    changed = True
        return code_list

    # Helper: Swap improvement
    def improve_by_swaps(code_list, max_iterations=50):
        counts = get_counts(code_list)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations and time.time() < deadline - 5:
            improved = False
            iterations += 1
            
            # Try to remove redundant words first
            code_list = prune_code(code_list)
            counts = get_counts(code_list)
            
            # Try swaps
            # For each codeword c, if it's not redundant, try to find a replacement
            # that allows removing c or leads to a better state.
            # Simpler: Try to replace c with w such that w covers all critical points of c
            # and w is not in code.
            
            random.shuffle(code_list)
            for c in code_list:
                if time.time() > deadline - 5:
                    break
                
                # Identify critical points for c: points covered only by c
                crit_points = []
                for nb in neighbors[c]:
                    if counts[nb] == 1:
                        crit_points.append(nb)
                        
                if not crit_points:
                    # c is redundant, prune_code should have removed it? 
                    # Maybe counts changed? Re-check
                    if np.min(counts[neighbors[c]]) > 1:
                        # Remove it
                        code_list.remove(c)
                        for nb in neighbors[c]:
                            counts[nb] -= 1
                        improved = True
                    continue
                
                # Find a replacement w not in code that covers all crit_points
                # Candidates: neighbors of the first critical point
                p = crit_points[0]
                candidates = [x for x in neighbors[p] if x not in code_list]
                
                best_w = -1
                best_score = -1
                
                for w in candidates:
                    # Check if w covers ALL crit_points
                    valid = True
                    for k in crit_points:
                        if k not in neighbors[w]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Score: How many points with count==2 does w cover? (Creates redundancy)
                    score = 0
                    for nb in neighbors[w]:
                        if counts[nb] == 2:
                            score += 1
                            
                    if score > best_score:
                        best_score = score
                        best_w = w
                        
                if best_w != -1:
                    # Perform swap
                    code_list.remove(c)
                    for nb in neighbors[c]:
                        counts[nb] -= 1
                        
                    code_list.append(best_w)
                    for nb in neighbors[best_w]:
                        counts[nb] += 1
                        
                    improved = True
                    break # Restart loop as counts changed
                    
        return code_list

    # Initial Solution: Randomized Greedy
    best_code = []
    
    # We will try a few different random initializations
    # Start with an empty set and build up? Or start with a large random set and prune?
    # Greedy from scratch is usually good.
    
    # 1. Greedy from scratch
    code = []
    counts = np.zeros(N, dtype=np.int32)
    uncovered_count = N
    
    while uncovered_count > 0 and time.time() < deadline - 20:
        # Find best candidate
        best_idx = -1
        max_gain = 0
        
        for i in range(N):
            # Gain is number of neighbors with count 0
            gain = 0
            for nb in neighbors[i]:
                if counts[nb] == 0:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        
        if max_gain == 0:
            break
            
        code.append(best_idx)
        for nb in neighbors[best_idx]:
            if counts[nb] == 0:
                uncovered_count -= 1
            counts[nb] += 1
            
    # Prune and Improve
    code = prune_code(code)
    code = improve_by_swaps(code)
    best_code = list(code)
    
    # 2. Kick-and-Repair Phase
    # Try to improve by kicking out random words and repairing
    kick_sizes = [1, 2, 3]
    num_kicks = 50 # Adjust based on time
    
    for _ in range(num_kicks):
        if time.time() > deadline - 10:
            break
            
        current_code = list(best_code)
        
        # Kick: Remove random words
        k = random.choice(kick_sizes)
        if len(current_code) <= k:
            continue
            
        words_to_remove = random.sample(current_code, k)
        for w in words_to_remove:
            current_code.remove(w)
            
        # Repair: Greedy fill
        current_code = greedy_repair(current_code)
        
        # Prune
        current_code = prune_code(current_code)
        
        # Improve
        current_code = improve_by_swaps(current_code)
        
        # Check validity and update best
        # Validity check: ensure all counts >= 1
        counts = get_counts(current_code)
        if np.min(counts) >= 1:
            if len(current_code) < len(best_code):
                best_code = current_code
            # Even if not smaller, if it's the same size, maybe keep it? 
            # No, we want the minimum size.
            
    # Final Prune
    best_code = prune_code(best_code)
    
    # Return as list of lists
    return [list(words[i]) for i in best_code]
