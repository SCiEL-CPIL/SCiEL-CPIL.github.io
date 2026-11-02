import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    limit = 148.0  # Safety margin before 150s kill
    
    total_words = q ** n
    
    # Precompute base powers for index conversion
    base_powers = [q ** i for i in range(n)]
    
    def idx_to_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits
        
    def digits_to_idx(digits):
        return sum(d * b for d, b in zip(digits, base_powers))
        
    # Precompute neighbor lists for each word
    # neighbor_lists[i] is a list of indices within Hamming distance <= radius of i
    neighbor_lists = [None] * total_words
    for i in range(total_words):
        digits = idx_to_digits(i)
        neighbors = {i}
        if radius >= 1:
            for pos in range(n):
                orig = digits[pos]
                for new_val in range(q):
                    if new_val != orig:
                        new_digits = digits.copy()
                        new_digits[pos] = new_val
                        neighbors.add(digits_to_idx(new_digits))
        neighbor_lists[i] = list(neighbors)
        
    def greedy_cover(randomize=False):
        uncovered = np.ones(total_words, dtype=bool)
        solution = []
        candidates = list(range(total_words))
        if randomize:
            random.shuffle(candidates)
            
        while True:
            if time.time() - start_time > limit - 5:
                break
            if not np.any(uncovered):
                break
                
            best_idx = -1
            best_count = 0
            
            for i in candidates:
                cnt = 0
                for nb in neighbor_lists[i]:
                    if uncovered[nb]:
                        cnt += 1
                
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
                elif cnt == best_count and cnt > 0 and randomize:
                    if random.random() < 0.5:
                        best_idx = i
                        # best_count stays same
                    
            if best_idx == -1 or best_count == 0:
                break
                
            solution.append(best_idx)
            for nb in neighbor_lists[best_idx]:
                uncovered[nb] = False
                
        return solution

    def optimize_code(code):
        code_set = set(code)
        cover_count = np.zeros(total_words, dtype=np.int32)
        for idx in code:
            for nb in neighbor_lists[idx]:
                cover_count[nb] += 1
                
        improved = True
        while improved:
            if time.time() - start_time > limit - 2:
                return code
            
            improved = False
            # 1. Remove redundant codewords
            for i in range(len(code) - 1, -1, -1):
                idx = code[i]
                redundant = True
                for nb in neighbor_lists[idx]:
                    if cover_count[nb] == 1:
                        redundant = False
                        break
                
                if redundant:
                    for nb in neighbor_lists[idx]:
                        cover_count[nb] -= 1
                    code.pop(i)
                    code_set.remove(idx)
                    improved = True
                    break
            
            if not improved:
                # 2. Try swaps
                swap_made = False
                for i in range(len(code)):
                    idx_out = code[i]
                    
                    critical = []
                    for nb in neighbor_lists[idx_out]:
                        if cover_count[nb] == 1:
                            critical.append(nb)
                    
                    if not critical:
                        # Should have been caught, but handle just in case
                        for nb in neighbor_lists[idx_out]:
                            cover_count[nb] -= 1
                        code.pop(i)
                        code_set.remove(idx_out)
                        improved = True
                        swap_made = True
                        break
                    
                    # Find candidates that cover all critical words
                    # candidates_in = intersection of neighbors[w] for w in critical
                    # neighbors[w] is the set of words that cover w.
                    # Since distance is symmetric, neighbors[w] is the same as the list of words at dist <= 1 from w.
                    
                    # Start with the first critical word's neighbors
                    candidates_in = set(neighbor_lists[critical[0]])
                    
                    for w in critical[1:]:
                        candidates_in.intersection_update(neighbor_lists[w])
                        if not candidates_in:
                            break
                    
                    # Filter out those already in code
                    candidates_in -= code_set
                    
                    if candidates_in:
                        # Pick a candidate
                        # Randomize to escape local minima
                        idx_in = random.choice(list(candidates_in))
                        
                        # Perform swap
                        for nb in neighbor_lists[idx_out]:
                            cover_count[nb] -= 1
                        code_set.remove(idx_out)
                        
                        for nb in neighbor_lists[idx_in]:
                            cover_count[nb] += 1
                        code_set.add(idx_in)
                        
                        code[i] = idx_in
                        improved = True
                        swap_made = True
                        break
                        
                if swap_made:
                    continue
                else:
                    break
                    
        return code

    best_code = None
    best_len = float('inf')
    
    # Run multiple rounds
    # We can run more rounds if time permits
    while time.time() - start_time < limit - 10:
        code = greedy_cover(randomize=True)
        
        if not code:
            continue
            
        opt_code = optimize_code(code)
        
        if len(opt_code) < best_len:
            best_len = len(opt_code)
            best_code = opt_code
            if best_len <= 486:
                break

    if best_code is None:
        best_code = greedy_cover(randomize=False)
        
    return [idx_to_digits(idx) for idx in best_code]
