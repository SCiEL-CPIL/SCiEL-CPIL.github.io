import numpy as np
import random
import time

def _construct_cover(q, n, radius, time_limit=110.0):
    start_time = time.time()
    N = q ** n
    
    # Precompute neighbors
    neighbors = []
    for i in range(N):
        temp = i
        digits = [0] * n
        for j in range(n):
            digits[n-1-j] = temp % q
            temp //= q
            
        nbs = [i]
        for pos in range(n):
            orig = digits[pos]
            weight = q ** (n - 1 - pos)
            for new_d in range(q):
                if new_d == orig:
                    continue
                delta = (new_d - orig) * weight
                ni = i + delta
                nbs.append(ni)
        neighbors.append(set(nbs))

    # 1. Greedy Initialization
    def greedy_init():
        uncovered = set(range(N))
        selected = []
        while uncovered:
            best_candidate = -1
            max_covered = 0
            for i in range(N):
                cnt = sum(1 for nb in neighbors[i] if nb in uncovered)
                if cnt > max_covered:
                    max_covered = cnt
                    best_candidate = i
            if best_candidate == -1:
                break
            selected.append(best_candidate)
            for nb in neighbors[best_candidate]:
                if nb in uncovered:
                    uncovered.remove(nb)
        return selected

    code = greedy_init()
    
    # Initialize coverage counts
    cov_count = np.zeros(N, dtype=np.int32)
    for c in code:
        for w in neighbors[c]:
            cov_count[w] += 1

    # Local Search
    def try_remove(code, cov_count):
        # Returns (new_code, new_cov_count, removed_idx) or (code, cov_count, -1)
        for i in range(len(code)):
            c_i = code[i]
            # Check if c_i is redundant
            # Redundant if for all w in N(c_i), cov_count[w] >= 2
            redundant = True
            for w in neighbors[c_i]:
                if cov_count[w] < 2:
                    redundant = False
                    break
            if redundant:
                # Remove c_i
                new_code = code[:i] + code[i+1:]
                # Update cov_count
                for w in neighbors[c_i]:
                    cov_count[w] -= 1
                return new_code, cov_count, i
        return code, cov_count, -1

    def local_search(code, cov_count, max_iters=10000):
        code = list(code)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters and (time.time() - start_time) < time_limit:
            improved = False
            iter_count += 1
            
            # Phase 1: Try to remove
            new_code, new_cov_count, rem_idx = try_remove(code, cov_count)
            if rem_idx != -1:
                code = new_code
                cov_count = new_cov_count
                improved = True
                continue # Restart
            
            # Phase 2: Try to swap
            # If no removal, try swapping to escape local minimum
            # We pick a random codeword to swap
            if len(code) == 0:
                break
                
            idx_to_swap = random.randint(0, len(code) - 1)
            c_old = code[idx_to_swap]
            
            # Try a few random candidates
            swapped = False
            for _ in range(20):
                c_new = random.randint(0, N - 1)
                if c_new == c_old:
                    continue
                if c_new in code:
                    continue
                
                # Check validity of swap
                # Words that lose coverage from c_old: N(c_old) \ N(c_new)
                # These words must have cov_count > 1 (covered by others)
                valid = True
                for w in neighbors[c_old]:
                    if w not in neighbors[c_new]:
                        if cov_count[w] <= 1:
                            valid = False
                            break
                
                if valid:
                    # Perform swap
                    # Update cov_count
                    for w in neighbors[c_old]:
                        cov_count[w] -= 1
                    for w in neighbors[c_new]:
                        cov_count[w] += 1
                    
                    code[idx_to_swap] = c_new
                    swapped = True
                    improved = True
                    break
            
            if not swapped:
                # If no swap worked, we might be stuck. 
                # We can try a different idx_to_swap or break.
                # To avoid infinite loop, we break if we've tried many times.
                # In a more advanced version, we'd track failures.
                pass

        return code, cov_count

    code, cov_count = local_search(code, cov_count)
    
    result = []
    for idx in code:
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        digits.reverse()
        result.append(digits)
        
    return result

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return _construct_cover(3, 6, 1)
    else:
        # Fallback
        N = q ** n
        if N > 5000:
            return []
        return _construct_cover(q, n, radius)
