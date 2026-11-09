import time
import random

def construct():
    """
    Build a set of distinct integers in 1..256 maximizing size such that
    no subset sum is achievable by subsets of different cardinalities.
    
    Uses a fast greedy construction with incremental sum tracking,
    then attempts local search improvements to grow the set further.
    """
    start_time = time.time()
    DEADLINE = 55.0  # safety margin before 90s kill
    
    def elapsed():
        return time.time() - start_time
    
    def is_done():
        return elapsed() > DEADLINE
    
    def compute_sums_by_k(A):
        """Compute sums_by_k for set A. sums_by_k[k] = set of sums from exactly k elements."""
        n = len(A)
        sums_by_k = [set() for _ in range(n + 1)]
        sums_by_k[0] = {0}
        for a in A:
            for k in range(n - 1, -1, -1):
                if sums_by_k[k]:
                    new_sums = {s + a for s in sums_by_k[k]}
                    sums_by_k[k + 1].update(new_sums)
        return sums_by_k
    
    def is_valid_sums(sums_by_k):
        """Check if no sum appears in two different cardinality classes."""
        n = len(sums_by_k)
        for i in range(n):
            for j in range(i + 1, n):
                if sums_by_k[i] & sums_by_k[j]:
                    return False
        return True
    
    def try_add_element(A, candidate, sums_by_k):
        """Try to add candidate to A. Returns (success, new_sums_by_k)."""
        if candidate in A:
            return False, None
        
        n = len(A)
        # Compute new sums_by_k after adding candidate
        new_sums_by_k = [set(s) for s in sums_by_k]
        new_sums_by_k.append(set())  # new cardinality level
        
        for k in range(n, -1, -1):
            if new_sums_by_k[k]:
                new_sums = {s + candidate for s in new_sums_by_k[k]}
                new_sums_by_k[k + 1].update(new_sums)
        
        # Check validity
        if not is_valid_sums(new_sums_by_k):
            return False, None
        
        return True, new_sums_by_k
    
    def greedy_build(order, max_size=60):
        """Build a valid set by greedily adding elements in the given order."""
        A = []
        sums_by_k = [{0}]
        
        for candidate in order:
            if len(A) >= max_size:
                break
            success, new_sums = try_add_element(A, candidate, sums_by_k)
            if success:
                A.append(candidate)
                sums_by_k = new_sums
            if is_done():
                break
        
        return A, sums_by_k
    
    def local_search_improve(A, sums_by_k, max_size=60):
        """Try to improve the set by swapping elements or adding new ones."""
        A = list(A)
        sums_by_k = [set(s) for s in sums_by_k]
        improved = True
        
        while improved and not is_done():
            improved = False
            n = len(A)
            
            # Try adding elements not in A
            for candidate in range(1, 257):
                if len(A) >= max_size:
                    break
                if candidate in A:
                    continue
                success, new_sums = try_add_element(A, candidate, sums_by_k)
                if success:
                    A.append(candidate)
                    sums_by_k = new_sums
                    improved = True
                    if is_done():
                        break
            
            if is_done():
                break
            
            # Try swapping: remove one element, add two
            if len(A) < max_size:
                for i in range(len(A)):
                    if is_done():
                        break
                    removed = A[i]
                    A_temp = A[:i] + A[i+1:]
                    # Recompute sums after removal
                    sums_temp = compute_sums_by_k(A_temp)
                    
                    # Try adding two elements
                    for c1 in range(1, 257):
                        if is_done():
                            break
                        if c1 in A_temp:
                            continue
                        success1, sums1 = try_add_element(A_temp, c1, sums_temp)
                        if not success1:
                            continue
                        for c2 in range(c1 + 1, 257):
                            if is_done():
                                break
                            if c2 in A_temp or c2 == c1:
                                continue
                            A_temp2 = A_temp + [c1]
                            success2, sums2 = try_add_element(A_temp2, c2, sums1)
                            if success2:
                                A = A_temp2 + [c2]
                                sums_by_k = sums2
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return A, sums_by_k
    
    # Strategy 1: Greedy from small to large
    best_A = []
    best_sums = None
    
    A1, sums1 = greedy_build(range(1, 257))
    if len(A1) > len(best_A):
        best_A = A1
        best_sums = sums1
    
    if is_done():
        return [int(x) for x in best_A] if best_A else [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # Strategy 2: Greedy from large to small
    A2, sums2 = greedy_build(range(256, 0, -1))
    if len(A2) > len(best_A):
        best_A = A2
        best_sums = sums2
    
    if is_done():
        return [int(x) for x in best_A] if best_A else [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # Strategy 3: Local search improvement on best found so far
    if best_A:
        best_A, best_sums = local_search_improve(best_A, best_sums)
    
    if is_done():
        return [int(x) for x in best_A] if best_A else [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # Strategy 4: Random restarts with greedy + local search
    random.seed(12345)
    
    for restart in range(500):
        if is_done():
            break
        if len(best_A) >= 60:
            break
        
        # Create a random order with bias toward certain ranges
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        # Mix in some structured order
        order = []
        for c in candidates:
            if len(order) < 60:
                order.append(c)
        
        A_rand, sums_rand = greedy_build(order)
        
        # Local search on this result
        A_rand, sums_rand = local_search_improve(A_rand, sums_rand)
        
        if len(A_rand) > len(best_A):
            best_A = A_rand
            best_sums = sums_rand
    
    # Strategy 5: Try specific structured sets
    structured_sets = [
        # Powers of 2
        [1, 2, 4, 8, 16, 32, 64, 128, 256],
        # Multiples of 3
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60,
         63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114,
         117, 120, 123, 126, 129, 132, 135, 138, 141, 144, 147, 150, 153, 156, 159, 162,
         165, 168, 171, 174, 177, 180, 183, 186, 189, 192, 195, 198, 201, 204, 207, 210,
         213, 216, 219, 222, 225, 228, 231, 234, 237, 240, 243, 246, 249, 252, 255],
        # Sparse set
        [1, 3, 7, 15, 31, 63, 127, 255],
        # Geometric with ratio 3
        [1, 3, 9, 27, 81, 243],
    ]
    
    for sset in structured_sets:
        if is_done():
            break
        # Filter to valid range
        sset = [x for x in sset if 1 <= x <= 256]
        # Deduplicate
        sset = list(dict.fromkeys(sset))
        
        if sset:
            sums_struct = compute_sums_by_k(sset)
            if is_valid_sums(sums_struct):
                # Try to extend
                A_struct, sums_struct = local_search_improve(sset, sums_struct)
                if len(A_struct) > len(best_A):
                    best_A = A_struct
                    best_sums = sums_struct
    
    if not best_A:
        best_A = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # Final validation
    sums_final = compute_sums_by_k(best_A)
    if not is_valid_sums(sums_final):
        # Fallback to powers of 2
        best_A = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # Ensure all are Python ints in a list
    return [int(x) for x in best_A]
