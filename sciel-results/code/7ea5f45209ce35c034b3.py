import time
from itertools import combinations

def _check_tournament(n, orientation):
    # Build in-neighborhoods N^-(u) as bitsets
    # N^-(u) contains v if v -> u
    # orientation index for (i, j) with i < j:
    # idx = i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
    # Let's precompute indices or just iterate.
    
    # Efficiently building bitsets
    # n <= 90, so bitsets fit in Python ints easily.
    
    N_minus = [0] * n
    
    # Map pair (i, j) to index in orientation
    # i < j
    # idx(i, j) = sum_{k=0}^{i-1} (n - 1 - k) + (j - i - 1)
    # Sum_{k=0}^{i-1} (n - 1 - k) = i * (n - 1) - i * (i - 1) // 2
    # So idx = i * (n - 1) - i * (i - 1) // 2 + j - i - 1
    
    def get_idx(i, j):
        return i * (n - 1) - i * (i - 1) // 2 + j - i - 1

    for i in range(n):
        for j in range(i + 1, n):
            idx = get_idx(i, j)
            bit = orientation[idx]
            if bit == 0:
                # i -> j
                # j is in N^+(i), so i is in N^-(j)
                N_minus[j] |= (1 << i)
            else:
                # j -> i
                # i is in N^+(j), so j is in N^-(i)
                N_minus[i] |= (1 << j)
                
    # Check all 4-subsets
    # For S in C(n, 4):
    #   C = intersection of N^-(u) for u in S
    #   if C & ~mask(S) == 0: return False
    
    # Optimization: Precompute masks? No, just shift.
    # Iterating combinations
    
    for S in combinations(range(n), 4):
        u1, u2, u3, u4 = S
        c = N_minus[u1] & N_minus[u2] & N_minus[u3] & N_minus[u4]
        mask = (1 << u1) | (1 << u2) | (1 << u3) | (1 << u4)
        if (c & ~mask) == 0:
            return False
    return True

def _paley_orientation(p):
    squares = {x * x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            # i -> j iff (j - i) % p in squares
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Primes p ≡ 3 mod 4
    # Check small to large to minimize n
    primes = [31, 43, 47, 59, 67, 71, 79, 83]
    
    # We know 83 is valid (seed).
    # We want to find the smallest valid p.
    
    # If time is tight, skip large checks? No, we want smallest n.
    # So we must check 31 first.
    
    # However, if 31 fails, 43 might fail, etc.
    # Checking 31 is very fast.
    # Checking 83 is slower.
    
    for p in primes:
        if time.time() > deadline:
            break
            
        orientation = _paley_orientation(p)
        # Check validity
        # For small p, this is fast.
        # For p=83, it might take a few seconds.
        
        # We can optimize the check by early termination.
        # The check function does that.
        
        if _check_tournament(p, orientation):
            return {'n': p, 'orientation': orientation}
            
    # Fallback to 83 if time runs out or none found (though 83 should be found if time allows)
    # If we didn't check 83 due to timeout, we must return a valid result.
    # We know 83 is valid. We can return it without checking if we are in a hurry?
    # But if we didn't check it, we should ensure it's the one we return.
    
    # If the loop finished without returning, it means no p < 83 worked, or we timed out before checking 83.
    # If we timed out before checking 83, we should return 83's orientation directly to be safe.
    
    p = 83
    orientation = _paley_orientation(p)
    return {'n': p, 'orientation': orientation}
