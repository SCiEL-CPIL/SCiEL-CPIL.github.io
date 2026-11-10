import time
from math import gcd

def construct():
    start_time = time.time()
    
    # Fixed input parameters
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    # Calculate LCM of divisors to determine periodicity
    # LCM(6, 10, 14, 15, 21, 22, 26, 33, 35, 39)
    # 6=2*3, 10=2*5, 14=2*7, 15=3*5, 21=3*7, 22=2*11, 26=2*13, 33=3*11, 35=5*7, 39=3*13
    # Primes: 2, 3, 5, 7, 11, 13
    lcm_val = 2 * 3 * 5 * 7 * 11 * 13  # 30030
    
    def get_candidates(S, L, d):
        """Get multiples of d in [S, S+L]"""
        # Smallest multiple >= S
        k = (S + d - 1) // d
        start_mult = k * d
        candidates = []
        val = start_mult
        limit = S + L
        while val <= limit:
            if val > 0 and val <= max_val:
                candidates.append(val)
            val += d
        return candidates

    def check_assignment(candidates):
        """
        Check if we can assign distinct values from candidates[i] to position i.
        candidates is a list of lists.
        Returns a list of assigned values or None.
        """
        assigned = [0] * n
        used = set()
        
        # Sort indices by number of candidates (MRV heuristic) to prune faster
        indices = list(range(n))
        indices.sort(key=lambda i: len(candidates[i]))
        
        def backtrack(idx):
            if idx == len(indices):
                return True
            
            i = indices[idx]
            for val in candidates[i]:
                if val not in used:
                    assigned[i] = val
                    used.add(val)
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
                    assigned[i] = 0
            return False
        
        if backtrack(0):
            return assigned
        return None

    def exists_window(L):
        """
        Check if there exists a window [S, S+L] with a valid distinct assignment.
        Returns the assignment if found, else None.
        """
        for S in range(1, lcm_val + 1):
            if time.time() - start_time > 80:
                return None
            
            candidates = []
            valid = True
            for d in divisors:
                c = get_candidates(S, L, d)
                if not c:
                    valid = False
                    break
                candidates.append(c)
            
            if not valid:
                continue
                
            # Check for distinct assignment
            res = check_assignment(candidates)
            if res is not None:
                return res
        
        return None

    # Binary search for minimum L
    # Minimum possible L is n-1 = 9
    low = 9
    high = 10000  # Safe upper bound. With L=10000, density is very high.
    
    best_L = None
    best_res = None
    
    def search_min_L(low, high):
        nonlocal best_L, best_res
        while low <= high:
            if time.time() - start_time > 75:
                break
            
            mid = (low + high) // 2
            res = exists_window(mid)
            if res is not None:
                best_L = mid
                best_res = res
                high = mid - 1
            else:
                low = mid + 1
        
        # If we broke due to time or low>high, we have best_res.
        # If best_res is None, we might have missed it or no solution in range.
        # Given constraints, a solution must exist.
        if best_res is None:
             # Fallback: try a larger range or linear scan from 9
             for L in range(9, 1000):
                 if time.time() - start_time > 80:
                     break
                 res = exists_window(L)
                 if res is not None:
                     best_L = L
                     best_res = res
                     break

    search_min_L(low, high)
    
    if best_res is None:
        # Emergency fallback: return the divisors themselves (which is the seed solution)
        # This is valid but not optimal.
        return divisors[:]
        
    # Ensure types are Python int
    # The values in best_res are Python ints from the lists generated in get_candidates.
    return best_res

# EVOLVE-BLOCK-END
