"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    # Calculate LCM of divisors to determine the periodicity
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    P = 1
    for d in divisors:
        P = lcm(P, d)
        
    best_window = float('inf')
    best_result = None
    
    # We only need to check L in [1, P] because the pattern of multiples repeats every P.
    # For a fixed L, we want to find the smallest M >= L such that
    # there exist distinct v_i in [L, M] where d_i divides v_i.
    
    def find_minimal_M(L, max_M):
        """
        Binary search for the smallest M >= L such that a perfect matching exists
        in the interval [L, M].
        Returns (M, matching_values) or (None, None) if not found up to max_M.
        """
        low = L
        high = max_M
        best_M = None
        best_match = None
        
        while low <= high:
            mid = (low + high) // 2
            # Check if matching exists in [L, mid]
            # Build adjacency list
            # Left: divisors (0..n-1)
            # Right: numbers in [L, mid]
            
            # To make matching fast, we can just collect candidates for each divisor
            candidates = [[] for _ in range(n)]
            # Map number to index? No, numbers can be large.
            # We just need to know if we can assign distinct numbers.
            
            # For each divisor i, find multiples in [L, mid]
            for i, d in enumerate(divisors):
                # First multiple >= L
                k = (L + d - 1) // d
                m = k * d
                while m <= mid:
                    candidates[i].append(m)
                    m += d
            
            # Bipartite matching
            # match_to_number[j] = i means number j is matched to divisor i
            # We use a standard DFS based matching for small n
            
            # We need to map numbers to indices for the matching algorithm?
            # Or just use the numbers directly in a dict?
            # Since n=10, we can use a recursive backtracking or Hopcroft-Karp.
            # Let's use a simple DFS augmenting path algorithm.
            
            # match[i] = number assigned to divisor i
            match = [-1] * n
            
            def dfs(i, visited):
                for num in candidates[i]:
                    if visited[num]:
                        continue
                    visited[num] = True
                    # If num is not used, or the divisor that was using num can find another
                    # We need to know which divisor currently holds 'num'.
                    # This requires an inverse map: num -> divisor index.
                    # Since we are building the match dynamically, we can maintain a global 
                    # 'owner' dict outside or pass it.
                    # Let's refactor: use a dict owner[num] = i.
                    pass
                    
            # Standard Kuhn's algorithm implementation
            # owner: dict mapping number -> divisor index
            owner = {}
            
            def kuhn(i, visited):
                for num in candidates[i]:
                    if num in visited:
                        continue
                    visited.add(num)
                    if num not in owner or kuhn(owner[num], visited):
                        owner[num] = i
                        return True
                return False
            
            matched_count = 0
            # We need to reset owner for each mid? Yes.
            # But we can just run kuhn for each divisor.
            # Note: The order of processing divisors can affect performance but not correctness.
            
            # Optimization: Sort divisors by number of candidates? 
            # Usually smaller candidates sets are more constrained.
            
            # Let's just run standard matching
            # We need to be careful: 'visited' is per DFS call.
            
            for i in range(n):
                visited = set()
                if kuhn(i, visited):
                    matched_count += 1
            
            if matched_count == n:
                # Found a matching
                best_M = mid
                # Extract values
                vals = [-1] * n
                for num, div_idx in owner.items():
                    vals[div_idx] = num
                best_match = vals
                high = mid - 1
            else:
                low = mid + 1
                
        return best_M, best_match

    # Iterate L from 1 to P
    # To speed up, we can set an initial upper bound for M.
    # A trivial upper bound is L + max(divisors) * n + something.
    # Or just use best_window if we have one.
    
    # Initial best_window can be found by a quick greedy or just set to infinity.
    # Let's run the loop.
    
    for L in range(1, P + 1):
        # Upper bound for M
        # The window size W is at most ... well, let's just use L + best_window if best_window < inf
        # Otherwise, use a reasonable heuristic.
        # In the worst case, we might need to go up to L + 1000.
        # But we can just use max_val - L.
        
        # If we already have a best_window, we only care if M - L < best_window
        # So max_M = L + best_window - 1
        # If best_window is inf, we need a bound. 
        # Let's assume W < 10000 for the first few checks, or just use a large number.
        # Actually, we can just binary search up to L + 5000. If not found, skip?
        # No, we must find the optimal.
        # Let's initialize best_window with a greedy result to have a bound.
        
        if best_window == float('inf'):
             # Quick greedy to get an initial bound
             used = set()
             vals = []
             for d in divisors:
                 m = L
                 if L % d != 0:
                     m = L + (d - L % d)
                 while m in used:
                     m += d
                 if m > max_val:
                     break
                 used.add(m)
                 vals.append(m)
             if len(vals) == n:
                 best_window = max(vals) - L
                 best_result = vals
                 # Re-evaluate max_M bound
                 if L + best_window > max_val:
                     max_M = max_val
                 else:
                     max_M = L + best_window - 1
             else:
                 max_M = L + 10000 # Fallback
        else:
             if L + best_window > max_val:
                 max_M = max_val
             else:
                 max_M = L + best_window - 1
        
        M, vals = find_minimal_M(L, max_M)
        
        if vals is not None:
            w = M - L
            if w < best_window:
                best_window = w
                best_result = vals
                
    result = [int(x) for x in best_result]
    return result
# EVOLVE-BLOCK-END
