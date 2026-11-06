"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin within 90s limit
    
    N = 512
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    def get_divisors(n):
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return divs
    
    # Precompute LCM table for fast lookups
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            l = i * j // g
            lcm_table[i][j] = l
            lcm_table[j][i] = l
    
    # Precompute compatibility bitsets
    # compat[i] has bit j set if lcm(i,j) <= N
    compat = [0] * (N + 1)
    for i in range(1, N + 1):
        bits = 0
        for j in range(1, N + 1):
            if lcm_table[i][j] <= N:
                bits |= (1 << j)
        compat[i] = bits
    
    best_set = set()
    best_size = 0
    
    def is_compatible_with_all(x, current_bits):
        """Check if x is compatible with all elements in current_bits."""
        return (compat[x] & current_bits) == current_bits
    
    def greedy_expand(initial_set):
        """Greedy expansion using bitsets for speed."""
        current = set(initial_set)
        current_bits = 0
        for x in current:
            current_bits |= (1 << x)
        
        # Try adding elements in ascending order (smaller numbers tend to be more compatible)
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if x not in current and is_compatible_with_all(x, current_bits):
                current.add(x)
                current_bits |= (1 << x)
        
        return current
    
    def local_search_swap(current, max_passes=5):
        """Local search with swap moves: remove one element, try to add another."""
        current = set(current)
        improved = True
        passes = 0
        
        while improved and passes < max_passes:
            if time.time() > deadline:
                break
            improved = False
            passes += 1
            
            current_bits = 0
            for x in current:
                current_bits |= (1 << x)
            
            # Try swapping each element with a non-current element
            for x in list(current):
                if time.time() > deadline:
                    break
                
                # Remove x
                without_x_bits = current_bits & ~(1 << x)
                
                # Find best replacement: an element not in current that is compatible with all others
                # and has higher "value" (we just want any valid swap that doesn't reduce size)
                # Actually, for size maximization, we want to find an element y not in current
                # such that y is compatible with all elements in current\{x}
                # Then we can swap x for y, maintaining size
                # But we want to eventually grow, so this is mainly for escaping local optima
                
                # Try to add a new element after removing x
                for y in range(1, N + 1):
                    if y in current:
                        continue
                    if is_compatible_with_all(y, without_x_bits):
                        # Swap x for y
                        current.remove(x)
                        current.add(y)
                        improved = True
                        break
                
                if improved:
                    # Rebuild bits
                    current_bits = 0
                    for x in current:
                        current_bits |= (1 << x)
                    break
        
        # After swaps, try to add more elements
        current_bits = 0
        for x in current:
            current_bits |= (1 << x)
        
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if x not in current and is_compatible_with_all(x, current_bits):
                current.add(x)
                current_bits |= (1 << x)
        
        return current
    
    # Strategy 1: Try divisor sets of all M in 1..N
    for M in range(1, N + 1):
        if time.time() > deadline:
            break
        
        base = get_divisors(M)
        current = greedy_expand(set(base))
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current
    
    # Strategy 2: Local search from the best set found
    if best_set and time.time() < deadline:
        improved_set = local_search_swap(best_set, max_passes=3)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
    
    # Strategy 3: Try greedy from individual starting points for elements not in best_set
    if time.time() < deadline:
        for start_elem in range(1, N + 1):
            if time.time() > deadline:
                break
            
            # Skip if start_elem is already in best_set (we've likely covered it)
            S = {start_elem}
            S_bits = 1 << start_elem
            
            # Greedy expansion
            for x in range(1, N + 1):
                if time.time() > deadline:
                    break
                if x not in S and is_compatible_with_all(x, S_bits):
                    S.add(x)
                    S_bits |= (1 << x)
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
    
    # Final local search from best set
    if best_set and time.time() < deadline:
        improved_set = local_search_swap(best_set, max_passes=5)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
    
    # Final verification
    best_list = sorted(best_set)
    for i in range(len(best_list)):
        for j in range(i + 1, len(best_list)):
            if lcm_table[best_list[i]][best_list[j]] > N:
                # Fallback
                return list(range(1, 23))
    
    if not best_list:
        return [1]
    
    return best_list
# EVOLVE-BLOCK-END
