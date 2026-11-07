"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    N = 210
    MAX_VAL = 10210
    
    # Precompute squarefree kernels
    sf = [0] * (MAX_VAL + 1)
    for i in range(1, MAX_VAL + 1):
        x = i
        kernel = 1
        d = 2
        while d * d <= x:
            count = 0
            while x % d == 0:
                x //= d
                count += 1
            if count % 2 == 1:
                kernel *= d
            d += 1
        if x > 1:
            kernel *= x
        sf[i] = kernel

    # Divisors of 210
    # 1, 2, 3, 5, 6, 7, 10, 14, 15, 21, 30, 35, 42, 70, 105, 210
    
    reachable = {1}
    parent = {1: (None, None)}  # state: (prev_state, number_used)
    
    for x in range(N + 1, MAX_VAL + 1):
        k = sf[x]
        # If k does not divide N, it can't be part of a product equal to N
        if N % k != 0:
            continue
            
        new_states = {}
        for p in list(reachable):
            q = p * k
            # q must divide N
            if N % q == 0:
                # Only add if q is not already reachable
                # This ensures we keep the path with the smallest max value
                if q not in reachable:
                    new_states[q] = (p, x)
                    
        # Update reachable and parent
        for q, (p, num) in new_states.items():
            reachable.add(q)
            parent[q] = (p, num)
            
        if N in reachable:
            # Backtrack to find the set of numbers
            selected = []
            current_state = N
            while current_state != 1:
                prev_state, num = parent[current_state]
                selected.append(num)
                current_state = prev_state
            # The selected list contains distinct integers > 210
            # Verify condition just in case
            prod = 1
            for num in selected:
                prod *= num
            # Check if prod * 210 is a perfect square
            total = prod * N
            root = math.isqrt(total)
            if root * root != total:
                # Should not happen, but fallback
                return [4 * N] 
            return selected
            
    # Fallback if no solution found (should not happen)
    return [4 * N]
# EVOLVE-BLOCK-END
