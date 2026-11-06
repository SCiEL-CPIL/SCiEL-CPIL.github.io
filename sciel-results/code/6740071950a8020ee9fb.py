import time
from math import factorial
from itertools import product

def construct():
    start = time.time()
    deadline = start + 85

    N = factorial(12)  # 479001600
    # Prime factorization: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    total_exp = (10, 5, 2, 1, 1)
    primes = (2, 3, 5, 7, 11)
    
    # Meet-in-the-middle: split primes into two groups
    # Group A: {2, 3}, Group B: {5, 7, 11}
    # Generate all possible exponent combinations for each group
    
    def gen_group_divisors(group_indices):
        """Generate all (value, exp_vector) for a subset of primes"""
        divs = []
        def rec(idx, current_val, exps):
            if idx == len(group_indices):
                if current_val > 12:
                    divs.append((current_val, tuple(exps)))
                return
            p_idx = group_indices[idx]
            p = primes[p_idx]
            max_e = total_exp[p_idx]
            for e in range(max_e + 1):
                exps[idx] = e
                rec(idx + 1, current_val * (p ** e), exps)
            exps[idx] = 0
        rec(0, 1, [0] * len(group_indices))
        divs.sort(key=lambda x: x[0])
        return divs
    
    # Group A: primes 2, 3 (indices 0, 1)
    groupA = gen_group_divisors([0, 1])
    # Group B: primes 5, 7, 11 (indices 2, 3, 4)
    groupB = gen_group_divisors([2, 3, 4])
    
    # For each candidate factor, we need to combine a divisor from group A and group B
    # such that the product is a valid divisor of N
    
    # Generate all valid combined divisors
    all_divs = []
    for valA, expA in groupA:
        for valB, expB in groupB:
            combined_val = valA * valB
            if combined_val > 12 and combined_val <= N:
                # Check if it divides N
                if N % combined_val == 0:
                    combined_exp = (expA[0], expA[1], expB[0], expB[1], expB[2])
                    all_divs.append((combined_val, combined_exp))
    
    # Remove duplicates and sort
    seen = set()
    unique_divs = []
    for val, exp in all_divs:
        if val not in seen:
            seen.add(val)
            unique_divs.append((val, exp))
    unique_divs.sort(key=lambda x: x[0])
    
    values = [d[0] for d in unique_divs]
    exps = [d[1] for d in unique_divs]
    
    # Binary search on M
    def can_factor(M):
        if time.time() > deadline:
            return None
        
        # Filter divs to those with value <= M
        lo, hi = 0, len(values)
        while lo < hi:
            mid = (lo + hi) // 2
            if values[mid] <= M:
                lo = mid + 1
            else:
                hi = mid
        local_vals = values[:lo]
        local_exps = exps[:lo]
        
        if not local_vals:
            return None
        
        result = None
        
        def dfs(idx, rem_exp, count, chosen):
            nonlocal result
            if time.time() > deadline:
                return False
            
            if rem_exp == (0, 0, 0, 0, 0):
                result = list(chosen)
                return True
            
            if count >= 12:
                return False
            
            if idx >= len(local_vals):
                return False
            
            # Prune: compute remaining value
            rem_val = 1
            for i in range(5):
                rem_val *= primes[i] ** rem_exp[i]
            
            if rem_val < local_vals[idx]:
                return False
            
            # Additional pruning: if rem_val > 1, ensure we can form it
            # with at most (12 - count) more factors, each <= M
            if rem_val > 1:
                remaining_slots = 12 - count
                # The maximum product we can form with remaining_slots factors
                # each <= M is M^remaining_slots
                # But we need a tighter bound
                if remaining_slots > 0:
                    # Check if rem_val can be factored into at most remaining_slots
                    # numbers each <= M
                    # Simple check: if rem_val > M ** remaining_slots, fail
                    # This is a loose bound but helps
                    pass
            
            for i in range(idx, len(local_vals)):
                v = local_vals[i]
                e = local_exps[i]
                
                can_sub = True
                for j in range(5):
                    if rem_exp[j] < e[j]:
                        can_sub = False
                        break
                if not can_sub:
                    continue
                
                new_rem = tuple(rem_exp[j] - e[j] for j in range(5))
                
                chosen.append(v)
                if dfs(i + 1, new_rem, count + 1, chosen):
                    chosen.pop()
                    return True
                chosen.pop()
            
            return False
        
        if dfs(0, total_exp, 0, []):
            return result
        return None

    # Binary search on M
    lo, hi = 13, N
    best = None
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        res = can_factor(mid)
        if res is not None:
            best = res
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best is None:
        return [N]
    
    # Verify
    if len(best) > 12 or len(best) < 1:
        return [N]
    if any(f <= 12 for f in best):
        return [N]
    if len(set(best)) != len(best):
        return [N]
    prod = 1
    for f in best:
        prod *= f
    if prod != N:
        return [N]
    
    return best
