"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from math import gcd

def construct():
    """Return a list of distinct ints from 1..256, pairwise coprime, maximizing sum."""
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s kill
    N = 256
    
    # 1. Precompute prime factorization
    def get_prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    prime_factors = [set() for _ in range(N + 1)]
    for v in range(1, N + 1):
        prime_factors[v] = get_prime_factors(v)
    
    # conflicts[v] = set of u such that gcd(u, v) > 1
    conflicts = [set() for _ in range(N + 1)]
    for v in range(1, N + 1):
        for u in range(1, N + 1):
            if u != v and prime_factors[v] & prime_factors[u]:
                conflicts[v].add(u)
    
    # 2. Build initial solution using prime powers + known good replacements
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, math.isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power
        
    result = [1] + list(prime_powers.values())
    
    # Apply known beneficial replacements
    if 121 in result and 23 in result:
        result.remove(121); result.remove(23); result.append(253)
    if 49 in result and 31 in result:
        result.remove(49); result.remove(31); result.append(217)
    if 125 in result and 47 in result:
        result.remove(125); result.remove(47); result.append(235)
    if 81 in result and 83 in result:
        result.remove(81); result.remove(83); result.append(249)
    if 169 in result and 19 in result:
        result.remove(169); result.remove(19); result.append(247)
    
    current_set = set(result)
    current_sum = sum(result)
    best_set = set(result)
    best_sum = current_sum
    
    # 3. Try ILP with scipy if available using prime-based constraints
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix
        import numpy as np
        
        # Variables: x_i for i in 1..N, binary
        # Maximize sum(i * x_i)
        # Constraints: for each prime p <= N, sum of x_k for k multiple of p <= 1
        
        # Identify all primes <= N
        primes_list = [p for p in range(2, N + 1) if is_prime[p]]
        
        # Build constraint matrix
        # Rows correspond to primes, columns to numbers 1..N (index 0..N-1)
        constraint_rows = []
        constraint_cols = []
        constraint_vals = []
        
        for p_idx, p in enumerate(primes_list):
            for k in range(p, N + 1, p):
                constraint_rows.append(p_idx)
                constraint_cols.append(k - 1)
                constraint_vals.append(1)
        
        if constraint_rows:
            A = csr_matrix((constraint_vals, (constraint_rows, constraint_cols)), shape=(len(primes_list), N))
            c = np.array([-i for i in range(1, N+1)], dtype=float)  # Minimize negative sum
            constraints = LinearConstraint(A, ub=1)
            bounds = Bounds(0, 1)
            integrality = np.ones(N)
            
            # Set time limit for the solver to 50 seconds to be safe
            # Note: milp in scipy does not directly take a time limit argument in all versions,
            # but we can rely on the overall deadline check or just let it run if it's fast.
            # However, HiGHS can be slow. We will catch any exceptions.
            
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if res.success and res.x is not None:
                ilp_set = set()
                for i, val in enumerate(res.x):
                    if val > 0.5:
                        ilp_set.add(i + 1)
                ilp_sum = sum(ilp_set)
                
                # Verify pairwise coprime
                valid = True
                ilp_list = list(ilp_set)
                for i in range(len(ilp_list)):
                    for j in range(i+1, len(ilp_list)):
                        if gcd(ilp_list[i], ilp_list[j]) != 1:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid and ilp_sum > best_sum:
                    best_sum = ilp_sum
                    best_set = ilp_set
                    current_set = ilp_set
                    current_sum = ilp_sum
    except:
        pass
    
    # 4. Local search refinement on the best solution found
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding a single element
            for v in range(N, 0, -1):
                if v not in current_set and not (current_set & conflicts[v]):
                    current_set.add(v)
                    current_sum += v
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_set = set(current_set)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-for-1 swaps
            selected_list = list(current_set)
            unselected_list = [v for v in range(1, N+1) if v not in current_set]
            found_swap = False
            for u in sorted(selected_list):
                if found_swap:
                    break
                for v in sorted(unselected_list, reverse=True):
                    if v <= u:
                        break
                    common = current_set & conflicts[v]
                    if not common or common == {u}:
                        current_set.remove(u)
                        current_set.add(v)
                        current_sum += (v - u)
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_set = set(current_set)
                        found_swap = True
                        break
            
            if found_swap:
                improved = True
                continue
            
            # Try 2-for-1 swaps (remove 2, add 1 larger)
            n = len(current_set)
            sel = list(current_set)
            found_21 = False
            for i in range(n):
                if found_21:
                    break
                for j in range(i+1, n):
                    a, b = sel[i], sel[j]
                    remaining = current_set - {a, b}
                    for x in range(N, 0, -1):
                        if x in current_set:
                            continue
                        if not (remaining & conflicts[x]):
                            gain = x - a - b
                            if gain > 0:
                                current_set.remove(a)
                                current_set.remove(b)
                                current_set.add(x)
                                current_sum += gain
                                if current_sum > best_sum:
                                    best_sum = current_sum
                                    best_set = set(current_set)
                                found_21 = True
                                break
                    if found_21:
                        break
            
            if found_21:
                improved = True
                continue
            
            # Try 1-for-2 swaps (remove 1, add 2 coprime)
            found_12 = False
            for u in sorted(current_set):
                if found_12:
                    break
                S_minus_u = current_set - {u}
                addable = [v for v in range(N, 0, -1) if v not in current_set and not (S_minus_u & conflicts[v])]
                if len(addable) >= 2:
                    best_pair = None
                    best_gain = 0
                    for i in range(len(addable)):
                        v1 = addable[i]
                        if v1 + addable[i+1] - u <= best_gain:
                            break
                        for j in range(i+1, len(addable)):
                            v2 = addable[j]
                            if v1 + v2 - u <= best_gain:
                                break
                            if gcd(v1, v2) == 1:
                                gain = v1 + v2 - u
                                if gain > best_gain:
                                    best_gain = gain
                                    best_pair = (v1, v2)
                    if best_pair and best_gain > 0:
                        v1, v2 = best_pair
                        current_set.remove(u)
                        current_set.add(v1)
                        current_set.add(v2)
                        current_sum += best_gain
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_set = set(current_set)
                        found_12 = True
                        improved = True
    
    # 5. Final high-precision polish: exhaustive single-element improvement scan
    if time.time() < deadline:
        # Ensure we're working from the best known set
        current_set = set(best_set)
        current_sum = best_sum
        
        # Multiple passes of exhaustive 1-for-1 and 2-for-1 improvements
        for _pass in range(3):
            if time.time() > deadline:
                break
            
            changed = True
            while changed and time.time() < deadline:
                changed = False
                
                # Exhaustive 1-for-1: for each element u, try replacing with best v > u
                sel_list = sorted(current_set)
                unsel_list = sorted([v for v in range(1, N+1) if v not in current_set], reverse=True)
                
                for idx_u, u in enumerate(sel_list):
                    if time.time() > deadline:
                        break
                    best_v = None
                    best_gain = 0
                    for v in unsel_list:
                        if v <= u:
                            break
                        common = current_set & conflicts[v]
                        if not common or common == {u}:
                            gain = v - u
                            if gain > best_gain:
                                best_gain = gain
                                best_v = v
                    
                    if best_v is not None and best_gain > 0:
                        current_set.remove(u)
                        current_set.add(best_v)
                        current_sum += best_gain
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_set = set(current_set)
                        changed = True
                        break  # Restart scan after change
            
            # Exhaustive 2-for-1: for each pair (a,b), try replacing with best single x
            if time.time() < deadline:
                changed = True
                while changed and time.time() < deadline:
                    changed = False
                    sel_list = sorted(current_set)
                    n = len(sel_list)
                    
                    for i in range(n):
                        if time.time() > deadline:
                            break
                        a = sel_list[i]
                        for j in range(i+1, n):
                            if time.time() > deadline:
                                break
                            b = sel_list[j]
                            remaining = current_set - {a, b}
                            best_x = None
                            best_gain = 0
                            for x in range(N, 0, -1):
                                if x in current_set:
                                    continue
                                if not (remaining & conflicts[x]):
                                    gain = x - a - b
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_x = x
                            
                            if best_x is not None and best_gain > 0:
                                current_set.remove(a)
                                current_set.remove(b)
                                current_set.add(best_x)
                                current_sum += best_gain
                                if current_sum > best_sum:
                                    best_sum = current_sum
                                    best_set = set(current_set)
                                changed = True
                                break
                        if changed:
                            break
    
    # Final verification
    result = list(best_set)
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # Fallback
                return [1] + list(prime_powers.values())
    
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
