import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start = time.time()
    deadline = start + 80.0
    N = 128
    
    # Generate forbidden sets
    # A set is forbidden if it contains a,b,c and a+b, a+c, b+c
    # We iterate a < b < c such that b+c <= N
    forbidden_sets = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            # c > b and b+c <= N
            max_c = N - b
            if max_c <= b:
                continue
            for c in range(b + 1, max_c + 1):
                # Check if sums are in domain
                s1 = a + b
                s2 = a + c
                s3 = b + c
                # s3 is max, and we ensured s3 <= N
                # s1, s2 < s3 so they are in domain
                vals = {a, b, c, s1, s2, s3}
                forbidden_sets.append(vals)
    
    # If no forbidden sets, return all
    if not forbidden_sets:
        return list(range(1, N + 1))
    
    # Setup MILP
    # x_i is 0 or 1
    n_vars = N
    c = -np.ones(n_vars) # Maximize sum x_i => Minimize -sum x_i
    
    # Constraints:
    # 1. 0 <= x_i <= 1 (handled by integrality and bounds)
    # 2. For each forbidden set V, sum_{i in V} x_i <= |V| - 1
    
    # We can use LinearConstraint
    # A matrix of size (num_constraints, n_vars)
    # We build it row by row
    
    # Number of constraints is len(forbidden_sets)
    # We might have duplicate sets. We should unique them to reduce constraints.
    unique_forbidden = list(set(tuple(sorted(f)) for f in forbidden_sets))
    
    if not unique_forbidden:
        return list(range(1, N + 1))
        
    A = np.zeros((len(unique_forbidden), n_vars))
    b_ub = np.zeros(len(unique_forbidden))
    
    for i, f_set in enumerate(unique_forbidden):
        for v in f_set:
            A[i, v - 1] = 1.0
        b_ub[i] = len(f_set) - 1
        
    # integrality: 1 for all variables
    integrality = np.ones(n_vars)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # LinearConstraint: A @ x <= b_ub
    # Also x >= 0 is in bounds
    constraints = LinearConstraint(A, ub=b_ub)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            # res.x contains the solution
            # res.x might have small floating point errors
            sol = np.round(res.x).astype(int)
            # Ensure validity just in case
            # The solution should be valid by construction of constraints
            
            result = [i + 1 for i in range(n_vars) if sol[i] == 1]
            
            # Verify validity (safety check)
            # If invalid, fallback to a simple valid set
            if not _is_valid(result, N):
                 # Fallback: The interval [63, 128] is always valid
                 # Let's check size
                 fallback = list(range(63, 129))
                 if len(fallback) > len(result):
                     return fallback
                 # Or try the local search from the old code
                 return _local_search_fallback(N, start, deadline)
            
            return result
            
        else:
            # If MILP fails or times out, use fallback
            return _local_search_fallback(N, start, deadline)
            
    except Exception:
        return _local_search_fallback(N, start, deadline)

def _is_valid(S, N):
    S_set = set(S)
    # Check for forbidden triples
    # a < b < c
    # a+b, a+c, b+c in S
    # We can just iterate all triples
    # O(N^3) is 128^3 ~ 2M, which is fine for one check
    for i in range(len(S)):
        a = S[i]
        for j in range(i + 1, len(S)):
            b = S[j]
            for k in range(j + 1, len(S)):
                c = S[k]
                if (a + b) in S_set and (a + c) in S_set and (b + c) in S_set:
                    return False
    return True

def _local_search_fallback(N, start, deadline):
    # Reuse the logic from the current program as a fallback
    # This ensures we always return *something* valid and decent
    # The current program's logic is robust.
    
    def time_left():
        return deadline - time.time()

    def can_add(mem, inS, x):
        for i in range(len(mem)):
            a = mem[i]
            xa = x + a
            if xa > N or not inS[xa]:
                continue
            for j in range(i + 1, len(mem)):
                b = mem[j]
                xb = x + b
                ab = a + b
                if xb > N or not inS[xb]:
                    continue
                if ab > N:
                    continue
                if inS[ab] or ab == x:
                    return False
        for i in range(len(mem)):
            a = mem[i]
            b = x - a
            if b <= 0 or b > N or b == a:
                continue
            if not inS[b]:
                continue
            for c in mem:
                if c == a or c == b:
                    continue
                ac = a + c
                bc = b + c
                ac_ok = (ac <= N and inS[ac]) or ac == x
                bc_ok = (bc <= N and inS[bc]) or bc == x
                if ac_ok and bc_ok:
                    return False
        return True

    def full_valid(mem):
        s = set(mem)
        m = sorted(mem)
        n = len(m)
        for i in range(n):
            a = m[i]
            for j in range(i + 1, n):
                b = m[j]
                ab = a + b
                if ab not in s:
                    continue
                for k in range(j + 1, n):
                    c = m[k]
                    if (a + c) in s and (b + c) in s:
                        return False
        return True

    def local_search(mem, inS):
        rng = random.Random(987654321)
        for _ in range(80):
            if time_left() < 4.0:
                break
            added = False
            for x in range(1, N + 1):
                if time_left() < 4.0:
                    break
                if inS[x]:
                    continue
                if can_add(mem, inS, x):
                    mem.append(x)
                    inS[x] = 1
                    added = True
            if added:
                continue
            sw = False
            for _ in range(80):
                if time_left() < 4.0:
                    break
                rm = rng.choice(mem)
                add = rng.randint(1, N)
                if inS[add]:
                    continue
                inS[rm] = 0
                mem.remove(rm)
                if can_add(mem, inS, add):
                    mem.append(add)
                    inS[add] = 1
                    sw = True
                    break
                else:
                    mem.append(rm)
                    inS[rm] = 1
            if not sw:
                break
        return mem

    def solve_restart(order):
        mem = [i for i in range(N // 2 + 1, N + 1)]
        inS = bytearray(N + 1)
        for i in mem:
            inS[i] = 1
        for x in order:
            if time_left() < 4.0:
                break
            if inS[x]:
                continue
            if can_add(mem, inS, x):
                mem.append(x)
                inS[x] = 1
        local_search(mem, inS)
        return mem

    base = [i for i in range(N // 2 + 1, N + 1)]
    best = base
    best_size = len(base)

    small = list(range(1, N // 2 + 1))
    orders = [list(small), list(reversed(small))]
    rng = random.Random(987654321)
    for _ in range(20):
        o = list(small)
        rng.shuffle(o)
        orders.append(o)

    for o in orders:
        if time_left() < 8.0:
            break
        mem = solve_restart(o)
        if len(mem) > best_size and full_valid(mem):
            best = mem
            best_size = len(mem)

    if time_left() > 10.0:
        inS = bytearray(N + 1)
        for i in best:
            inS[i] = 1
        mem = list(best)
        local_search(mem, inS)
        if len(mem) > best_size and full_valid(mem):
            best = mem
            best_size = len(mem)

    if not full_valid(best):
        best = base

    return [int(x) for x in best]
