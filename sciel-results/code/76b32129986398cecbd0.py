import time
import random
import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_MILP = True
except ImportError:
    HAS_MILP = False

N = 128
MAX_TIME = 85.0

def _generate_forbidden_configs():
    """
    Generate all forbidden configurations.
    A configuration is a set of indices (0-based) corresponding to values in 1..128.
    We store them as bitsets for efficient checking.
    """
    configs = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s1 > N or s2 > N or s3 > N:
                    continue
                # The configuration is {a, b, c, s1, s2, s3}
                # Note: a,b,c are distinct. s1,s2,s3 might coincide with a,b,c or each other.
                # But the constraint is that ALL of them are in the set.
                # So we need x_a + x_b + x_c + x_s1 + x_s2 + x_s3 <= 5.
                # We represent this as a list of indices.
                # Indices are 0-based, so value v corresponds to index v-1.
                idxs = [a-1, b-1, c-1, s1-1, s2-1, s3-1]
                # We can deduplicate indices if any values coincide, but the sum constraint
                # still holds. If s1 == a, then x_a is counted twice?
                # No, the logical condition is x_a=1 AND x_b=1 AND x_c=1 AND x_s1=1 AND x_s2=1 AND x_s3=1.
                # If s1 == a, then the condition is x_a=1 AND ... which is just x_a=1.
                # The linear constraint x_a + x_b + x_c + x_s1 + x_s2 + x_s3 <= 5
                # becomes x_a + x_b + x_c + x_a + x_s2 + x_s3 <= 5 => 2*x_a + x_b + x_c + x_s2 + x_s3 <= 5.
                # This is correct because if x_a=1, the LHS is at least 2, so it allows the rest to be 1.
                # If x_a=0, LHS is at most 4, so it allows the rest to be 1.
                # Wait, if x_a=1, x_b=1, x_c=1, x_s2=1, x_s3=1, then LHS = 2+1+1+1+1 = 6 > 5. Correct.
                # So we just sum the variables for all 6 positions, even if indices repeat.
                configs.append(idxs)
    return configs

def _solve_milp(configs, time_limit=60.0):
    """
    Solve the MIS problem using ILP.
    """
    if not HAS_MILP:
        return None
    
    n_vars = N
    c = np.ones(n_vars)  # Maximize sum x_i
    
    # Objective is to maximize, milp minimizes. So we minimize -sum x_i.
    # Or we can just maximize by setting c to -1? No, milp minimizes c @ x.
    # So we set c = -1 to maximize sum x_i.
    c = -np.ones(n_vars)
    
    A_rows = []
    b_lows = []
    b_ups = []
    
    for idxs in configs:
        row = np.zeros(n_vars)
        for idx in idxs:
            row[idx] += 1
        A_rows.append(row)
        b_lows.append(0)
        b_ups.append(5)
        
    if not A_rows:
        # No constraints, take all
        return list(range(1, N+1))
        
    A = np.array(A_rows)
    b_l = np.array(b_lows)
    b_u = np.array(b_ups)
    
    constraints = LinearConstraint(A, b_l, b_u)
    integrality = np.ones(n_vars)
    bounds = Bounds(0, 1)
    
    start_time = time.time()
    try:
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": time_limit, "mip_rel_gap": 0.0}
        )
    except Exception:
        return None
        
    if res.status == 0:  # Optimal
        x = res.x
        # x is a numpy array of 0s and 1s (or close to it)
        selected = [int(i+1) for i in range(n_vars) if x[i] > 0.5]
        return selected
    else:
        # If not optimal but a feasible solution was found?
        # res.status 1 means "Optimal" for some versions? No, 0 is optimal.
        # Check if res.x is available.
        if res.x is not None:
            selected = [int(i+1) for i in range(n_vars) if res.x[i] > 0.5]
            return selected
        return None

def _fast_validity_check(s):
    """Check if the set s is valid (no forbidden triples)."""
    s_set = set(s)
    bitset = 0
    for x in s_set:
        bitset |= (1 << (x - 1))
    s_list = sorted(s_set)
    n = len(s_list)
    
    for i in range(n):
        a = s_list[i]
        if i + 1 >= n: break
        if a + s_list[i+1] > N: break
        for j in range(i+1, n):
            b = s_list[j]
            ab = a + b
            if ab > N: break
            if not ((bitset >> (ab - 1)) & 1):
                continue
            for k in range(j+1, n):
                c = s_list[k]
                if b + c > N: break
                if ((bitset >> (a + c - 1)) & 1) and ((bitset >> (b + c - 1)) & 1):
                    return False
    return True

def _make_bitset(s_set):
    bitset = 0
    for x in s_set:
        bitset |= (1 << (x - 1))
    return bitset

def _can_add(bitset, x):
    """
    Check if adding x to the set represented by bitset keeps it valid.
    """
    # Case 1: x is a member of the triple (a = x).
    # Need distinct b, c in S such that x+b, x+c, b+c are all in S U {x}.
    # V = { y in S | x+y in S }
    V_mask = bitset & (bitset >> x)
    if V_mask.bit_count() >= 2:
        v_list = []
        temp = V_mask
        while temp:
            lsb = temp & -temp
            v_list.append(lsb.bit_length())  # 1-indexed
            temp -= lsb
        n_v = len(v_list)
        for i in range(n_v):
            b = v_list[i]
            for j in range(i + 1, n_v):
                c = v_list[j]
                s = b + c
                if s == x or ((bitset >> (s - 1)) & 1):
                    return False

    # Case 2: x is a sum of two members (a+b = x).
    # Need distinct a, b, c in S with a+b=x, a+c in S, b+c in S.
    for a in range(1, x // 2 + 1):
        b = x - a
        if a >= b: continue
        if ((bitset >> (a - 1)) & 1) == 0:
            continue
        if ((bitset >> (b - 1)) & 1) == 0:
            continue
        # c in S \ {a,b} such that a+c in S and b+c in S
        U_mask = (bitset & (bitset >> a)) & (bitset & (bitset >> b))
        U_mask &= ~(1 << (a - 1))
        U_mask &= ~(1 << (b - 1))
        if U_mask != 0:
            return False

    return True

def _local_search(S_set, start_time, max_passes=50, swap_limit=20, deadline_margin=5.0):
    """Improve S via additions and systematic swaps using _can_add."""
    S_set = set(S_set)
    bitset = _make_bitset(S_set)
    not_in_S = [x for x in range(1, N+1) if x not in S_set]
    
    for _ in range(max_passes):
        if time.time() > start_time + MAX_TIME - deadline_margin:
            break

        improved = False
        random.shuffle(not_in_S)

        for x in not_in_S[:]:
            if time.time() > start_time + MAX_TIME - deadline_margin:
                break

            # Try to add x directly
            if _can_add(bitset, x):
                bitset |= (1 << (x - 1))
                S_set.add(x)
                not_in_S.remove(x)
                improved = True
                continue

            # Try swaps: remove y from S, add x
            S_list = list(S_set)
            random.shuffle(S_list)

            found_swap = False
            for y in S_list[:swap_limit]:
                if time.time() > start_time + MAX_TIME - deadline_margin:
                    break
                S_temp_mask = bitset ^ (1 << (y - 1))
                if _can_add(S_temp_mask, x):
                    bitset = S_temp_mask | (1 << (x - 1))
                    S_set.remove(y)
                    S_set.add(x)
                    not_in_S.append(y)
                    if x in not_in_S:
                        not_in_S.remove(x)
                    found_swap = True
                    improved = True
                    break

            if found_swap:
                break

        if not improved:
            break

    return S_set

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    best_s = []
    best_len = 0
    
    # 1. Try ILP
    if HAS_MILP:
        configs = _generate_forbidden_configs()
        ilp_result = _solve_milp(configs, time_limit=30.0)
        if ilp_result:
            if _fast_validity_check(ilp_result):
                best_s = ilp_result
                best_len = len(ilp_result)
    
    # 2. Local Search Fallback / Improvement
    # Start with a good initial set.
    # The top half (65-128) is a valid set of size 64.
    # But we can do better.
    # Let's start with the ILP result if available, otherwise a greedy set.
    
    if not best_s:
        # Greedy construction
        s_set = set()
        bitset = 0
        for x in range(1, N+1):
            if _can_add(bitset, x):
                bitset |= (1 << (x - 1))
                s_set.add(x)
        best_s = list(s_set)
        best_len = len(best_s)
    else:
        best_s = set(best_s)
        
    # Run local search to improve
    s_set = _local_search(best_s, start_time, max_passes=50, swap_limit=20, deadline_margin=5.0)
    if len(s_set) > best_len:
        best_s = s_set
        best_len = len(best_s)
        
    # Final validation
    best_s = list(best_s)
    if not _fast_validity_check(best_s):
        # Fallback to known valid set (top half)
        best_s = list(range(65, 129))
        
    return [int(x) for x in best_s]
