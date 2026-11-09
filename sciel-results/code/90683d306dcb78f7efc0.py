import time
import math

def construct():
    """
    Returns a list of distinct ints from 1..256, pairwise coprime, maximizing sum.
    Uses MILP (exact) with branch-and-bound fallback and local search polish.
    """
    start_time = time.time()
    deadline = start_time + 70  # Safety margin before 90s kill

    N = 256

    # Generate primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, math.isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]

    # Fallback solution: largest prime powers + 1
    fallback = [1]
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        fallback.append(power)

    # Primary: Try MILP (exact solution via totally unimodular constraint matrix)
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np

        n_vars = N
        n_constraints = len(primes)

        # Objective: maximize sum(i * x_i) -> minimize -sum(i * x_i)
        c = np.array([-float(i) for i in range(1, N + 1)])

        # Constraints: For each prime p, sum of x_i for i divisible by p <= 1
        A = np.zeros((n_constraints, n_vars))
        for idx_p, p in enumerate(primes):
            for i in range(1, N + 1):
                if i % p == 0:
                    A[idx_p, i - 1] = 1.0

        b = np.ones(n_constraints)

        # Bounds: 0 <= x_i <= 1
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))

        # Integrality: all variables are integer
        integrality = np.ones(n_vars)

        res = milp(c=c, constraints=LinearConstraint(A, ub=b), bounds=bounds, integrality=integrality)

        if res.status == 0:
            x = res.x
            result = [int(i + 1) for i in range(N) if x[i] > 0.5]
            # Verify pairwise coprime
            valid = True
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    if math.gcd(result[i], result[j]) != 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid and len(result) > 0:
                return [int(v) for v in result]
            else:
                # MILP returned invalid, fall through to B&B
                pass

    except Exception:
        pass

    # Secondary: Branch and bound Maximum Weight Independent Set
    # Build conflict graph: edge between i,j if gcd(i,j) > 1
    conflict = [0] * N
    for i in range(N):
        vi = i + 1
        for j in range(i + 1, N):
            vj = j + 1
            if math.gcd(vi, vj) > 1:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)

    weights = list(range(1, N + 1))

    # Greedy initial solution
    picked = 0
    greedy_sum = 0
    for i in range(N - 1, -1, -1):
        if (conflict[i] & picked) == 0:
            picked |= (1 << i)
            greedy_sum += weights[i]

    best_sum = greedy_sum
    best_mask = picked

    # Greedy coloring upper bound
    def color_bound(cand_mask):
        remaining = cand_mask
        color_classes = []
        while remaining:
            best_v = -1
            best_deg = -1
            best_w = -1
            r = remaining
            while r:
                lsb = r & (-r)
                v = lsb.bit_length() - 1
                deg_in_rem = bin(conflict[v] & remaining).count('1')
                w = weights[v]
                if deg_in_rem > best_deg or (deg_in_rem == best_deg and w > best_w):
                    best_deg = deg_in_rem
                    best_w = w
                    best_v = v
                r ^= lsb

            color_class_mask = 0
            color_class_weight = 0
            available = remaining
            while available:
                lsb = available & (-available)
                v = lsb.bit_length() - 1
                if (conflict[v] & color_class_mask) == 0:
                    color_class_mask |= lsb
                    color_class_weight += weights[v]
                    available ^= lsb
                else:
                    available ^= lsb

            color_classes.append(color_class_weight)
            remaining &= ~color_class_mask

        return color_classes

    def bb(cand_mask, cur_sum, cur_mask):
        nonlocal best_sum, best_mask

        if time.time() > deadline:
            return

        if cand_mask == 0:
            if cur_sum > best_sum:
                best_sum = cur_sum
                best_mask = cur_mask
            return

        color_classes = color_bound(cand_mask)
        ub = cur_sum + sum(color_classes)
        if ub <= best_sum:
            return

        best_v = -1
        best_deg = -1
        best_w = -1
        r = cand_mask
        while r:
            lsb = r & (-r)
            v = lsb.bit_length() - 1
            deg = bin(conflict[v] & cand_mask).count('1')
            w = weights[v]
            if deg > best_deg or (deg == best_deg and w > best_w):
                best_deg = deg
                best_w = w
                best_v = v
            r ^= lsb

        v_bit = 1 << best_v
        v_w = weights[best_v]

        new_cand = cand_mask & ~(v_bit | conflict[best_v])
        bb(new_cand, cur_sum + v_w, cur_mask | v_bit)

        if time.time() > deadline:
            return

        new_cand2 = cand_mask & ~v_bit
        bb(new_cand2, cur_sum, cur_mask)

    bb((1 << N) - 1, 0, 0)

    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)

    if 1 not in result:
        result.append(1)

    # Tertiary: Local search polish
    if time.time() < deadline - 5:
        result_set = set(result)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try single swaps: replace a smaller element with a larger compatible one
            for i in range(len(result)):
                a = result[i]
                for b in range(N, a, -1):
                    if b in result_set:
                        continue
                    compatible = True
                    for c in result:
                        if c == a:
                            continue
                        if math.gcd(b, c) != 1:
                            compatible = False
                            break
                    if compatible:
                        result_set.remove(a)
                        result_set.add(b)
                        result[i] = b
                        improved = True
                        break

    # Final validation
    final_result = [int(x) for x in sorted(result)]
    for i in range(len(final_result)):
        for j in range(i + 1, len(final_result)):
            if math.gcd(final_result[i], final_result[j]) != 1:
                return fallback
            if final_result[i] < 1 or final_result[i] > 256:
                return fallback
        if len(final_result) != len(set(final_result)):
            return fallback

    return final_result
