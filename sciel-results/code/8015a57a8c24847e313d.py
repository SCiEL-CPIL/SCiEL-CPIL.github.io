# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import time

def _square_free_up_to(n):
    """Return list of square-free numbers up to n."""
    # Sieve for smallest prime factor
    spf = list(range(n + 1))
    for i in range(2, int(n**0.5) + 1):
        if spf[i] == i:  # prime
            for j in range(i*i, n+1, i):
                if spf[j] == j:
                    spf[j] = i
    result = []
    for k in range(1, n+1):
        # Check if square-free: no prime factor appears more than once
        x = k
        seen = set()
        is_sf = True
        while x > 1:
            p = spf[x]
            if p in seen:
                is_sf = False
                break
            seen.add(p)
            x //= p
            # Check if p divides x again
            if x % p == 0:
                is_sf = False
                break
        if is_sf:
            result.append(k)
    return result


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k)*log(k)/k subject to
    M(x) = sum_k f(k)*floor(x/k) <= 1 for all x in [1, 10K].
    """
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin before 630s kill

    # Step 1: Choose domain as square-free numbers up to B
    B = 3000
    sf_list = _square_free_up_to(B)
    m = len(sf_list)
    domain = np.array(sf_list, dtype=np.int64)
    
    # Objective coefficients: c_k = -log(k)/k (we maximize sum c_k * f_k)
    logs = np.log(domain.astype(np.float64))
    c = -logs / domain.astype(np.float64)
    
    # Step 2: Generate test points for M(x) constraints
    # M(x) = sum_k f_k * floor(x/k)
    # Critical x values: multiples of domain elements up to 10*B
    # But we need to be selective. Use x values that are multiples of small domain elements.
    # For efficiency, use x in [1, 10*B] that are "interesting"
    
    max_x = 10 * B
    # Generate test points: all x that are multiples of any domain element <= 100,
    # plus all domain elements themselves, plus some random sampling
    test_points = set()
    
    # All domain elements
    for k in domain:
        test_points.add(int(k))
    
    # Multiples of small domain elements
    small_domain = [k for k in domain if k <= 50]
    for k in small_domain:
        for mult in range(1, min(10*B // k + 1, 200)):
            x = k * mult
            if x <= max_x:
                test_points.add(x)
    
    # Also add x = 1
    test_points.add(1)
    
    # Add some evenly spaced points
    for x in range(1, max_x + 1, max(1, max_x // 5000)):
        test_points.add(x)
    
    test_points = sorted(test_points)
    
    # Limit test points to keep LP manageable
    max_test_points = 8000
    if len(test_points) > max_test_points:
        # Keep the most critical ones: small x and multiples of small k
        test_points_set = set(test_points)
        # Prioritize: x <= 100, then multiples of domain elements <= 20
        critical = []
        for x in test_points:
            if x <= 200:
                critical.append(x)
        for k in domain:
            if k <= 20:
                for mult in range(1, max_x // k + 1):
                    x = k * mult
                    if x <= max_x and x not in critical:
                        critical.append(x)
        critical = sorted(set(critical))
        if len(critical) > max_test_points:
            critical = critical[:max_test_points]
        # Add some random from remaining
        remaining = [x for x in test_points if x not in set(critical)]
        if remaining and len(critical) < max_test_points:
            n_add = min(len(remaining), max_test_points - len(critical))
            step = max(1, len(remaining) // n_add)
            critical.extend(remaining[::step][:n_add])
            critical = sorted(set(critical))
        test_points = critical[:max_test_points]
    
    n_tp = len(test_points)
    test_points_arr = np.array(test_points, dtype=np.int64)
    
    # Step 3: Build constraint matrix A such that A @ f <= b
    # A[i, j] = floor(test_points[i] / domain[j])
    # b[i] = 1 for all i
    # Also add bounds: -10 <= f_k <= 10
    
    # Compute A efficiently
    A = np.zeros((n_tp, m), dtype=np.float64)
    for j, k in enumerate(domain):
        A[:, j] = test_points_arr // k
    
    b = np.ones(n_tp, dtype=np.float64)
    
    # Step 4: Solve LP
    # Maximize c @ f subject to A @ f <= b, -10 <= f <= 10
    # linprog minimizes, so minimize -c @ f
    c_neg = -c
    
    bounds = [(-10.0, 10.0)] * m
    
    # Use HiGHS solver if available, else interior-point
    try:
        res = linprog(c_neg, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    except Exception:
        try:
            res = linprog(c_neg, A_ub=A, b_ub=b, bounds=bounds, method='interior-point')
        except Exception:
            res = None
    
    if res is None or not res.success:
        # Fallback: use Möbius function values
        # mu(n) for square-free n: (-1)^number_of_prime_factors
        values = np.zeros(m, dtype=np.float64)
        for j, k in enumerate(domain):
            # Count prime factors
            x = k
            cnt = 0
            p = 2
            while p * p <= x:
                while x % p == 0:
                    cnt += 1
                    x //= p
                p += 1
            if x > 1:
                cnt += 1
            values[j] = 1.0 if cnt % 2 == 0 else -1.0
        return domain, values
    
    f_opt = res.x
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Return what we have
        return domain, f_opt
    
    # Step 5: Verify feasibility and compute objective
    # Check M(x) <= 1 + tolerance for test points
    M_vals = A @ f_opt
    max_M = np.max(M_vals)
    
    if max_M > 1.001:
        # Infeasible or solver issue, scale down
        scale = 1.0 / max_M if max_M > 0 else 1.0
        f_opt = f_opt * scale * 0.999
    
    # Compute objective a = -sum f(k) * log(k) / k = c @ f
    a_val = np.dot(c, f_opt)
    
    return domain, f_opt


# EVOLVE-BLOCK-END
