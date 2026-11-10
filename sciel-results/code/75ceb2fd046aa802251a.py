# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Uses a Linear Program formulation to find the optimal f on a chosen domain.
    """
    t_start = time.time()
    deadline = t_start + 580.0  # Safety margin

    # Domain size to use for the LP.
    # Reduced K to 100 to ensure fast solution time and reliability.
    # K=100 results in 1000 constraints and 100 variables, which is very fast for HiGHS.
    K = 100

    domain = np.arange(1, K + 1, dtype=np.int64)

    # Objective: Maximize a = -sum f(k) * log(k) / k
    # This is equivalent to Minimize c^T f where c[k] = log(k+1) / (k+1)
    # Note: In LP, we minimize. So we minimize sum (log(k)/k) * f(k).
    # Wait, the objective is to MAXIMIZE a = -sum f(k) log(k)/k.
    # So we want to MINIMIZE sum f(k) log(k)/k.
    # Thus c[k] = log(k)/k.
    c = np.zeros(K)
    for k in range(1, K + 1):
        c[k - 1] = np.log(k) / k

    # Constraints:
    # 1. M(x) <= 1.0001 for x = 1..10K
    #    M(x) = sum_{k=1}^K f(k) floor(x/k)
    #    A_ub is a matrix of shape (10K, K) where A_ub[x-1, k-1] = floor(x/k)
    #    b_ub is a vector of shape (10K,) filled with 1.0001
    n_x = 10 * K
    # Vectorized construction of A_ub
    # x ranges from 1 to n_x
    x_vals = np.arange(1, n_x + 1)[:, np.newaxis]
    k_vals = np.arange(1, K + 1)[np.newaxis, :]
    A_ub = (x_vals // k_vals).astype(float)

    b_ub = np.full(n_x, 1.0001)

    # 2. sum f(k) / k = 0
    #    This is an equality constraint: A_eq f = 0
    #    A_eq is a vector of shape (1, K) where A_eq[0, k-1] = 1/k
    A_eq = np.zeros((1, K))
    for k in range(1, K + 1):
        A_eq[0, k - 1] = 1.0 / k
    b_eq = np.array([0.0])

    # Bounds:
    # Values are clipped to [-10, 10] by the evaluator.
    # So we can bound f(k) to [-10, 10].
    bounds = [(-10.0, 10.0)] * K

    # Solve the LP
    values = None
    try:
        # Check time before solving
        if time.time() > deadline - 10:
            raise TimeoutError("Not enough time for LP")
            
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if res.success:
            values = res.x
        else:
            # Fallback if LP fails
            pass

    except Exception:
        values = None

    # Fallback solution if LP failed or timed out
    if values is None:
        # Use a simple valid heuristic.
        # We need sum f(k)/k = 0 and M(x) <= 1.
        # A known simple solution: f(1)=0, f(2)=-1, f(3)=1.5, f(6)=0... 
        # Let's construct a small valid one.
        # Try domain [1, 2, 3]
        # f(1)/1 + f(2)/2 + f(3)/3 = 0
        # M(1) = f(1) <= 1
        # M(2) = f(1) + f(2) <= 1
        # M(3) = f(1) + f(2) + f(3) <= 1
        # M(4) = f(1) + f(2) + f(3) + f(4)...
        # Let's just use the seed-like solution from the prompt description if available, or a trivial one.
        # Trivial valid: f(k) = 0 for all k. Score 0.
        # Better: f(2) = -1, f(3) = 1.5, f(1) = -1.5? 
        # Sum: -1.5 + -0.5 + 0.5 = -1.5 != 0.
        # Let's try: f(1)=0, f(2)=-1, f(3)=1.5. Sum: 0 -0.5 + 0.5 = 0.
        # M(1) = 0 <= 1.
        # M(2) = 0 + (-1) = -1 <= 1.
        # M(3) = 0 + (-1) + 1.5 = 0.5 <= 1.
        # M(4) = 0 + (-1) + 1.5 = 0.5 <= 1 (floor(4/2)=2, floor(4/3)=1 -> 0 + 2(-1) + 1(1.5) = -0.5).
        # M(5) = 0 + 2(-1) + 1(1.5) = -0.5.
        # M(6) = 0 + 3(-1) + 2(1.5) = 0.
        # M(x) for x>6: floor(x/2) grows, floor(x/3) grows.
        # M(x) = -floor(x/2) + 1.5 floor(x/3).
        # For large x, -x/2 + 1.5x/3 = -x/2 + x/2 = 0.
        # It seems bounded.
        # Let's check max M(x).
        # x=1: 0
        # x=2: -1
        # x=3: 0.5
        # x=4: -0.5
        # x=5: -0.5
        # x=6: 0
        # x=7: -1 + 1.5 = 0.5
        # x=8: -2 + 1.5 = -0.5
        # x=9: -2 + 3 = 1.0
        # x=10: -3 + 3 = 0
        # x=11: -3 + 3 = 0
        # x=12: -3 + 4 = 1.0
        # It oscillates around 0 to 1. Max is 1.0.
        # So this is valid.
        domain_fb = np.array([1, 2, 3], dtype=np.int64)
        values_fb = np.array([0.0, -1.0, 1.5], dtype=np.float64)
        return domain_fb, values_fb

    # Check time
    if time.time() > deadline:
        # If we are running out of time, return the fallback
        domain_fb = np.array([1, 2, 3], dtype=np.int64)
        values_fb = np.array([0.0, -1.0, 1.5], dtype=np.float64)
        return domain_fb, values_fb

    return domain, values


# EVOLVE-BLOCK-END
