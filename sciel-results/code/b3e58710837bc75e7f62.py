# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    The function f is scored by a fixed evaluator (see evaluator.py) that:
      * subtracts a constant from f(1) so that sum_k f(k)/k == 0,
      * requires the step function M(x) = sum_k f(k) * floor(x/k) to stay <= 1 for every
        real x in [1, 10*max_key] (otherwise the construction is infeasible), and
      * returns the objective a = -sum_k f(k) * log(k) / k (to be maximized; optimum 1.0).
    The optimum is approached by the Mobius function mu(k), for which the classic identity
    sum_{k<=x} mu(k) floor(x/k) = 1 holds and -sum mu(k) log(k)/k -> 1.

    Representation:
        domain : np.ndarray, shape (m,), dtype int   -> positive integers in the domain of f
        values : np.ndarray, shape (m,), dtype float -> the corresponding real values f(k)

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    start_time = time.time()
    deadline = start_time + 600
    
    # Compute Mobius function up to K
    # We need to choose K. The constraint is M(x) <= 1 for x in [1, 10K].
    # For mu, sum_{k<=x} mu(k) floor(x/k) = 1 for all x.
    # However, we only have a partial function. We include k up to K.
    # The evaluator computes M(x) = sum_{k in domain} f(k) floor(x/k).
    # If we use mu(k) for k=1..K, then for x <= K, sum_{k=1}^x mu(k) floor(x/k) = 1.
    # For x > K, the sum is over k=1..K.
    # Let's analyze the error term.
    # M(x) = sum_{k=1}^K mu(k) floor(x/k).
    # We know sum_{k=1}^infty mu(k) floor(x/k) = 1.
    # So M(x) = 1 - sum_{k=K+1}^infty mu(k) floor(x/k).
    # The tail sum is roughly O(x/K) or related to Mertens function.
    # Actually, sum_{k>x} mu(k) floor(x/k) = 0 because floor(x/k)=0 for k>x.
    # So for x <= K, M(x) = 1 exactly.
    # For x > K, M(x) = 1 - sum_{k=K+1}^x mu(k) floor(x/k).
    # Note that floor(x/k) is 0 for k > x. So the sum is from K+1 to x.
    # Since x goes up to 10K, the sum is from K+1 to min(x, 10K).
    # Wait, the evaluator checks x in [1, 10K].
    # If x > K, floor(x/k) can be non-zero for k up to x.
    # But our domain only goes up to K.
    # So M(x) = sum_{k=1}^K mu(k) floor(x/k).
    # Is this <= 1?
    # Let's check small K.
    # K=1: f(1)=1. M(x) = floor(x/1) = x. For x=2, M(2)=2 > 1. Infeasible.
    # So we need K large enough or we need to adjust values.
    
    # The prompt says: "f(1) is replaced by f(1) - S with S = sum_k f(k)/k ... so that afterwards sum_k f(k)/k = 0".
    # This adjustment changes f(1).
    # Let f be our initial values.
    # S = sum_{k=1}^K mu(k)/k.
    # New f(1) = mu(1) - S = 1 - S.
    # New M(x) = (1-S)*floor(x/1) + sum_{k=2}^K mu(k) floor(x/k).
    # M(x) = floor(x/1) + sum_{k=1}^K mu(k) floor(x/k) - S*floor(x/1)
    # M(x) = x + 1 - S*x (for x <= K, assuming mu sum is 1)
    # Wait, sum_{k=1}^K mu(k) floor(x/k) = 1 for x <= K.
    # So M(x) = x + 1 - S*x = 1 + x(1-S).
    # We need M(x) <= 1.
    # So x(1-S) <= 0. Since x>0, we need 1-S <= 0 => S >= 1.
    # S = sum_{k=1}^K mu(k)/k.
    # For K=1, S=1. 1-S=0. M(x) = 1. Feasible?
    # If K=1, domain=[1], values=[1].
    # S = 1/1 = 1.
    # New f(1) = 1 - 1 = 0.
    # M(x) = 0 * floor(x/1) = 0 <= 1. Feasible.
    # Objective a = -sum f(k) log(k)/k = 0. Score 0.
    
    # We want to maximize a = -sum f(k) log(k)/k.
    # Using mu(k) gives a close to 1.
    # But the constraint M(x) <= 1 is tight.
    
    # Let's try to compute mu(k) for a larger K and check feasibility.
    # If we just return mu(k) for k=1..K, the evaluator adjusts f(1).
    # Let's see if we can find a K where the adjusted M(x) <= 1.
    
    # Actually, the identity sum_{k|x} mu(k) = 1 if x=1 else 0.
    # sum_{k=1}^x mu(k) floor(x/k) = 1.
    # This holds for all x.
    # So for x <= K, M_raw(x) = 1.
    # For x > K, M_raw(x) = 1 - sum_{k=K+1}^x mu(k) floor(x/k).
    
    # After adjustment:
    # f_new(1) = 1 - S.
    # M_new(x) = (1-S) floor(x/1) + sum_{k=2}^K mu(k) floor(x/k)
    #           = floor(x/1) + sum_{k=1}^K mu(k) floor(x/k) - S floor(x/1)
    #           = x + M_raw(x) - S x
    #           = M_raw(x) + x(1-S).
    
    # For x <= K: M_raw(x) = 1.
    # M_new(x) = 1 + x(1-S).
    # We need 1 + x(1-S) <= 1 => x(1-S) <= 0 => S >= 1.
    
    # For x > K: M_raw(x) = 1 - R(x) where R(x) = sum_{k=K+1}^x mu(k) floor(x/k).
    # M_new(x) = 1 - R(x) + x(1-S).
    # We need 1 - R(x) + x(1-S) <= 1 => x(1-S) <= R(x).
    
    # If S > 1, then 1-S is negative.
    # For x <= K, x(1-S) is negative, so M_new(x) < 1. Feasible.
    # For x > K, x(1-S) is negative. R(x) is likely positive?
    # R(x) = sum_{k=K+1}^x mu(k) floor(x/k).
    # This is the tail of the Mertens summation.
    # If R(x) is always >= 0 for x in (K, 10K], then we are feasible if S >= 1.
    
    # Let's calculate S for various K.
    # S(K) = sum_{k=1}^K mu(k)/k.
    # This converges to 0? No, 1/zeta(1) is divergent.
    # Wait, sum mu(k)/k converges to 0.
    # So S(K) starts at 1 (K=1), 1-1/2=0.5 (K=2), 0.5-1/3=0.166 (K=3), 0.166-1/5=0 (K=5)...
    # So S(K) is not >= 1 for K>1.
    # This means for K>1, 1-S > 0.
    # Then for x <= K, M_new(x) = 1 + x(1-S) > 1. Infeasible!
    
    # So we cannot simply use mu(k) for k=1..K with K>1 because the adjustment to f(1) breaks the constraint for x <= K.
    
    # We need to modify the values to satisfy the constraint.
    # The constraint is M_new(x) <= 1 for all x in [1, 10K].
    # M_new(x) = f(1) floor(x/1) + sum_{k=2}^K f(k) floor(x/k).
    # And f(1) is determined by sum f(k)/k = 0.
    # f(1) = - sum_{k=2}^K f(k)/k.
    # So M_new(x) = - (sum_{k=2}^K f(k)/k) x + sum_{k=2}^K f(k) floor(x/k).
    # Let g(k) = f(k) for k>1.
    # M_new(x) = sum_{k=2}^K g(k) (floor(x/k) - x/k).
    # Note that floor(x/k) - x/k is in [-1, 0).
    # Specifically, it is 0 if k divides x, and negative otherwise.
    # Let h_k(x) = floor(x/k) - x/k.
    # M_new(x) = sum_{k=2}^K g(k) h_k(x).
    # We want M_new(x) <= 1.
    # Also we want to maximize a = - sum_{k=1}^K f(k) log(k)/k.
    # a = - f(1) log(1)/1 - sum_{k=2}^K g(k) log(k)/k = - sum_{k=2}^K g(k) log(k)/k.
    # Since log(1)=0.
    # So we want to maximize - sum_{k=2}^K g(k) log(k)/k.
    # This is equivalent to minimizing sum_{k=2}^K g(k) log(k)/k.
    # Since log(k)/k > 0 for k>=2, we want g(k) to be as negative as possible.
    # But g(k) is constrained by M_new(x) <= 1.
    # M_new(x) = sum_{k=2}^K g(k) h_k(x) <= 1.
    # h_k(x) <= 0.
    # If we make g(k) negative, the term g(k) h_k(x) becomes positive (negative * negative).
    # So M_new(x) increases.
    # We need to keep M_new(x) <= 1.
    
    # Let's try to set g(k) = -mu(k) for k>1?
    # Then M_new(x) = sum_{k=2}^K -mu(k) (floor(x/k) - x/k).
    # = - sum_{k=2}^K mu(k) floor(x/k) + sum_{k=2}^K mu(k) x/k
    # = - (sum_{k=1}^K mu(k) floor(x/k) - mu(1)floor(x/1)) + x (sum_{k=2}^K mu(k)/k)
    # = - (M_raw(x) - x) + x (S(K) - 1)
    # = - M_raw(x) + x + x S(K) - x
    # = - M_raw(x) + x S(K).
    
    # For x <= K, M_raw(x) = 1.
    # M_new(x) = -1 + x S(K).
    # We need -1 + x S(K) <= 1 => x S(K) <= 2.
    # Max x is K. So K S(K) <= 2.
    # S(K) ~ 0. So this is likely satisfied.
    # For x > K, M_raw(x) = 1 - R(x).
    # M_new(x) = - (1 - R(x)) + x S(K) = -1 + R(x) + x S(K).
    # We need -1 + R(x) + x S(K) <= 1 => R(x) + x S(K) <= 2.
    # R(x) is the tail sum.
    
    # Let's check the objective.
    # a = - sum_{k=2}^K (-mu(k)) log(k)/k = sum_{k=2}^K mu(k) log(k)/k.
    # We know sum_{k=1}^infty mu(k) log(k)/k = -1.
    # So sum_{k=2}^K mu(k) log(k)/k approaches -1.
    # So a approaches -1?
    # Wait, a = - sum f(k) log(k)/k.
    # If f(k) = -mu(k) for k>1, then a = sum mu(k) log(k)/k.
    # This is approx -1.
    # The baseline score is 1.0.
    # The baseline had a positive a.
    # Let's re-read the baseline.
    # domain [1,2,3,5,30], values [3.47, -1, -1, -1, 1].
    # f(1) is adjusted.
    # S = 3.47/1 - 1/2 - 1/3 - 1/5 + 1/30 = 3.47 - 0.5 - 0.33 - 0.2 + 0.033 = 2.47.
    # f_new(1) = 3.47 - 2.47 = 1.
    # M_new(x) = 1*x + (-1)floor(x/2) + (-1)floor(x/3) + (-1)floor(x/5) + 1 floor(x/30).
    # For x=1: 1.
    # For x=2: 2 - 1 = 1.
    # For x=3: 3 - 1 - 1 = 1.
    # For x=5: 5 - 2 - 1 = 2? Wait.
    # floor(5/2)=2, floor(5/3)=1, floor(5/5)=1.
    # M(5) = 5 - 2 - 1 - 1 + 0 = 1.
    # It seems the baseline is carefully constructed to keep M(x) <= 1.
    
    # Let's try to optimize g(k) using the constraint M_new(x) <= 1.
    # This is a linear programming problem.
    # Maximize - sum g(k) log(k)/k
    # Subject to sum g(k) h_k(x) <= 1 for all x in [1, 10K].
    # h_k(x) = floor(x/k) - x/k.
    
    # We can solve this for small K.
    # Let's try K=10.
    # Variables g(2)..g(10).
    # Constraints for x=1..100.
    
    # We can use scipy.optimize.linprog.
    
    # However, we must be careful with time.
    # Let's implement a solver for small K and check if it improves.
    
    def solve_lp(K):
        # Variables: g(2), ..., g(K). Total K-1 variables.
        n_vars = K - 1
        # Objective: minimize sum g(k) log(k)/k
        c = np.zeros(n_vars)
        for i, k in enumerate(range(2, K+1)):
            c[i] = np.log(k) / k
        
        # Constraints: sum g(k) h_k(x) <= 1 for x in 1..10K
        # A_ub * g <= b_ub
        A_ub = np.zeros((10*K, n_vars))
        b_ub = np.ones(10*K)
        
        for x in range(1, 10*K + 1):
            for i, k in enumerate(range(2, K+1)):
                A_ub[x-1, i] = np.floor(x/k) - x/k
        
        # Bounds: g(k) can be any real?
        # The problem doesn't restrict g(k) to be integers or bounded.
        # But values are clipped to [-10, 10] in evaluator.
        # So we should bound g(k) to [-10, 10].
        bounds = [(-10, 10)] * n_vars
        
        from scipy.optimize import linprog
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            g = res.x
            # Construct f
            f = np.zeros(K+1)
            f[2:K+1] = g
            S = np.sum(f[2:K+1] / np.arange(2, K+1))
            f[1] = -S
            
            # Check feasibility with clipping
            f_clipped = np.clip(f[1:K+1], -10, 10)
            # Recalculate S with clipped values?
            # "values are clipped to [-10, 10]. Then f(1) is replaced..."
            # So the evaluator clips ALL values including f(1) initially?
            # "values holds the finite reals f(k)... values are clipped to [-10, 10]. Then f(1) is replaced..."
            # Yes, clip first, then adjust f(1).
            
            # So we must simulate this.
            f_raw = np.zeros(K+1)
            f_raw[1:K+1] = f[1:K+1]
            f_clipped = np.clip(f_raw, -10, 10)
            S_clip = np.sum(f_clipped[1:K+1] / np.arange(1, K+1))
            f_clipped[1] -= S_clip
            
            # Check M(x)
            domain = np.arange(1, K+1)
            values = f_clipped[1:K+1]
            
            # Calculate M(x) for x in 1..10K
            # M(x) = sum f(k) floor(x/k)
            # We can vectorize this.
            # floor(x/k) for x in 1..10K, k in 1..K.
            # Shape (10K, K).
            x_vals = np.arange(1, 10*K + 1)
            k_vals = np.arange(1, K+1)
            # floor(x/k)
            # Use broadcasting
            # x_vals[:, None] / k_vals[None, :]
            # This might be large. 10K * K.
            # If K=100, 1000 * 100 = 100,000. OK.
            # If K=1000, 10000 * 1000 = 10,000,000. OK.
            
            F = np.floor(x_vals[:, None] / k_vals[None, :])
            M = F @ values
            
            if np.any(M > 1.0001):
                return None
            
            # Calculate objective
            a = -np.sum(values * np.log(k_vals) / k_vals)
            
            return domain, values, a
        
        return None

    # Try increasing K
    best_result = None
    best_a = -np.inf
    
    # Start with small K
    for K in [10, 20, 50, 100, 200, 500]:
        if time.time() > deadline - 60:
            break
            
        res = solve_lp(K)
        if res is not None:
            domain, values, a = res
            if a > best_a:
                best_a = a
                best_result = (domain, values)
            
            # Print for debugging if needed
            # print(f"K={K}, a={a}")
            
    if best_result is not None:
        return best_result
    else:
        # Fallback to baseline
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values


# EVOLVE-BLOCK-END
