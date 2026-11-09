# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of positive integers to reals
    that maximizes the objective related to the Prime Number Theorem.

    The evaluator:
      * clips values to [-10, 10]
      * adjusts f(1) so that sum_k f(k)/k = 0
      * requires M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for all x in [1, 10*K]
      * computes a = -sum_k f(k) * log(k) / k (to be maximized)

    Strategy: Use the Möbius function mu(k) as a starting point, then refine
    via projected gradient descent to maximize the objective while keeping M(x) <= 1.
    """
    start_time = time.time()
    deadline = 600.0  # seconds
    safety_margin = 30.0  # seconds before deadline to stop

    def check_deadline():
        return (time.time() - start_time) > (deadline - safety_margin)

    def mobius_up_to(n):
        """Compute Möbius function mu(k) for k = 1..n using sieve."""
        mu = np.ones(n + 1, dtype=np.float64)
        is_prime = np.ones(n + 1, dtype=bool)
        primes = []
        for i in range(2, n + 1):
            if is_prime[i]:
                primes.append(i)
                mu[i] = -1.0
            for p in primes:
                if i * p > n:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    mu[i * p] = 0.0
                    break
                else:
                    mu[i * p] = -mu[i]
        mu[0] = 0.0
        return mu[1:]  # return mu(1)..mu(n)

    def compute_M_matrix(domain, max_x):
        """
        Compute the matrix A where A[x-1, i] = floor(x / domain[i])
        for x in 1..max_x and i in 0..m-1.
        """
        m = len(domain)
        A = np.empty((max_x, m), dtype=np.float64)
        for i, k in enumerate(domain):
            A[:, i] = np.floor(np.arange(1, max_x + 1, dtype=np.float64) / k)
        return A

    def evaluate_objective(domain, values):
        """Compute the objective a = -sum f(k)*log(k)/k after adjustment."""
        m = len(domain)
        if m == 0:
            return 0.0
        # Clip values
        v = np.clip(values, -10.0, 10.0)
        # Compute S = sum v(k)/k
        S = np.sum(v / domain.astype(np.float64))
        # Adjust f(1) if 1 is in domain
        idx_1 = np.where(domain == 1)[0]
        if len(idx_1) > 0:
            v[idx_1[0]] -= S
        else:
            # If 1 not in domain, we need to add it with value -S
            # But for evaluation, we just compute a without the adjustment effect on a
            # Actually, the evaluator replaces f(1) by f(1) - S. If 1 not in domain,
            # f(1) = 0 before adjustment, so f(1) = -S after.
            # So effectively we add a term -S*log(1)/1 = 0 to the sum, but -S*log(1)=0.
            # Wait, the metric is a = -sum_k f(k)*log(k)/k. If we add f(1)=-S, then
            # the new sum includes -S*log(1)/1 = 0. So a doesn't change?
            # Let me re-read: "f(1) is replaced by f(1) - S ... so that afterwards sum_k f(k)/k = 0"
            # If 1 is not in the domain, f(1)=0 before, so f(1) becomes -S.
            # The metric is computed over the adjusted values.
            # So a = -sum_{k in domain} f_adjusted(k)*log(k)/k - (-S)*log(1)/1
            # But log(1) = 0, so the f(1) term contributes 0.
            # However, the sum over k in domain uses the original values (adjusted for k=1 if present).
            # So if 1 not in domain, a = -sum_{k in domain} v(k)*log(k)/k
            pass
        a = -np.sum(v * np.log(domain.astype(np.float64)) / domain.astype(np.float64))
        return a

    def check_feasibility(domain, values):
        """Check if M(x) <= 1.0001 for all x in [1, 10*K]."""
        m = len(domain)
        K = max(domain)
        max_x = 10 * K
        v = np.clip(values, -10.0, 10.0)
        # Compute S and adjust f(1)
        S = np.sum(v / domain.astype(np.float64))
        idx_1 = np.where(domain == 1)[0]
        if len(idx_1) > 0:
            v[idx_1[0]] -= S
        # Compute M(x) for all x
        # M(x) = sum_i v[i] * floor(x / domain[i])
        # We can compute this efficiently
        M_max = 0.0
        for x in range(1, max_x + 1):
            Mx = 0.0
            for i, k in enumerate(domain):
                Mx += v[i] * (x // k)
            if Mx > M_max:
                M_max = Mx
        return M_max <= 1.0001, M_max

    # Start with a reasonable domain size
    K_max = 200000
    # Use a moderate K to keep computation tractable
    K = 5000
    mu = mobius_up_to(K)
    
    # Select a subset: use indices where mu != 0, plus ensure 1 is included
    domain_full = np.arange(1, K + 1, dtype=np.int64)
    # Filter to mu != 0
    mask = mu != 0
    domain = domain_full[mask]
    values = mu[mask].copy()
    
    # Ensure 1 is in domain (it should be, mu(1) = 1)
    if 1 not in domain:
        domain = np.insert(domain, 0, 1)
        values = np.insert(values, 0, 1.0)
    
    # Sort domain
    sort_idx = np.argsort(domain)
    domain = domain[sort_idx]
    values = values[sort_idx]
    
    m = len(domain)
    
    # Precompute log(k)/k for objective
    log_k_over_k = np.log(domain.astype(np.float64)) / domain.astype(np.float64)
    
    # Projected gradient descent
    # Objective: maximize a = -sum f(k)*log(k)/k = -dot(values, log_k_over_k)
    # So we minimize -a = dot(values, log_k_over_k)
    # Gradient of -a w.r.t. values: log_k_over_k
    # Constraints: M(x) <= 1.0001 for all x in [1, 10*K]
    
    # For efficiency, we'll use a subset of x constraints or approximate
    # Let's do a simple projected gradient with a penalty for constraint violation
    
    v = values.copy()
    
    # Precompute the constraint matrix for a subset of x values
    # To keep memory manageable, sample x values
    K_cur = max(domain)
    max_x = 10 * K_cur
    # Sample x values: use every 10th x, plus x=1,2,...,10
    x_sample = np.unique(np.concatenate([
        np.arange(1, min(11, max_x + 1), dtype=np.int64),
        np.arange(10, max_x + 1, 10, dtype=np.int64),
        np.array([max_x], dtype=np.int64)
    ]))
    x_sample = x_sample[x_sample <= max_x]
    
    # Build A matrix: A[j, i] = floor(x_sample[j] / domain[i])
    A = np.empty((len(x_sample), m), dtype=np.float64)
    for j, x in enumerate(x_sample):
        A[j, :] = np.floor(x / domain.astype(np.float64))
    
    # Penalty-based optimization
    alpha = 0.1  # step size
    penalty = 100.0  # penalty weight
    n_iter = 1000
    
    best_v = v.copy()
    best_obj = evaluate_objective(domain, v)
    
    for it in range(n_iter):
        if check_deadline():
            break
        
        # Compute current adjusted values
        v_adj = np.clip(v, -10.0, 10.0)
        S = np.sum(v_adj / domain.astype(np.float64))
        idx_1 = np.where(domain == 1)[0]
        if len(idx_1) > 0:
            v_adj[idx_1[0]] -= S
        
        # Compute M(x) for sampled x
        M_vals = A @ v_adj
        violation = np.maximum(0.0, M_vals - 1.0001)
        penalty_loss = penalty * np.sum(violation)
        
        # Gradient of objective: -log_k_over_k (we're maximizing, so gradient of -a is log_k_over_k)
        # We want to maximize a, so minimize -a + penalty
        grad_obj = log_k_over_k  # gradient of -a
        
        # Gradient of penalty
        # penalty = penalty * sum_j max(0, M_j - 1.0001)
        # d(penalty)/d(v_adj) = penalty * A.T @ indicator(M_j > 1.0001)
        active = (M_vals > 1.0001).astype(np.float64)
        grad_pen = penalty * (A.T @ active)
        
        # Total gradient
        grad = grad_obj + grad_pen
        
        # Projected gradient step
        v_new = v - alpha * grad
        
        # Project to box [-10, 10]
        v_new = np.clip(v_new, -10.0, 10.0)
        
        # Evaluate new objective
        obj_new = evaluate_objective(domain, v_new)
        feas_new, Mmax_new = check_feasibility(domain, v_new)
        
        if feas_new:
            v = v_new
            if obj_new > best_obj:
                best_obj = obj_new
                best_v = v.copy()
        else:
            # Reduce step size
            alpha *= 0.5
            if alpha < 1e-8:
                alpha = 1e-8
    
    # Use best feasible solution
    if check_feasibility(domain, best_v)[0]:
        values = best_v
    else:
        values = v
    
    # Final check
    feasible, Mmax = check_feasibility(domain, values)
    if not feasible:
        # Fall back to simple Mobius
        return domain, values
    
    return domain, values


# EVOLVE-BLOCK-END
