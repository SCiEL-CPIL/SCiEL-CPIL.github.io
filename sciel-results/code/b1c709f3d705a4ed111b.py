"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Strategy: Bisection on target overlap T, solving a convex QP feasibility problem
using scipy.optimize.linprog (converted from QP via active set or using
scipy.optimize.minimize with SLSQP but initialized via bisection logic).
Actually, to be robust and fast, we will use scipy.optimize.minimize with
SLSQP for the QP, but the outer loop is a bisection on T.
The QP is: min ||h||^2 s.t. A h <= b, sum h = N/2, 0<=h<=1.
This is a convex problem. We can solve it using scipy.optimize.minimize.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill

# Precompute the matrix A for the linear constraints C_k <= T
# C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
# Wait, the overlap definition in the prompt is:
# "minimize the maximum overlap of h with shifted 1-h"
# The current program defines:
# C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
# Let's verify this definition.
# Overlap of h and shifted 1-h.
# Let h be the step heights. 1-h is the complement.
# Shifted 1-h by k.
# Overlap = sum h_i * (1-h)_{i-k} = sum h_i * (1 - h_{i-k}) = sum h_i - sum h_i h_{i-k}
# If h is normalized such that sum h_i = N/2, then sum h_i = N/2.
# So Overlap_k = N/2 - sum_{i=k}^{N-1} h_i h_{i-k} (assuming indices wrap or are valid)
# The current code uses:
# r_k = sum_{i=k}^{N-1} h[i] * h[i-k]
# cs = (2.0 / n) * (n / 2.0 - r)
# This matches the "Overlap" scaled by 2/N.
# So the objective is to minimize max_k C_k.
# C_k is a quadratic function of h.
# Minimizing max C_k is non-convex because C_k involves products h_i h_j.
# HOWEVER, the prompt asks for a structural change.
# The current program minimizes max C_k directly using gradient descent on h.
# The proposed strategy is "Bisection on T with QP feasibility".
# But C_k <= T is a quadratic constraint, not linear.
# A QP minimizes a quadratic objective subject to LINEAR constraints.
# If we fix T, the constraint C_k <= T is quadratic.
# So it's a Quadratically Constrained Quadratic Program (QCQP), which is non-convex.
#
# Let's re-read the inspiration and directive.
# Inspiration 3 uses coordinate descent.
# Knowledge: "Bisection on t with Nesterov accelerated projected gradient escapes local minima... Measured: 1.182 -> 1.213"
# The knowledge base explicitly suggests bisection on t.
# How does one check feasibility of C_k <= T?
# If we cannot solve the QCQP easily, maybe we can reformulate?
# Or perhaps the "QP" in the strategy description was a simplification for "convex relaxation" or "specific structure".
#
# Actually, there is a known trick for this specific problem (Erdos overlap).
# If we assume h is a "step function" or has specific symmetry, the problem might simplify.
# But h is a vector of N heights.
#
# Let's look at the structure of C_k.
# C_k = (2/N)(N/2 - <h, shift_k(h)>)
# We want min max_k C_k.
#
# Alternative Structural Change:
# Instead of optimizing h directly, optimize the **autocorrelation sequence** r_k.
# The constraints on r_k are that it must be the autocorrelation of some vector h in [0,1] with sum N/2.
# This is a moment problem / PSD constraint.
# The Gram matrix G of h must be PSD.
# G_{ij} = h_i h_j? No, that's rank 1.
# The autocorrelation r_k = sum h_i h_{i-k}.
# This is the diagonal of the circulant matrix formed by h?
#
# Let's stick to the **Bisection + Nesterov Projected Gradient** strategy mentioned in the knowledge base, as it is proven to work (1.182 -> 1.213) and is a structural change from SLSQP.
# The "feasibility" step in bisection usually implies checking if a solution exists for a given T.
# If we use Projected Gradient Descent to minimize max(C_k) - T, and it goes to 0, it's feasible.
# But simpler: Just run Nesterov Projected Gradient on the objective f(h) = max(C_k) directly, but use the bisection idea to guide the step size or to verify convergence?
#
# Actually, the knowledge says: "Bisection on t ... escapes local minima".
# This implies the inner loop is a solver for a sub-problem parameterized by t.
# If the sub-problem is "minimize ||h - h_prev||^2 subject to C_k <= t", that's a QCQP.
#
# Let's try a different structural change: **Alternating Minimization / Coordinate Descent with Bisection**.
# Or simply: **Nesterov Accelerated Projected Gradient Descent** on the smooth approximation of the max function, but with a much more aggressive and robust schedule, combined with **random restarts** guided by the bisection logic (i.e., if we get stuck, assume the current max is an upper bound and try to find a lower one).
#
# However, the most distinct structural change from the current SLSQP is to use **First-Order Methods (Nesterov PGD)** which are global in nature for convex problems, and for non-convex, less prone to sharp local minima if properly accelerated.
#
# Let's implement **Nesterov Accelerated Projected Gradient Descent** on the objective $f(h) = \max_k C_k(h)$.
# To handle the non-smoothness, we use the log-sum-exp smooth approximation with a high beta, or we use the exact max and subgradients.
# The current code uses SLSQP with smooth max.
# Switching to Nesterov PGD is a different optimization method family.
#
# Additionally, we can incorporate the **Bisection** idea:
# We don't just minimize. We try to prove that a lower T is infeasible.
# But for a coding task, simply using a robust first-order method with multiple restarts and a good initialization is a solid structural change.
#
# Let's refine the "Bisection" aspect:
# We can perform a binary search on the answer T.
# For a given T, we want to find h such that C_k(h) <= T for all k, sum h = N/2, 0<=h<=1.
# This is a feasibility problem.
# We can solve it by minimizing the violation: $V(h) = \sum_k \max(0, C_k(h) - T)$.
# If min V(h) is close to 0, T is feasible.
# We use Projected Gradient Descent (with Nesterov acceleration) to minimize V(h).
# This decouples the "search for the optimal value" (bisection) from the "finding a witness" (PGD).
# This is structurally different from directly minimizing the max.

def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs

def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h

def _project(h):
    """
    Project h onto the set {x : sum x = N/2, 0 <= x <= 1}.
    This is the projection onto the intersection of a hyperplane and a box.
    We can solve this using a binary search on the Lagrange multiplier for the sum constraint.
    """
    h = np.asarray(h, dtype=np.float64)
    target = N / 2.0
    
    # If sum is already close and within bounds, return
    if np.all(h >= 0) and np.all(h <= 1) and abs(np.sum(h) - target) < 1e-9:
        return h
    
    # Binary search for lambda such that sum(clip(h - lambda, 0, 1)) = target
    # f(lambda) = sum(clip(h - lambda, 0, 1)) is monotonically decreasing in lambda.
    lo = np.min(h) - 1.0
    hi = np.max(h) + 1.0
    
    for _ in range(50):
        mid = (lo + hi) / 2.0
        proj = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(proj)
        if s > target:
            lo = mid
        else:
            hi = mid
            
    proj = np.clip(h - (lo + hi) / 2.0, 0.0, 1.0)
    # Fine tune to ensure exact sum
    diff = target - np.sum(proj)
    if abs(diff) > 1e-12:
        mask = (proj > 0) & (proj < 1)
        if np.any(mask):
            proj[mask] += diff / np.sum(mask)
            proj = np.clip(proj, 0.0, 1.0)
    return proj

def _nesterov_pgd(h0, T, max_iter=1000, step_size=0.01):
    """
    Minimize V(h) = sum_k max(0, C_k(h) - T) using Nesterov Accelerated PGD.
    Returns (h, min_violation).
    """
    h = h0.copy()
    h_prev = h0.copy()
    t_k = 1.0
    
    for i in range(max_iter):
        t_new = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
        y = h + (t_k - 1) / t_new * (h - h_prev)
        
        # Compute gradient of V(y)
        cs = _linear_overlap_values(y)
        violation = np.maximum(0, cs - T)
        active = (cs > T)
        
        # Gradient of C_k
        # dC_k/dh_j = -(2/N) * ( h_{j-k} (if j>=k) + h_{j+k} (if j+k<N) )
        # We only need gradient for active k
        grad = np.zeros(N)
        if np.any(active):
            # Vectorized gradient for active constraints
            # For each active k, add contribution to grad
            # This loop is O(N * #active). #active can be up to N. O(N^2).
            # For N=100, 10000 ops per iteration. 1000 iterations -> 10M ops. Fast.
            for k in np.where(active)[0]:
                wk = 1.0 # weight for max(0, .) is 1 if active
                factor = -(2.0 / N) * wk
                if k > 0:
                    grad[k:] += factor * y[:N-k]
                if k < N:
                    grad[:N-k] += factor * y[k:]
        
        # Projected Gradient Step
        h_new = _project(y - step_size * grad)
        
        h_prev = h
        h = h_new
        t_k = t_new
        
        # Early stopping if violation is very small
        v = np.sum(violation)
        if v < 1e-6:
            break
            
    return h, float(np.sum(np.maximum(0, _linear_overlap_values(h) - T)))

def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    # Initial guess: Cosine
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # Bisection on T
    # Lower bound: 0 (theoretical min)
    # Upper bound: current best score
    T_lo = 0.0
    T_hi = best_score
    
    # We will iterate, trying to find a feasible h for midpoints.
    # If feasible, T_hi = mid. Else T_lo = mid.
    # However, the "feasibility" check via PGD might fail to find a solution even if one exists.
    # So we use a heuristic: if PGD reduces the score significantly, accept it.
    
    # Instead of strict bisection which requires a reliable feasibility oracle,
    # we use the bisection logic to **guide restarts**.
    # We try to find a solution with score < T_mid.
    
    # Strategy:
    # 1. Start with T = best_score.
    # 2. Run Nesterov PGD to minimize violation of T.
    # 3. If it finds a solution with score < T, update best.
    # 4. Tighten T and repeat.
    
    T_target = best_score
    
    while time.time() - t0 < _DEADLINE - 10:
        # Try to beat T_target
        # Use multiple random restarts for PGD to escape local minima
        found_better = False
        
        for restart in range(5):
            if time.time() - t0 > _DEADLINE - 5:
                break
                
            # Initialize h
            if restart == 0:
                h_init = best_h.copy() if best_h is not None else h_cos
            else:
                # Perturb best_h or use random
                if best_h is not None:
                    h_init = best_h + 0.1 * np.random.randn(N)
                else:
                    h_init = np.random.uniform(0, 1, N)
                h_init = _fix_sum(h_init)
            
            # Run PGD to minimize violation of T_target
            h_res, viol = _nesterov_pgd(h_init, T_target, max_iter=500, step_size=0.05)
            
            s = _score(h_res)
            if s < T_target - 1e-6:
                _update_best(h_res)
                found_better = True
                T_target = best_score
                break # Restart with new tighter target
            
        if not found_better:
            # If we couldn't beat T_target, maybe T_target is too tight for PGD to handle?
            # Or we are stuck.
            # Let's try a direct optimization step with SLSQP on the current best to polish?
            # No, stick to first-order.
            # Try a looser target to see if we can find a better landscape?
            # Actually, if PGD fails to reduce below T_target, it might be a local min.
            # Let's try a different step size or more iterations.
            
            # If we are stuck, we might just exit the bisection loop and do a final polish.
            # But let's try one more time with a slightly larger T to see if we can get a different basin.
            T_try = T_target * 1.05
            h_res, _ = _nesterov_pgd(best_h if best_h is not None else h_cos, T_try, max_iter=1000, step_size=0.02)
            s = _score(h_res)
            if s < best_score:
                _update_best(h_res)
                T_target = best_score
            else:
                # Stagnation. Break out of bisection loop.
                break

    # Final polish with SLSQP if time permits, using the best h found by PGD
    if best_h is not None and time.time() - t0 < _DEADLINE - 2:
        def _smooth_max_grad(h, beta):
            h = np.asarray(h, dtype=np.float64)
            n = len(h)
            cs = _linear_overlap_values(h)
            m = np.max(cs)
            exp_terms = np.exp(beta * (cs - m))
            sum_exp = np.sum(exp_terms)
            w = exp_terms / sum_exp
            val = m + np.log(sum_exp) / beta
            grad = np.zeros(n)
            for k in range(n):
                wk = w[k]
                if wk == 0.0:
                    continue
                factor = -(2.0 / n) * wk
                if k > 0:
                    grad[k:] += factor * h[:n-k]
                if k < n:
                    grad[:n-k] += factor * h[k:]
            return val, grad

        def obj_func(x):
            val, _ = _smooth_max_grad(x, 50.0)
            return val
        def jac_func(x):
            _, grad = _smooth_max_grad(x, 50.0)
            return grad

        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                 'jac': lambda x: np.ones(N)}]
        bounds = [(0.0, 1.0)] * N
        try:
            res = minimize(obj_func, best_h.copy(), jac=jac_func, method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 100, 'ftol': 1e-12})
            if res.success:
                _update_best(res.x)
        except:
            pass

    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _fix_sum(best_h)
    return best_h
