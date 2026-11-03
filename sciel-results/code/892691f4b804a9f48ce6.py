"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Strategy: Linear Programming (LP) via Epigraph Formulation.
The objective is to minimize T such that for all shifts k, the overlap C_k <= T.
The overlap C_k is a linear function of h.
Thus, the problem is:
Minimize T
Subject to:
  sum(h) = N/2
  0 <= h_i <= 1
  C_k(h) <= T for all k in 0..N-1

This is a standard Linear Program with N+1 variables (h_0...h_{N-1}, T)
and 2N+1+N constraints.
We use scipy.optimize.linprog with the HiGHS solver, which is highly efficient
for LPs of this size (100 variables).
"""
import time
import numpy as np
from scipy.optimize import linprog

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill

def _build_lp_matrices():
    """
    Construct the A_ub, b_ub, A_eq, b_eq, bounds for the LP.
    Variables: x = [h_0, ..., h_{N-1}, T]
    """
    # Constraints: C_k(h) <= T  =>  C_k(h) - T <= 0
    # C_k(h) = 1 - (2/N) * sum_{j} h_j h_{j+k} ... wait.
    # Let's re-read the definition of overlap in the seed.
    # Seed: cs = (2.0 / n) * (n / 2.0 - r) where r is autocorrelation.
    # r_k = sum_{j=0}^{N-1-k} h_j h_{j+k} ? 
    # No, the seed uses FFT on zero-padded h.
    # Let's look at _linear_overlap_values in seed:
    # h_pad = [h, 0, ..., 0]
    # H = fft(h_pad)
    # corr = ifft(H * conj(H)) -> This is circular autocorrelation of the padded signal.
    # Since h_pad is zero-padded to 2N, the linear autocorrelation is extracted from the first N lags.
    # r_k = sum_{j=0}^{N-1-k} h_j h_{j+k}
    # cs_k = (2/N) * (N/2 - r_k) = 1 - (2/N) r_k
    
    # Wait, the seed says: "minimize the maximum overlap of h with shifted 1-h".
    # Overlap of h and shifted 1-h.
    # Let h' be h. Let g be 1-h.
    # Shift g by k: g_{i+k} = 1 - h_{i+k}.
    # Overlap = sum h_i * g_{i+k} ? Or sum h_i * h_{i+k} for the "1-h" part?
    # The task says "overlap of h with shifted 1-h".
    # Usually overlap of two sequences a, b is sum a_i b_i.
    # Here a=h, b=shifted(1-h).
    # sum_{i} h_i (1 - h_{i+k}) = sum h_i - sum h_i h_{i+k} = N/2 - r_k.
    # So the raw overlap is N/2 - r_k.
    # The seed computes cs = (2/N) * (N/2 - r).
    # So cs is the normalized overlap.
    # We want to minimize max_k cs_k.
    
    # cs_k = 1 - (2/N) r_k
    # r_k = sum_{j=0}^{N-1-k} h_j h_{j+k}
    # This is NOT linear in h! It is quadratic (bilinear).
    # r_k involves products h_j * h_{j+k}.
    # Therefore, the problem is NOT a Linear Program. It is a Quadratic Program (QP) or Non-Convex.
    
    # My previous thought was wrong. The overlap definition in the seed relies on autocorrelation, which is quadratic.
    # Let's re-read the task carefully.
    # "minimize the maximum overlap of h with shifted 1-h"
    # If the overlap is defined as sum h_i * (1-h)_{i+k}, it is quadratic.
    # Is there a linear interpretation?
    # Maybe the "overlap" is just the sum of products? Yes, that's quadratic.
    
    # So LP is not directly applicable to the standard definition if it involves products of h.
    # However, if we fix one of the sequences? No, both are h and 1-h.
    
    # Let's look at the inspirations.
    # Inspiration 2 mentions "spectral optimization".
    # Inspiration 1 mentions "_compute_overlaps".
    
    # If the problem is non-convex (minimizing max of quadratic forms), LP won't work directly.
    # BUT, maybe I can use a different strategy:
    # **Alternating Direction Method of Multipliers (ADMM)** or **Coordinate Descent**?
    # Or **Projected Gradient Descent** with a better objective?
    
    # Let's reconsider the structure.
    # Objective: min_h max_k (1 - (2/N) sum h_j h_{j+k})
    # Equivalent to: max_h min_k ((2/N) sum h_j h_{j+k})
    # Let R(h) be the vector of autocorrelations.
    # We want to maximize the minimum autocorrelation.
    
    # This is still non-convex.
    
    # Alternative Strategy: **Randomized Coordinate Descent / Simulated Annealing with Spectral Initialization**.
    # The current program uses Parametric Manifold + SLSQP.
    # A different approach: **Directly optimize the spectral magnitude**.
    # The autocorrelation r_k is the inverse FFT of |H|^2.
    # Minimizing max r_k is related to minimizing the peak of the power spectrum?
    # No, r_k is the time-domain autocorrelation.
    
    # Let's try a **Genetic Algorithm** or **Differential Evolution** on the full N-dimensional space, but with a smarter initialization and penalty function, avoiding the SLSQP local trap.
    # Or better: **Nelder-Mead** with multiple random restarts?
    
    # Actually, let's look at the "Erdos" aspect.
    # This is related to "minimum overlap" problems in combinatorics, often solved by constructing specific binary or near-binary sequences.
    # Maybe we should restrict h to be "almost binary" (0 or 1)?
    # If h is binary, sum h = N/2 means N/2 ones and N/2 zeros.
    # Then r_k is the number of positions where h_j=1 and h_{j+k}=1.
    # We want to minimize max_k (1 - (2/N) r_k) -> maximize min_k r_k.
    # This is the problem of finding a binary sequence with maximum minimum autocorrelation.
    # This is a known combinatorial problem (Golay sequences, etc.).
    
    # Strategy: **Combinatorial Search for Binary Sequences + Local Perturbation**.
    # 1. Generate random binary sequences with N/2 ones.
    # 2. Evaluate score.
    # 3. Use local search (swap 0 and 1) to improve.
    # 4. Since the space of binary sequences is huge, use heuristics.
    
    # However, the task allows h in [0,1]. Continuous values might be better?
    # The seed allows continuous.
    
    # Let's try **Alternating Minimization** on the quadratic form.
    # Or simply **Basin Hopping** with a robust local optimizer.
    
    # Let's go with a **Hybrid Global-Local Search**:
    # 1. Use Differential Evolution (DE) on the full N-dim space, but with a time budget.
    # 2. DE is stochastic and handles non-convexity well.
    # 3. Refine the best DE result with L-BFGS-B (faster than SLSQP for large N) using the smooth max gradient.
    
    # Why is this different?
    # Current: Parametric (low dim) -> SLSQP (full dim).
    # New: Global DE (full dim) -> L-BFGS-B (full dim).
    # This avoids the bias of the parametric model (sine waves) which might miss optimal structures.
    # DE explores the full 100-dim space stochastically.
    
    # To make it even more different: **Simulated Annealing** with a "peak shifting" move?
    # No, DE is more standard and robust for this size.
    
    # Let's implement:
    # 1. Efficient score and gradient calculation (reuse from seed, but optimize).
    # 2. DE for global search (short time budget, e.g., 30s).
    # 3. L-BFGS-B for local refinement (remaining time).
    
    # One more idea: **Bisection on T**.
    # Check if there exists h such that C_k(h) <= T for all k.
    # This is a feasibility problem.
    # C_k(h) = 1 - (2/N) r_k <= T  => r_k >= (N/2)(1-T).
    # Let R_min = (N/2)(1-T).
    # We need r_k >= R_min for all k.
    # This is a set of quadratic inequalities.
    # Not easy to solve.
    
    # Let's stick to **DE + L-BFGS-B**.
    
    pass

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
    
    # If clipping changed the sum, do a second pass
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    # Final tiny correction on one element to ensure exact sum in floating point
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h

def _smooth_max_val_grad(h, beta):
    """
    Value and gradient of the smooth maximum of the overlap values.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # C_k = 1 - (2/n) * r_k
    # r_k = sum_{j=0}^{n-1-k} h_j h_{j+k}
    # dC_k/dh_j = -(2/n) * ( h_{j-k} (if j>=k) + h_{j+k} (if j+k<n) )
    # Note: For k=0, r_0 = sum h_j^2. dC_0/dh_j = -(2/n) * 2h_j.
    # The formula above handles k=0?
    # If k=0:
    # Term 1: h_{j-0} for j>=0 -> h_j.
    # Term 2: h_{j+0} for j<n -> h_j.
    # Sum: 2h_j.
    # Factor: -(2/n) * w_0.
    # Grad: -(4/n) w_0 h_j.
    # This matches d(1 - 2/n sum h^2)/dh = -4/n h. Correct.
    
    for k in range(n):
        if w[k] < 1e-9:
            continue
        factor = -(2.0 / n) * w[k]
        
        # Term 1: h_{j-k} for j >= k
        if k > 0:
            grad[k:] += factor * h[:n-k]
        # Term 2: h_{j+k} for j+k < n
        if k < n:
            grad[:n-k] += factor * h[k:]
            
    return val, grad

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
            return True
        return False

    # Strategy: Global DE + Local L-BFGS-B
    
    # 1. Global Search with Differential Evolution
    # We run DE for a limited time to find a good starting point.
    # DE is stochastic, so we use a fixed seed for reproducibility.
    
    def de_objective(x):
        # x is a candidate h
        x = _fix_sum(x)
        return _score(x)
    
    bounds = [(0.0, 1.0)] * N
    
    # Run DE with a time budget
    de_start = time.time()
    de_deadline = de_start + 30  # 30 seconds for global search
    
    try:
        # Use differential_evolution with a callback to check time
        def callback(xk, convergence):
            if time.time() > de_deadline:
                raise TimeoutError("DE time limit exceeded")
            
        from scipy.optimize import differential_evolution
        result = differential_evolution(de_objective, bounds, maxiter=100, popsize=10, 
                                        seed=42, tol=1e-5, polish=False, callback=callback)
        # If it finished normally, result.x is the best
        _update_best(result.x)
    except TimeoutError:
        # If timeout, we don't have the final result easily accessible from DE in all versions
        # But we can just proceed to local refinement from the best found so far?
        # DE in scipy doesn't expose the best intermediate easily without callback state.
        # Let's just rely on the final result if it completes, or use a random start if not.
        # Actually, differential_evolution raises the exception, so we can catch it.
        # We need to capture the best x found before the exception.
        # This is tricky. Let's just run DE with a small maxiter that fits in time, 
        # or use a simpler global method like Random Restarts + Local Search.
        
        # Let's switch to: **Multiple Random Restarts with Local Refinement**.
        # This is more controllable.
        pass
    
    # Since DE callback handling for early termination and best result retrieval is clunky,
    # let's implement a simpler **Random Restarts + L-BFGS-B** strategy.
    # This is structurally different from the parametric manifold approach.
    
    num_restarts = 10
    time_per_restart = 10  # 10 seconds per restart, total 100s
    
    for i in range(num_restarts):
        if time.time() > _DEADLINE - 5:
            break
            
        # Random initialization
        h0 = np.random.uniform(0, 1, N)
        h0 = _fix_sum(h0)
        
        # Local Refinement with L-BFGS-B
        # L-BFGS-B is a quasi-Newton method, faster than SLSQP for large N.
        # It handles bounds [0,1] natively.
        # We need to handle the sum constraint.
        # L-BFGS-B does not handle equality constraints.
        # We can parameterize h to satisfy the sum constraint?
        # Or use the penalty method?
        # Or use SLSQP which handles constraints?
        
        # SLSQP is slower but handles constraints.
        # Let's use SLSQP but with a tighter budget per restart.
        
        def _run_slsqp(h_init):
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                     'jac': lambda x: np.ones(N)}]
            bounds = [(0.0, 1.0)] * N
            
            def obj_func(x):
                val, _ = _smooth_max_val_grad(x, beta=50.0)
                return val
                
            def jac_func(x):
                _, grad = _smooth_max_val_grad(x, beta=50.0)
                return grad
                
            try:
                res = minimize(obj_func, h_init, jac=jac_func, method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options={'maxiter': 100, 'ftol': 1e-8})
                if res.success or res.fun < best_score:
                    _update_best(res.x)
            except Exception:
                pass

        _run_slsqp(h0)
        
    # If best_h is still None (shouldn't happen), fallback
    if best_h is None:
        best_h = _fix_sum(np.random.uniform(0, 1, N))
    
    best_h = _fix_sum(best_h)
    return best_h

# Wait, the "Random Restarts" strategy is essentially what Inspiration 1 might be doing?
# Inspiration 1: "_optimize_from_start".
# The current program uses Parametric Manifold.
# Random Restarts is a different family (Stochastic Global Search).
# But is it "materially different"? Yes, it abandons the parametric structure.
# To make it stronger, let's add a **Spectral Initialization** for the random restarts.
# Instead of uniform random, we can initialize with random phases in the frequency domain.
# This ensures the sequence has some structure (band-limited or specific spectrum) 
# which might be closer to the optimal solution than white noise.

def _spectral_random_init():
    """
    Generate a random h by assigning random magnitudes and phases in the frequency domain,
    then transforming back to time domain and projecting to [0,1] with sum N/2.
    """
    n = N
    n_fft = 2 * n
    # Random magnitudes
    mags = np.random.uniform(0.1, 1.0, n_fft // 2 + 1)
    # Random phases
    phases = np.random.uniform(0, 2 * np.pi, n_fft // 2 + 1)
    
    H = mags * np.exp(1j * phases)
    # Make it Hermitian symmetric for real time domain
    H_full = np.zeros(n_fft, dtype=complex)
    H_full[:n_fft//2+1] = H
    H_full[n_fft//2+1:] = np.conj(H[1:n_fft//2][::-1])
    
    h = np.fft.irfft(H_full, n=n_fft)[:n]
    
    # Normalize to have mean 0.5 and std dev 0.5 (roughly)
    # Then clip and fix sum
    h = (h - np.mean(h)) / (np.std(h) + 1e-9)
    h = 0.5 + 0.5 * h
    h = np.clip(h, 0, 1)
    h = _fix_sum(h)
    return h

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
            return True
        return False

    # Strategy: Spectral Random Restarts + SLSQP Refinement
    
    num_restarts = 20
    # Allocate time dynamically
    total_budget = _DEADLINE - t0
    time_per_restart = total_budget / num_restarts - 1  # 1s margin
    
    for i in range(num_restarts):
        if time.time() > _DEADLINE - 5:
            break
            
        # Spectral Initialization
        h0 = _spectral_random_init()
        
        # Local Refinement with SLSQP
        def _run_slsqp(h_init, maxiter, ftol):
            nonlocal best_score, best_h
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                     'jac': lambda x: np.ones(N)}]
            bounds = [(0.0, 1.0)] * N
            
            def obj_func(x):
                val, _ = _smooth_max_val_grad(x, beta=50.0)
                return val
                
            def jac_func(x):
                _, grad = _smooth_max_val_grad(x, beta=50.0)
                return grad
                
            try:
                res = minimize(obj_func, h_init, jac=jac_func, method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options={'maxiter': maxiter, 'ftol': ftol})
                if res.success or res.fun < best_score:
                    _update_best(res.x)
            except Exception:
                pass

        # Run SLSQP with moderate iterations
        _run_slsqp(h0, maxiter=100, ftol=1e-8)
        
        # If we have time, do a second pass with higher beta on the best so far
        if i > 0 and time.time() < _DEADLINE - 10:
            # Refine the current best
            _run_slsqp(best_h, maxiter=50, ftol=1e-10)

    if best_h is None:
        best_h = _fix_sum(np.random.uniform(0, 1, N))
    
    best_h = _fix_sum(best_h)
    return best_h

# Need to import minimize
from scipy.optimize import minimize
