import numpy as np
import time
from scipy.optimize import linprog

N = 50
DELTA = 0.5 / N

# Precompute the convolution matrix C such that (h*h)_k = sum_i h_i * h_{i+k-N+1} approx?
# Actually, discrete convolution: c_k = sum_i h_i h_{k-i}
# For full convolution of length 2N-1.
# We want to minimize max_k c_k.
# c_k = sum_i sum_j h_i h_j [i+j == k]
# This is a quadratic form h^T A_k h where A_k is a symmetric matrix with 1s at (i, k-i) and (k-i, i).
# Minimizing max_k h^T A_k h is not a standard QP directly because of the max.
# However, we can fix the peak M and check feasibility:
# Minimize sum(h^2) subject to:
# 1. sum(h) = 1
# 2. h_i >= 0
# 3. h^T A_k h <= M for all k

# Wait, h^T A_k h is quadratic. scipy.optimize.minimize can handle this with SLSQP,
# but the "inspiration" mentioned using analytic gradients for convex reformulation.
# Actually, minimizing sum(h^2) subject to quadratic constraints is a QP if the constraints are linear.
# But h^T A_k h is not linear.
# 
# Let's look at the structure again.
# If we restrict h to be non-negative, the max of the autoconvolution is achieved at some lag.
# Is there a simpler structure?
# 
# Alternative approach: 
# The objective is R(h) = ||h||_2^2 / (||h||_1 * max(h*h)).
# Since R is scale-invariant, let ||h||_1 = 1.
# Then R(h) = ||h||_2^2 / max(h*h).
# We want to maximize this.
# Equivalent to: Minimize max(h*h) subject to ||h||_2^2 >= 1 and ||h||_1 = 1? No, that's not right.
# We want to maximize ratio. 
# Let f(h) = max(h*h). We want to minimize f(h) for a fixed ||h||_2^2?
# No, we want to maximize ||h||_2^2 / f(h).
# This is equivalent to finding the smallest possible peak f(h) for a given "energy" ||h||_2^2.
# Or, fixing ||h||_1 = 1, we want to minimize f(h) while keeping ||h||_2^2 as large as possible?
# Actually, for a fixed support, making h more uniform increases ||h||_2^2? No, uniform minimizes ||h||_2^2 for fixed L1.
# Spike maximizes ||h||_2^2 but also maximizes f(h).
#
# Let's use the binary search on the score S.
# We want to know if there exists h such that:
# ||h||_2^2 / (||h||_1 * max(h*h)) >= S
# Let ||h||_1 = 1.
# ||h||_2^2 >= S * max(h*h)
# This is still non-convex (quadratic >= quadratic * constant).
#
# Let's try the dual perspective or a specific structure.
# The inspiration mentioned: "Solving the convex reformulation (minimize peak autoconvolution subject to L1=1) using SLSQP with analytic gradients allows finding global or near-global optima".
# Wait, if we minimize peak autoconvolution subject to L1=1, we get the "flattest" possible convolution.
# Let h* be the minimizer of max(h*h) subject to sum(h)=1, h>=0.
# Let M_min = max(h* * h*).
# Let L2_min = ||h*||_2^2.
# Is this the optimal solution for the ratio?
# Ratio = L2^2 / M.
# If we minimize M, we also change L2.
# Usually, spreading mass out reduces M (peak) but also reduces L2^2 (concentration).
# We need to balance them.
#
# However, there is a known result for this type of problem (autocorrelation inequality).
# Often, the optimal shape is related to a "bump" or a specific sparse set.
#
# Let's try a different structural change: 
# Use **Gradient Descent on the Ratio** with a **Log-Barrier** for non-negativity and **L1 normalization** handled by reparameterization.
# Or better: Use **Alternating Direction Method of Multipliers (ADMM)** or simple **Projected Gradient Descent** on the objective function directly, but with a **random restart** strategy that samples from a **Gaussian Process** prior?
#
# No, the most promising structural change based on the "inspiration" is to use **SLSQP with analytic gradients** on the convex subproblem.
# Let's define the problem as:
# Minimize M
# Subject to:
# sum(h) = 1
# h >= 0
# h^T A_k h <= M for all k
#
# This is a non-convex problem because of the quadratic constraints. 
# BUT, if we fix the support of h to be small, or if we assume h is "nice", maybe it behaves well.
#
# Actually, let's look at the score 1.306 result. It likely used a specific initialization or a different formulation.
#
# Let's try a **Genetic Algorithm** or **Simulated Annealing** on the **indices** of a sparse vector, but with **continuous heights** optimized via SLSQP for each mask.
# Strategy: 
# 1. Define a set of candidate masks (subsets of indices).
# 2. For each mask, solve the QP: Minimize M subject to sum(h_i)=1, h_i=0 for i not in mask, h_i>=0, and h^T A_k h <= M.
#    Wait, minimizing M with quadratic constraints is hard.
#
# Let's flip it:
# Maximize L2^2 subject to M <= 1, sum(h)=1, h>=0.
# This is also non-convex.
#
# What if we use **Nelder-Mead** (derivative-free) but with a **very smart initialization** based on **Fourier features**?
# The autoconvolution peak is related to the integral of the squared Fourier transform (Parseval).
# max(h*h) >= (1/2N) * (sum h)^2 = 1/(2N) if sum=1.
#
# Let's try **Evolutionary Strategy** (ES) on the vector h.
# It is robust to non-convexity.
#
# Implementation Plan:
# 1. Define objective: - (L2^2 / (L1 * M))
# 2. Use `scipy.optimize.differential_evolution` or a custom CMA-ES? 
#    `differential_evolution` is available in SciPy and is robust for global optimization of non-smooth/non-convex functions.
#    It handles bounds well.
# 3. To speed it up, we can reduce the dimensionality by assuming symmetry or using a lower-dimensional parameterization (e.g., Gaussian mixture).
#    But the task asks for a structural change. Global optimization via DE is a different family from local SLSQP.
#
# Let's combine:
# 1. Use `differential_evolution` to find a good global candidate.
# 2. Polish with SLSQP.
#
# To make it faster, we can limit the DE population and iterations, and use a coarse grid?
# DE is stochastic.
#
# Let's try a **Randomized Coordinate Descent** or **Random Restart Local Search** but with a **Convex Relaxation** hint?
#
# Actually, the "convex reformulation" hint is key.
# Minimize M subject to L1=1.
# If we solve this, we get the "flattest" convolution.
# Let h_flat be the solution.
# Then we can scale h_flat? No, scaling doesn't change the ratio.
# The ratio is fixed for h_flat.
# Is h_flat the optimal solution?
# Let's test this hypothesis in the code.
# If minimizing M gives a low L2^2, the ratio might be low.
# We want to maximize L2^2 / M.
#
# Let's try maximizing L2^2 subject to M <= M_target.
# We can binary search on M_target.
# For a fixed M_target, maximizing L2^2 subject to quadratic constraints is still non-convex.
#
# HOWEVER, if we restrict h to be **non-negative**, the problem of minimizing the peak of the autoconvolution for a fixed L1 norm is a known problem.
# The optimal solution is often a **uniform distribution on a Sidon set** or similar.
#
# Let's try a **Hybrid Strategy**:
# 1. Generate a diverse set of initial points using different heuristics (Sidon, Gaussian, Random).
# 2. For each point, run a **local SLSQP** optimization on the **log-objective** with **analytic gradients**.
#    The analytic gradient of the ratio will guide the optimizer much faster than numerical gradients.
#
# Let's implement the analytic gradient of the score function.
# Score S(h) = L2^2 / (L1 * M)
# Let L1 = sum(h), L2_sq = sum(h^2), M = max_k (h*h)_k.
# Note: M is not differentiable everywhere (max function).
# But we can approximate the gradient using the subgradient of the max (which is the convolution value at the peak index).
#
# Gradient of L2_sq: 2h
# Gradient of L1: 1
# Gradient of M: Let k* be the index of the peak. 
#   M = sum_i sum_j h_i h_j delta(i+j, k*)
#   dM/dh_i = sum_j h_j delta(i+j, k*) + sum_j h_j delta(j+i, k*) 
#            = sum_j h_{k*-i} + sum_j h_{k*-j} (careful with indices)
#   Actually, (h*h)_k = sum_i h_i h_{k-i}.
#   d(h*h)_k / dh_i = h_{k-i} + h_{k-i} (if i != k-i) or 2h_i (if i = k-i)?
#   No, derivative of sum_i h_i h_{k-i} w.r.t h_i:
#   Term i: h_{k-i}
#   Term k-i: h_i
#   So grad_k[h]_i = h_{k-i} + h_{k-i} ? No.
#   Let's derive:
#   f(h) = sum_p h_p h_{k-p}
#   df/dh_i = h_{k-i} + h_{k-i} (from p=i and p=k-i terms, assuming i != k-i).
#   If i = k-i, then term is h_i^2, derivative is 2h_i.
#   So grad_M[h]_i = h_{k*-i} + h_{k*-i} = 2 * h_{k*-i} (with bounds check).
#
# Then use the quotient rule for the gradient of S.
#
# This analytic gradient will make SLSQP much more effective.
#
# Structural Change: **Analytic Gradient-Based Optimization with Multi-Start** vs. Numerical Gradient/Heuristic Search.
# The current program uses SLSQP but likely relies on numerical gradients or less efficient initializations.
# I will implement the analytic gradient and a more aggressive multi-start strategy.

def _score_and_grad(h):
    """
    Returns score and gradient of the score.
    Score = L2^2 / (L1 * M)
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0, np.zeros_like(h)
    
    l2_sq = np.sum(h**2)
    
    # Compute autoconvolution
    h_conv = np.convolve(h, h, mode='full')
    k_max = np.argmax(h_conv)
    m_val = h_conv[k_max]
    
    if m_val == 0:
        return 0.0, np.zeros_like(h)
    
    score = l2_sq / (l1 * m_val)
    
    # Gradient calculation
    # S = A / (B * C) where A = L2^2, B = L1, C = M
    # dS/dh = (dA * B * C - A * (dB * C + B * dC)) / (B * C)^2
    #       = (dA * C - A * (dB * C + B * dC) / B) / (B * C^2) ?
    # Let's use log-derivative or just quotient rule carefully.
    # S = A * (B*C)^-1
    # dS = dA * (B*C)^-1 + A * (-1) * (B*C)^-2 * (dB*C + B*dC)
    # dS = (dA * B*C - A * (dB*C + B*dC)) / (B^2 * C^2)
    
    # dA = 2 * h
    # dB = 1
    # dC = grad_M
    
    # Construct grad_M
    # k_max is in [0, 2N-2]
    # h_conv[k] = sum_i h_i h_{k-i}
    # Indices i must be in [0, N-1] and k-i in [0, N-1]
    grad_m = np.zeros(N)
    for i in range(N):
        j = k_max - i
        if 0 <= j < N:
            # Contribution from term h_i * h_j
            # If i == j, derivative of h_i^2 is 2h_i.
            # If i != j, derivative of h_i*h_j + h_j*h_i is h_j + h_i.
            if i == j:
                grad_m[i] += 2 * h[i]
            else:
                grad_m[i] += h[j]
                # The other term h_j * h_i contributes to grad_m[j], not grad_m[i].
                # Wait, the sum is over p.
                # f = sum_p h_p h_{k-p}
                # df/dh_i = h_{k-i} (from p=i) + h_{k-i} (from p=k-i, if k-i != i)
                # So if i != k-i, grad is h_{k-i} + h_{k-i} = 2 h_{k-i}.
                # If i == k-i, grad is 2 h_i.
                # So in all cases, grad_m[i] = 2 * h[k_max - i] if k_max - i is valid.
                # Let's re-verify.
                # If i != j:
                # Term p=i: h_i h_j -> deriv h_j
                # Term p=j: h_j h_i -> deriv h_i
                # So grad_m[i] gets h_j from p=i term.
                # grad_m[j] gets h_i from p=j term.
                # So my loop above is slightly wrong if I just add h[j].
                # I need to add h[j] to grad_m[i].
                # And h[i] to grad_m[j].
                pass
        
    # Let's do it cleanly:
    grad_m = np.zeros(N)
    for i in range(N):
        j = k_max - i
        if 0 <= j < N:
            if i == j:
                grad_m[i] += 2 * h[i]
            else:
                grad_m[i] += h[j]
                grad_m[j] += h[i]
    
    # Now compute gradient of S
    # dS = ( (2h) * (l1 * m_val) - l2_sq * (1 * m_val + l1 * grad_m) ) / (l1^2 * m_val^2)
    
    numerator = (2 * h) * (l1 * m_val) - l2_sq * (m_val + l1 * grad_m)
    denominator = (l1 * m_val) ** 2
    
    if denominator == 0:
        grad_s = np.zeros_like(h)
    else:
        grad_s = numerator / denominator
        
    return score, grad_s

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    best_h = np.ones(N)
    best_score = _score_and_grad(best_h)[0]
    
    # Optimization function for SLSQP
    # We maximize score, so minimize -score.
    # Constraint: sum(h) = 1 (to fix scale)
    # Bounds: h >= 0
    
    def neg_score_h(h):
        s, _ = _score_and_grad(h)
        return -s
        
    def neg_grad_h(h):
        _, g = _score_and_grad(h)
        return -g
        
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0, 'jac': lambda h: np.ones(N)}
    ]
    bounds = [(0.0, 1.0)] * N
    
    # Generate initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Gaussians
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N/2, 10):
            h_g = np.exp(-0.5 * ((np.arange(N) - c) / w)**2)
            if np.sum(h_g) > 0:
                h_g /= np.sum(h_g)
                inits.append(h_g)
                
    # 3. Triangles
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N, 10):
            h_t = np.maximum(0, 1 - np.abs(np.arange(N) - c) / w)
            if np.sum(h_t) > 0:
                h_t /= np.sum(h_t)
                inits.append(h_t)
                
    # 4. Random Sparse (Sidon-like)
    for _ in range(50):
        num_peaks = np.random.randint(5, 15)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_s = np.zeros(N)
        h_s[indices] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # 5. Random Smooth
    for _ in range(50):
        h_r = np.random.rand(N)
        h_r = np.convolve(h_r, np.ones(5)/5, mode='same')
        h_r /= np.sum(h_r)
        inits.append(h_r)

    # Optimize from each init
    for h_init in inits:
        if time.time() > deadline:
            break
            
        h_init = np.maximum(h_init, 0)
        if np.sum(h_init) == 0:
            continue
        h_init /= np.sum(h_init)
        
        try:
            res = minimize(
                neg_score_h, 
                h_init, 
                jac=neg_grad_h, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints, 
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            
            if res.success:
                h_candidate = res.x
                h_candidate = np.maximum(h_candidate, 0)
                if np.sum(h_candidate) > 0:
                    h_candidate /= np.sum(h_candidate)
                    s_candidate, _ = _score_and_grad(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate
        except Exception:
            pass

    # Final polish
    if time.time() < deadline - 5:
        h_start = best_h / np.sum(best_h)
        try:
            res = minimize(
                neg_score_h, 
                h_start, 
                jac=neg_grad_h, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints, 
                options={'maxiter': 500, 'ftol': 1e-15}
            )
            if res.success:
                h_candidate = res.x
                h_candidate = np.maximum(h_candidate, 0)
                if np.sum(h_candidate) > 0:
                    h_candidate /= np.sum(h_candidate)
                    s_candidate, _ = _score_and_grad(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate
        except Exception:
            pass

    return list(best_h)

from scipy.optimize import minimize # Ensure imported at top level for clarity, though it's in the code block above implicitly via module import if I put it there. I'll put it at the top of the file.
