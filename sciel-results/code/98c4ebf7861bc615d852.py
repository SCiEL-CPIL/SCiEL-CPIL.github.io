"""
Optimized uncertainty bound minimization using a 2-parameter reparameterization.

Key structural change: Instead of optimizing 3 raw coefficients (a0, a4, a8) where
a12 is determined by the constraint g(0)=0, we reparameterize using the two
independent directions in the null space of the constraint. This reduces the
problem to 2 dimensions, making it easier for optimizers to find the global minimum.

The constraint is: a0 + 3*a4 + 105*a8 + 945*a12 = 0
So a12 = -(a0 + 3*a4 + 105*a8)/945

We can parameterize as:
  a0 = u
  a4 = v
  a8 = w
  a12 = -(u + 3v + 105w)/945

But this is still 3 parameters. However, we can use a different basis:
Let's define new parameters (s, t) such that:
  a0 = s
  a4 = t
  a8 = 0
  a12 = -(s + 3t)/945

Wait, that's only 2 parameters but we lose a8. Let me think differently.

Actually, the 3 free parameters are a0, a4, a8 (with a12 determined). So it's a
3-dimensional optimization problem. The key insight is to use a better
reparameterization or better optimization strategy.

Let me use the approach from the inspiration but with more careful handling and
a better search strategy. The main improvement is:
1. Precompute constants once for speed
2. Use a smarter initialization strategy
3. Use tighter tolerances and more iterations
4. Add a final local polish step
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195  # Safety margin before 210s kill


def _precompute():
    """Precompute all constant arrays for fast evaluation."""
    sp = np.sqrt(np.pi)
    
    # I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
    I = np.zeros(14)
    I[0] = sp
    for n in range(1, 14):
        # Use recurrence: I_n = (2n-1) * I_{n-1} / (2n) ... wait
        # I_n = (2n)! / (4^n n!) * sqrt(pi)
        # I_{n-1} = (2n-2)! / (4^{n-1} (n-1)!) * sqrt(pi)
        # I_n / I_{n-1} = (2n-2+1)(2n-2+2) / (4 * n) = (2n-1)(2n) / (4n) = (2n-1)/2
        I[n] = I[n-1] * (2*n - 1) / 2.0
    
    # Coefficient matrix: p_k = sum_c A[k,c] * a_c + A12[k] * a12
    # where a12 = -(a0 + 3*a4 + 105*a8)/945
    # So p_k = sum_c (A[k,c] - A12[k]/945 * [1,3,105][c]) * a_c
    
    # A[k,c] for c in {0,4,8} (coefficients of x^{2k} in H0, H4, H8)
    A = np.array([
        [1.0,   3.0,  105.0],
        [0.0,  -6.0, -420.0],
        [0.0,   1.0,  210.0],
        [0.0,   0.0,  -28.0],
        [0.0,   0.0,    1.0],
        [0.0,   0.0,    0.0],
        [0.0,   0.0,    0.0]
    ])
    
    # A12[k] = coefficient of x^{2k} in H12
    A12 = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
    
    # Full coefficient matrix: B[k,c] = A[k,c] - A12[k] * [1,3,105][c] / 945
    constraint_vec = np.array([1.0, 3.0, 105.0])
    B = np.zeros((7, 3))
    for c in range(3):
        B[:, c] = A[:, c] - A12 * constraint_vec[c] / 945.0
    
    return I, B


_I, _B = _precompute()


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    params = [a0, a4, a8]
    a12 is determined by the constraint g(0)=0.
    """
    a0, a4, a8 = params[0], params[1], params[2]
    
    # p_k = B[:,c] * a_c
    p = np.zeros(7)
    for c in range(3):
        p += _B[:, c] * params[c]
    
    # m_k = integral x^{2k} |P|^2 e^{-x^2} dx = sum_{i,j} p_i p_j I[k+i+j]
    m0 = 0.0
    m2 = 0.0
    for i in range(7):
        pi = p[i]
        if pi == 0.0:
            continue
        for j in range(7):
            idx0 = i + j
            if idx0 < 14:
                m0 += pi * p[j] * _I[idx0]
            idx2 = 2 + i + j
            if idx2 < 14:
                m2 += pi * p[j] * _I[idx2]
    
    # For Fourier transform: q_k = p_k * (-1)^k
    q = np.zeros(7)
    for k in range(7):
        q[k] = p[k] * ((-1.0) ** k)
    
    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n0 = 0.0
    n2 = 0.0
    for i in range(7):
        qi = q[i]
        if qi == 0.0:
            continue
        for j in range(7):
            idx0 = i + j
            if idx0 < 14:
                n0 += qi * q[j] * _I[idx0]
            idx2 = 2 + i + j
            if idx2 < 14:
                n2 += qi * q[j] * _I[idx2]
    
    n0 *= 2.0 * np.pi
    n2 *= 2.0 * np.pi
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    if not np.isfinite(bound):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    # Compute gradient analytically
    grad = np.zeros(3)
    for c in range(3):
        dp = _B[:, c]
        
        # dm_k/da_c = 2 * sum_i dp_i * (sum_j p_j I[k+i+j])
        dm0 = 0.0
        dm2 = 0.0
        for i in range(7):
            dpi = dp[i]
            if dpi == 0.0:
                continue
            inner0 = 0.0
            inner2 = 0.0
            for j in range(7):
                idx0 = i + j
                if idx0 < 14:
                    inner0 += p[j] * _I[idx0]
                idx2 = 2 + i + j
                if idx2 < 14:
                    inner2 += p[j] * _I[idx2]
            dm0 += 2.0 * dpi * inner0
            dm2 += 2.0 * dpi * inner2
        
        # dq_k/da_c = dp_k * (-1)^k
        dq = np.zeros(7)
        for k in range(7):
            dq[k] = dp[k] * ((-1.0) ** k)
        
        # dn_k/da_c = 2 * 2*pi * sum_i dq_i * (sum_j q_j I[k+i+j])
        dn0 = 0.0
        dn2 = 0.0
        for i in range(7):
            dqi = dq[i]
            if dqi == 0.0:
                continue
            inner0 = 0.0
            inner2 = 0.0
            for j in range(7):
                idx0 = i + j
                if idx0 < 14:
                    inner0 += q[j] * _I[idx0]
                idx2 = 2 + i + j
                if idx2 < 14:
                    inner2 += q[j] * _I[idx2]
            dn0 += 2.0 * 2.0 * np.pi * dqi * inner0
            dn2 += 2.0 * 2.0 * np.pi * dqi * inner2
        
        dvm2 = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn2 = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    if not np.all(np.isfinite(grad)):
        grad = np.array([0.0, 0.0, 0.0])
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    """
    Tune a Fourier-invariant Hermite-polynomial times Gaussian test function to reduce
    the upper bound on A(f)*A(Fourier(f)).
    
    Returns three coefficients (c0, c4, c8) of the Hermite combination.
    The degree-12 weight is fixed so g(0) = 0.
    """
    elapsed = time.time() - _START_TIME
    deadline = _START_TIME + 195
    
    # Seed solution from prior work
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Generate diverse starting points
    starts = [x0.copy()]
    
    # Structured starts around the seed
    rng = np.random.default_rng(42)
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for i in range(5):
            starts.append(x0 + rng.normal(0, scale, 3))
    
    # Some random starts
    for i in range(30):
        starts.append(rng.normal(0, 1.0, 3))
    
    # Some targeted starts
    starts.append(np.array([0.0, 0.0, 0.0]))
    starts.append(np.array([-1.0, 0.05, 0.001]))
    starts.append(np.array([-2.0, 0.1, 0.002]))
    starts.append(np.array([-0.5, 0.02, 0.0005]))
    starts.append(np.array([-1.5, 0.08, 0.0015]))
    starts.append(np.array([1.0, -0.05, 0.0005]))
    starts.append(np.array([-3.0, 0.15, 0.003]))
    starts.append(np.array([-0.8, 0.03, 0.0008]))
    starts.append(np.array([-1.2, 0.06, 0.0012]))
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: L-BFGS-B with analytic gradient from multiple starts
    for start in starts:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Polish with Nelder-Mead from best
    if best_params is not None and time.time() < deadline - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 100000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: More random restarts with L-BFGS-B if time permits
    if time.time() < deadline - 20:
        for i in range(50):
            if time.time() > deadline - 10:
                break
            x_rand = rng.normal(0, 1.5, 3)
            try:
                res = minimize(_objective_grad, x_rand, method='L-BFGS-B',
                              jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
        
        # Final polish
        if best_params is not None and time.time() < deadline - 5:
            try:
                res = minimize(_objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 200000, 'xatol': 1e-18, 'fatol': 1e-20})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    
    if best_params is not None:
        return best_params
    return x0

# EVOLVE-BLOCK-END
