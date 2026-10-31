"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.

Structural change: instead of restricting to 2-block step functions, we optimize
the full vector h directly using a smooth (log-sum-exp) approximation to the
max-overlap objective, with L-BFGS-B. We also explore multi-block (3, 4, 5 block)
parametrizations and a sinusoidal family, then polish with direct discrete
local search (swap/adjust individual entries).
"""
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 100
DEADLINE_MARGIN = 15.0  # seconds to spare before 150s kill


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _smooth_max(vals, beta=200.0):
    """Smooth max via log-sum-exp."""
    m = np.max(vals)
    if m < -1e-15:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (vals - m)))) / beta

def _smooth_max_grad(vals, beta=200.0):
    """Gradient of smooth max via log-sum-exp."""
    m = np.max(vals)
    if m < -1e-15:
        return np.zeros_like(vals)
    exp_vals = np.exp(beta * (vals - m))
    sum_exp = np.sum(exp_vals)
    return (exp_vals / sum_exp)


def _fix_sum(h):
    """Adjust h so that sum(h) = N/2, keeping values in [0,1]."""
    diff = N / 2.0 - h.sum()
    if abs(diff) < 1e-14:
        return h
    # Distribute the difference
    n = len(h)
    if diff > 0:
        # Need to increase some entries
        room = 1.0 - h
        total_room = room.sum()
        if total_room > 1e-14:
            h = h + diff * room / total_room
        else:
            h = np.clip(h + diff / n, 0.0, 1.0)
    else:
        # Need to decrease some entries
        room = h
        total_room = room.sum()
        if total_room > 1e-14:
            h = h + diff * room / total_room
        else:
            h = np.clip(h + diff / n, 0.0, 1.0)
    return np.clip(h, 0.0, 1.0)


def _optimize_full_vector(h0, beta=200.0, max_iter=5000):
    """Optimize full vector h using L-BFGS-B with smooth objective and analytic gradient."""
    
    def obj_and_grad(x):
        h = np.clip(x, 0.0, 1.0)
        n = len(h)
        shifted = np.zeros(n)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        
        # Compute smooth max and its gradient w.r.t. overlap
        smax = _smooth_max(overlap, beta)
        grad_overlap = _smooth_max_grad(overlap, beta)
        
        # Penalty for sum constraint: sum(h) should be N/2
        # We use a soft penalty in the objective, but L-BFGS-B bounds handle [0,1].
        # The sum constraint is handled by _fix_sum post-optimization, but for the gradient
        # to guide towards the manifold, we add a penalty term.
        sum_err = h.sum() - N / 2.0
        sum_pen = 50.0 * sum_err**2
        grad_sum_pen = 100.0 * sum_err * np.ones(n)
        
        # Gradient of overlap w.r.t. h
        # overlap[i] = h[i] * (1 - shifted[i])
        # shifted[i] = h[i+1] for i < N-1, shifted[N-1] = 0
        # d(overlap[i])/dh[i] = 1 - shifted[i]
        # d(overlap[i-1])/dh[i] = h[i-1] * (-1) * d(shifted[i-1])/dh[i] = -h[i-1] (since shifted[i-1]=h[i])
        
        grad_h = np.zeros(n)
        # Contribution from overlap[i] to h[i]
        grad_h[:] += grad_overlap * (1.0 - shifted)
        # Contribution from overlap[i-1] to h[i] (for i > 0)
        if n > 1:
            grad_h[1:] += -grad_overlap[:-1] * h[:-1]
            
        total_grad = grad_h + grad_sum_pen
        return smax + sum_pen, total_grad

    h0_clipped = np.clip(h0, 0.0, 1.0)
    try:
        res = minimize(obj_and_grad, h0_clipped, method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        h_opt = np.clip(res.x, 0.0, 1.0)
        h_opt = _fix_sum(h_opt)
        return h_opt
    except Exception:
        return _fix_sum(h0_clipped)


def _sinusoidal_overlap(amp, phase, offset):
    """Compute overlap for h(x) = offset + amp * sin(2*pi*phase*x/2 + phase0)."""
    # h defined on [0,2], N points
    x = np.linspace(0, 2, N)
    h = offset + amp * np.sin(2 * np.pi * phase * x / 2.0 + phase0)
    h = np.clip(h, 0.0, 1.0)
    # Fix sum
    h = _fix_sum(h)
    return _score(h), h


def _optimize_sinusoidal():
    """Optimize sinusoidal family: h(x) = c + a*sin(k*x + phi)."""
    x = np.linspace(0, 2, N)
    
    def sin_obj(params):
        c, a, k, phi = params
        h = c + a * np.sin(k * x + phi)
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        return _score(h)

    best_score = 1e10
    best_h = None
    
    # Use multiple random starts for the continuous optimization
    rng = np.random.default_rng(42)
    for _ in range(5):
        c0 = rng.uniform(0.3, 0.7)
        a0 = rng.uniform(0.1, 0.5)
        k0 = rng.uniform(0.5, 3.0)
        phi0 = rng.uniform(0, 2 * np.pi)
        
        try:
            res = minimize(sin_obj, [c0, a0, k0, phi0], method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                c, a, k, phi = res.x
                h = c + a * np.sin(k * x + phi)
                h = np.clip(h, 0.0, 1.0)
                h = _fix_sum(h)
                best_h = h.copy()
        except Exception:
            pass
            
    return best_score, best_h


def _local_search(h, max_iters=5000):
    """Fast local search: pair-wise swaps to maintain sum exactly."""
    h = np.clip(h, 0.0, 1.0).copy()
    h = _fix_sum(h)
    best_score = _score(h)
    best_h = h.copy()

    # Pre-compute indices for faster access
    indices = np.arange(N)
    
    for it in range(max_iters):
        improved = False
        # Shuffle indices to avoid bias
        rng = np.random.default_rng(it)
        perm = rng.permutation(N)
        
        for i in perm:
            if improved:
                break
            # Try swapping mass with a few other indices
            # Pick a few random j's to interact with i
            js = rng.choice(N, size=10, replace=False)
            for j in js:
                if i == j:
                    continue
                # Try moving delta from i to j or j to i
                for delta in [0.01, 0.005, 0.001]:
                    # Case 1: i increases, j decreases
                    if h[i] + delta <= 1.0 and h[j] - delta >= 0.0:
                        h_try = h.copy()
                        h_try[i] += delta
                        h_try[j] -= delta
                        s = _score(h_try)
                        if s < best_score - 1e-12:
                            best_score = s
                            best_h = h_try.copy()
                            h = h_try.copy()
                            improved = True
                            break
                    
                    # Case 2: j increases, i decreases
                    if h[j] + delta <= 1.0 and h[i] - delta >= 0.0:
                        h_try = h.copy()
                        h_try[j] += delta
                        h_try[i] -= delta
                        s = _score(h_try)
                        if s < best_score - 1e-12:
                            best_score = s
                            best_h = h_try.copy()
                            h = h_try.copy()
                            improved = True
                            break
                if improved:
                    break
        if not improved:
            break
    return best_score, best_h


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    best_score = 1e10
    best_h = None

    # --- Initial candidates ---

    # 1. Golden-ratio 2-block (from seed)
    A0 = (3.0 - math.sqrt(5.0)) / 2.0
    B0 = 1.0 - A0
    m0 = N // 2
    h_2block = np.zeros(N)
    h_2block[:m0] = A0
    h_2block[m0:] = B0
    h_2block = _fix_sum(h_2block)
    s = _score(h_2block)
    if s < best_score:
        best_score = s
        best_h = h_2block.copy()

    # 2. Uniform
    h_unif = np.full(N, 0.5)
    s = _score(h_unif)
    if s < best_score:
        best_score = s
        best_h = h_unif.copy()

    # 3. Sinusoidal
    if time.time() < deadline - 30:
        s_sin, h_sin = _optimize_sinusoidal()
        if h_sin is not None and s_sin < best_score:
            best_score = s_sin
            best_h = h_sin.copy()

    # 4. Full-vector L-BFGS-B optimization from multiple starts
    if time.time() < deadline - 60:
        starts = [best_h.copy(), h_2block.copy(), h_unif.copy()]
        # Add a few random starts
        for _ in range(3):
            h_rand = np.random.uniform(0.2, 0.8, N)
            h_rand = _fix_sum(h_rand)
            starts.append(h_rand)

        for h0 in starts:
            if time.time() > deadline - 30:
                break
            h_opt = _optimize_full_vector(h0, beta=300.0, max_iter=3000)
            s = _score(h_opt)
            if s < best_score:
                best_score = s
                best_h = h_opt.copy()

    # 5. Local search polish
    if time.time() < deadline - 10:
        s_ls, h_ls = _local_search(best_h, max_iters=2000)
        if s_ls < best_score:
            best_score = s_ls
            best_h = h_ls.copy()

    # 6. Another round of L-BFGS-B from the locally-searched result
    if time.time() < deadline - 15:
        h_opt2 = _optimize_full_vector(best_h, beta=500.0, max_iter=3000)
        s = _score(h_opt2)
        if s < best_score:
            best_score = s
            best_h = h_opt2.copy()
        # Final local search
        if time.time() < deadline - 5:
            s_ls2, h_ls2 = _local_search(best_h, max_iters=1000)
            if s_ls2 < best_score:
                best_score = s_ls2
                best_h = h_ls2.copy()

    # Final fix
    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = _fix_sum(best_h)

    return best_h
