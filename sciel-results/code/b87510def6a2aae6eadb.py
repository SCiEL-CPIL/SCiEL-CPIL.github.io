import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 100
TARGET_SUM = N / 2.0

def _score(h):
    """Compute the maximum overlap of h with shifted (1-h)."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _overlap_values(h):
    """Compute overlap vector h_i * (1 - h_{i+1}) with boundary h_n = 0."""
    n = len(h)
    h_next = np.zeros(n)
    h_next[:-1] = h[1:]
    return h * (1.0 - h_next)

def _overlap_grad(h):
    """Gradient of overlap vector w.r.t. h.
    overlap_i = h_i * (1 - h_{i+1})
    d(overlap_i)/dh_i = 1 - h_{i+1}
    d(overlap_i)/dh_{i+1} = -h_i
    """
    n = len(h)
    h_next = np.zeros(n)
    h_next[:-1] = h[1:]
    h_prev = np.zeros(n)
    h_prev[1:] = h[:-1]
    
    grad = np.zeros(n)
    grad += (1.0 - h_next)   # from overlap_i
    grad -= h_prev           # from overlap_{i-1}
    return grad

def _smooth_max(x, alpha=200.0):
    """Smooth approximation of max(x)."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _smooth_max_grad(x, alpha=200.0):
    """Gradient of smooth max w.r.t. x (softmax weights)."""
    x_max = np.max(x)
    e = np.exp(alpha * (x - x_max))
    return e / np.sum(e)

def _fix_sum(h):
    """Project h to have sum exactly N/2 while staying in [0,1]."""
    h = h.copy()
    for _ in range(100):
        diff = TARGET_SUM - np.sum(h)
        if abs(diff) < 1e-15:
            break
        slack_up = 1.0 - h
        slack_down = h
        if diff > 0:
            order = np.argsort(-slack_up)
            for i in order:
                if abs(diff) < 1e-15:
                    break
                amt = min(slack_up[i], diff)
                h[i] += amt
                diff -= amt
        else:
            order = np.argsort(-slack_down)
            for i in order:
                if abs(diff) < 1e-15:
                    break
                amt = min(slack_down[i], -diff)
                h[i] -= amt
                diff += amt
    return h

def _two_state_dp(a, b, P1, P2, phase=0):
    """
    Generate a sequence based on a two-state Markov chain structure.
    States: 0 (value a), 1 (value b).
    We construct a periodic pattern with P1 steps of 'a' and P2 steps of 'b'.
    This is effectively a deterministic cycle, but we treat it as a parametric family.
    """
    period = int(P1) + int(P2)
    if period <= 0:
        return np.full(N, 0.5)
    
    # Create one period
    period_h = np.zeros(period)
    period_h[:int(P1)] = a
    period_h[int(P1):] = b
    
    # Tile to fill N
    n_tiles = (N + period - 1) // period
    h = np.tile(period_h, n_tiles)[:N]
    
    # Apply phase
    if phase > 0:
        h = np.roll(h, -int(phase))
        
    return h

def _markov_transition_obj(params):
    """
    Objective for optimizing transition probabilities / pattern parameters.
    params: [a, b, P1, P2, phase]
    """
    a, b, P1, P2, phase = params
    h = _two_state_dp(a, b, P1, P2, phase)
    h = np.clip(h, 0.0, 1.0)
    h = _fix_sum(h)
    return _score(h)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    h_best = np.full(N, 0.5)
    score_best = _score(h_best)

    # Strategy 1: Differential Evolution for Two-State Pattern (Markov-like)
    # This explores the space of (a, b, P1, P2, phase)
    # a, b are the two state values. P1, P2 are run lengths.
    
    def sawtooth_objective(params):
        a, b, P1, P2, phase = params
        # Ensure P1, P2 are integers for the pattern generation
        P1_int = int(round(P1))
        P2_int = int(round(P2))
        if P1_int < 1: P1_int = 1
        if P2_int < 1: P2_int = 1
        if P1_int > N: P1_int = N
        if P2_int > N: P2_int = N
        
        h = _two_state_dp(a, b, P1_int, P2_int, phase)
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        return _score(h)

    bounds = [(0.0, 1.0), (0.0, 1.0), (1.0, 50.0), (1.0, 50.0), (0.0, 100.0)]

    if time.time() < deadline - 55:
        try:
            result = differential_evolution(
                sawtooth_objective,
                bounds=bounds,
                seed=42,
                maxiter=300,
                popsize=20,
                tol=1e-10,
                polish=False,
                init='latinhypercube'
            )
            a, b, P1, P2, phase = result.x
            P1_int = int(round(P1))
            P2_int = int(round(P2))
            h_candidate = _two_state_dp(a, b, P1_int, P2_int, phase)
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
            h_candidate = _fix_sum(h_candidate)
            sc = _score(h_candidate)
            if sc < score_best:
                score_best = sc
                h_best = h_candidate.copy()
        except Exception:
            pass

    # Strategy 2: Epigraph optimization with SLSQP
    # Refine the continuous solution using the full vector h
    if time.time() < deadline - 25:
        n_params = N + 1  # h_0..h_{N-1}, t
        x0 = np.concatenate([h_best, [score_best]])

        def obj(x):
            h = x[:N]
            t = x[N]
            overlap = _overlap_values(h)
            sm = _smooth_max(overlap, alpha=200.0)
            penalty = max(0.0, sm - t) ** 2
            return t + penalty

        def grad(x):
            h = x[:N]
            t = x[N]
            overlap = _overlap_values(h)
            grad_overlap = _overlap_grad(h)
            weights = _smooth_max_grad(overlap, alpha=200.0)
            grad_h = weights * grad_overlap
            sm = _smooth_max(overlap, alpha=200.0)
            penalty = max(0.0, sm - t)
            grad_t = 1.0 - 2.0 * penalty
            return np.concatenate([grad_h, [grad_t]])

        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x[:N]) - TARGET_SUM,
             'jac': lambda x: np.concatenate([np.ones(N), [0.0]])}
        ]
        bnds = [(0.0, 1.0)] * N + [(0.0, 1.0)]

        try:
            res = minimize(
                obj, x0, jac=grad, method='SLSQP',
                bounds=bnds, constraints=cons,
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            h_cand = np.clip(res.x[:N], 0.0, 1.0)
            h_cand = _fix_sum(h_cand)
            sc = _score(h_cand)
            if sc < score_best:
                score_best = sc
                h_best = h_cand.copy()
        except Exception:
            pass

    # Strategy 3: Projected gradient descent with smooth max
    # Further refine with local search
    if time.time() < deadline - 15:
        h = h_best.copy()
        for alpha in [100.0, 200.0, 400.0, 800.0]:
            if time.time() > deadline - 15:
                break
            lr = 0.05
            stuck = 0
            for step in range(1000):
                if time.time() > deadline - 15:
                    break
                h_clipped = np.clip(h, 0.0, 1.0)
                h_clipped = _fix_sum(h_clipped)
                overlap = _overlap_values(h_clipped)
                grad_ov = _overlap_grad(h_clipped)
                weights = _smooth_max_grad(overlap, alpha)
                grad_smooth = weights * grad_ov
                h_new = h_clipped - lr * grad_smooth
                h_new = np.clip(h_new, 0.0, 1.0)
                h_new = _fix_sum(h_new)
                sc_new = _score(h_new)
                if sc_new < score_best - 1e-13:
                    score_best = sc_new
                    h_best = h_new.copy()
                    h = h_new.copy()
                    stuck = 0
                    lr *= 0.997
                else:
                    stuck += 1
                    lr *= 0.92
                    if stuck > 60 or lr < 1e-9:
                        break

    # Strategy 4: Final high-precision polish with Nelder-Mead in Fourier space
    # This allows for non-periodic structures if the two-state assumption is suboptimal
    x = np.arange(N)
    for K in [5, 8, 12]:
        if time.time() > deadline - 5:
            break
        n_params = 2 * K
        coeffs = np.zeros(n_params)
        for k in range(K):
            cos_k = np.cos(2 * np.pi * k * x / N)
            sin_k = np.sin(2 * np.pi * k * x / N)
            coeffs[2*k] = np.dot(h_best - 0.5, cos_k) * 2 / N
            coeffs[2*k+1] = np.dot(h_best - 0.5, sin_k) * 2 / N

        def fourier_obj(p):
            h = np.full(N, 0.5)
            for k in range(K):
                h += p[2*k] * np.cos(2 * np.pi * k * x / N)
                h += p[2*k+1] * np.sin(2 * np.pi * k * x / N)
            h = np.clip(h, 0.0, 1.0)
            h = _fix_sum(h)
            return _score(h)

        try:
            res = minimize(
                fourier_obj, coeffs, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            h_cand = np.full(N, 0.5)
            for k in range(K):
                h_cand += res.x[2*k] * np.cos(2 * np.pi * k * x / N)
                h_cand += res.x[2*k+1] * np.sin(2 * np.pi * k * x / N)
            h_cand = np.clip(h_cand, 0.0, 1.0)
            h_cand = _fix_sum(h_cand)
            sc = _score(h_cand)
            if sc < score_best:
                score_best = sc
                h_best = h_cand.copy()
        except Exception:
            pass

    # Final verification
    h_best = np.clip(h_best, 0.0, 1.0)
    h_best = _fix_sum(h_best)
    return h_best
