import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1  # 99
W = 1.0 / (M + 1)  # 1/100
TIME_LIMIT = 150.0


def compute_value(h):
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def _value_neg(h, tau=0.01):
    """Negative value with smooth max approximation for gradient methods."""
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    # Smooth max via log-sum-exp
    g_max = np.max(g)
    if g_max <= 0:
        return 0.0
    shifted = g / g_max
    log_sum = np.log(np.sum(np.exp(shifted / tau - 1.0))) + np.log(g_max)
    norm_inf_smooth = np.exp(log_sum)
    if norm_1 <= 0 or norm_inf_smooth <= 0:
        return 0.0
    return -l2 / (norm_1 * norm_inf_smooth)


def _gradient_descent(h0, lr=0.01, n_iter=500, tau=0.01):
    """Simple projected gradient descent on the smooth surrogate."""
    h = np.maximum(0.0, h0.copy())
    for it in range(n_iter):
        if time.time() > _deadline:
            break
        # Compute gradient numerically
        grad = np.zeros(N)
        eps = 1e-6
        for i in range(N):
            h_plus = h.copy()
            h_plus[i] += eps
            h_minus = h.copy()
            h_minus[i] = max(0.0, h[i] - eps)
            grad[i] = (_value_neg(h_plus, tau) - _value_neg(h_minus, tau)) / (2 * eps)
        # Update
        h -= lr * grad
        h = np.maximum(0.0, h)
        # Adaptive learning rate
        if it % 100 == 0:
            lr *= 0.95
    return h


def _evolve_population(pop, n_gen=50, pop_size=20):
    """Evolutionary strategy: mutate and select."""
    global _deadline
    scored = [(compute_value(np.maximum(0.0, p)), p) for p in pop]
    scored.sort(key=lambda t: -t[0])
    pop = [p.copy() for _, p in scored[:pop_size]]
    
    rng = np.random.RandomState(42)
    for gen in range(n_gen):
        if time.time() > _deadline:
            break
        # Generate children
        children = []
        for _ in range(pop_size * 2):
            parent = rng.choice(pop)
            # Gaussian mutation
            child = parent + rng.normal(0, 0.1, N)
            child = np.maximum(0.0, child)
            # Occasionally do larger jumps
            if rng.random() < 0.1:
                child = parent + rng.normal(0, 0.3, N)
                child = np.maximum(0.0, child)
            children.append(child)
        
        # Score and select
        scored = [(compute_value(np.maximum(0.0, c)), c) for c in children]
        scored.sort(key=lambda t: -t[0])
        pop = [p.copy() for _, p in scored[:pop_size]]
    
    return pop[0], compute_value(pop[0])


def _direct_optimize():
    """
    Direct structural optimization:
    
    Key insight: value = l2 / (norm_1 * norm_inf)
    
    Since h >= 0, g = convolve(h,h) >= 0, so:
    - norm_1 = sum(g) / (m+1) = (sum(h))^2 / (m+1)
    - norm_inf = max(g)
    
    The value is scale-invariant (both numerator and denominator scale as h^2).
    
    To maximize l2 / (sum(g) * max(g)):
    - We want g to be "smooth" (adjacent terms similar) to maximize l2
    - We want g to have low max relative to its sum (spread out, not peaked)
    
    Strategy: try various parametric forms and optimize parameters.
    """
    best_val = 0.0
    best_h = np.ones(N)
    
    x = np.arange(N, dtype=float)
    
    # Form 1: Single Gaussian
    for center in np.linspace(10, 40, 15):
        for width in np.linspace(3, 20, 15):
            h = np.exp(-0.5 * ((x - center) / width) ** 2)
            v = compute_value(h)
            if v > best_val:
                best_val = v
                best_h = h.copy()
    
    # Form 2: Two Gaussians
    for c1 in np.linspace(8, 20, 10):
        for c2 in np.linspace(30, 42, 10):
            for w in np.linspace(3, 12, 8):
                h = np.exp(-0.5 * ((x - c1) / w) ** 2) + \
                    np.exp(-0.5 * ((x - c2) / w) ** 2)
                v = compute_value(h)
                if v > best_val:
                    best_val = v
                    best_h = h.copy()
    
    # Form 3: Cosine window
    for width_factor in np.linspace(0.3, 1.0, 15):
        arg = np.pi * width_factor * (x - (N - 1) / 2) / ((N - 1) / 2)
        h = np.maximum(0.0, np.cos(arg))
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
    
    # Form 4: Power-law bump
    for center in np.linspace(15, 35, 10):
        for r in np.linspace(8, 20, 10):
            for p in np.linspace(0.3, 2.0, 10):
                base = np.maximum(0.0, 1.0 - np.abs(x - center) / r)
                h = base ** p
                v = compute_value(h)
                if v > best_val:
                    best_val = v
                    best_h = h.copy()
    
    # Form 5: Semicircle
    for center in np.linspace(15, 35, 10):
        for radius in np.linspace(8, 20, 10):
            d = (x - center) / radius
            h = np.sqrt(np.maximum(0.0, 1.0 - d ** 2))
            v = compute_value(h)
            if v > best_val:
                best_val = v
                best_h = h.copy()
    
    return best_h, best_val


def _refine(h0, n_iterations=1000):
    """Local refinement using coordinate descent with fine steps."""
    h = np.maximum(0.0, h0.copy())
    best_val = compute_value(h)
    
    for it in range(n_iterations):
        if time.time() > _deadline:
            break
        improved = False
        for i in range(N):
            # Try fine adjustments
            for delta in [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]:
                h_new = h.copy()
                h_new[i] = max(0.0, h[i] + delta)
                v = compute_value(h_new)
                if v > best_val:
                    h = h_new
                    best_val = v
                    improved = True
                    break
        if not improved:
            # Try smaller steps
            for i in range(N):
                for delta in [-0.01, -0.005, -0.002, 0.002, 0.005, 0.01]:
                    h_new = h.copy()
                    h_new[i] = max(0.0, h[i] + delta)
                    v = compute_value(h_new)
                    if v > best_val:
                        h = h_new
                        best_val = v
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    
    return h, best_val


_deadline = None


def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + (TIME_LIMIT - 25.0)  # 25s safety margin
    
    # Phase 1: Direct structural search
    h_init, v_init = _direct_optimize()
    
    # Phase 2: Gradient descent from best initial
    best_h = h_init.copy()
    best_val = v_init
    
    # Try gradient descent with different tautau values
    for tau in [0.005, 0.01, 0.02, 0.05]:
        if time.time() > _deadline:
            break
        h_grad = _gradient_descent(best_h, lr=0.02, n_iter=300, tau=tau)
        v_grad = compute_value(h_grad)
        if v_grad > best_val:
            best_h = h_grad
            best_val = v_grad
    
    # Phase 3: Coordinate refinement
    h_ref, v_ref = _refine(best_h, n_iterations=500)
    if v_ref > best_val:
        best_h = h_ref
        best_val = v_ref
    
    # Phase 4: Evolutionary strategy
    if time.time() < _deadline:
        pop = [best_h.copy() for _ in range(10)]
        # Add some diverse starting points
        x = np.arange(N, dtype=float)
        pop.append(np.ones(N))
        pop.append(0.5 * (1 + np.cos(2 * np.pi * x / (N - 1))))
        for center in [15, 25, 35]:
            for width in [5, 10, 15]:
                pop.append(np.exp(-0.5 * ((x - center) / width) ** 2))
        
        best_evo, v_evo = _evolve_population(pop, n_gen=30, pop_size=15)
        if v_evo > best_val:
            best_h = best_evo
            best_val = v_evo
    
    # Phase 5: Final fine-tuning
    if time.time() < _deadline:
        h_final, v_final = _refine(best_h, n_iterations=300)
        if v_final > best_val:
            best_h = h_final
            best_val = v_final
    
    # Ensure validity
    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    
    return best_h.tolist()


if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
