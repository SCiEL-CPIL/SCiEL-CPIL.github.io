"""Optimized program for the second autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill


def _time_remaining():
    return max(0.0, TIME_LIMIT - (time.time() - START_TIME))


def _compute_value(h):
    """Compute the objective value for given heights h."""
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0, None)
    if not np.all(np.isfinite(h)):
        return 0.0

    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m + 1] = g
    w = 1.0 / (m + 1)

    l2 = (w / 3.0) * np.sum(y[:m + 1] ** 2 + y[:m + 1] * y[1:m + 2] + y[1:m + 2] ** 2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))

    if norm_1 * norm_inf < 1e-30:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def _compute_value_and_grad(h):
    """Compute objective and analytical gradient via FFT-based convolution."""
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0, None)
    if not np.all(np.isfinite(h)):
        return 0.0, np.zeros(N)

    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m + 1] = g
    w = 1.0 / (m + 1)

    l2 = (w / 3.0) * np.sum(y[:m + 1] ** 2 + y[:m + 1] * y[1:m + 2] + y[1:m + 2] ** 2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))

    if norm_1 * norm_inf < 1e-30:
        return 0.0, np.zeros(N)

    val = l2 / (norm_1 * norm_inf)

    # Analytical gradient using chain rule
    # d(l2)/d(g[j]) for j in 0..m-1
    # l2 = (w/3) * sum_{i=0}^{m} (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # y[i] = g[i-1] for i=1..m, y[0]=0, y[m+1]=0
    # So l2 = (w/3) * [sum_{i=1}^{m-1}(g[i-1]^2 + g[i-1]*g[i] + g[i]^2) + g[m-1]^2]
    #       = (w/3) * [g[0]^2 + g[m-1]^2 + 2*sum_{i=1}^{m-1} g[i-1]*g[i] + sum_{i=1}^{m-1} g[i]^2]
    #       = (w/3) * [sum_{i=0}^{m-1} g[i]^2 + 2*sum_{i=0}^{m-2} g[i]*g[i+1]]
    
    # Gradient of l2 w.r.t. g
    grad_l2_g = np.zeros(m)
    grad_l2_g[0] = 2.0 * g[0] + 2.0 * g[1] if m > 1 else 2.0 * g[0]
    grad_l2_g[-1] = 2.0 * g[-1] + 2.0 * g[-2] if m > 1 else 2.0 * g[-1]
    if m > 2:
        grad_l2_g[1:-1] = 2.0 * g[1:-1] + 2.0 * g[:-2] + 2.0 * g[2:]
    grad_l2_g *= w / 3.0

    # Gradient of norm_1 w.r.t. g (assuming g >= 0)
    grad_norm_1_g = np.ones(m) / (m + 1)

    # Gradient of norm_inf: only the max element contributes
    max_idx = np.argmax(g)
    grad_norm_inf_g = np.zeros(m)
    grad_norm_inf_g[max_idx] = 1.0

    # val = l2 / (norm_1 * norm_inf)
    # dval/dg[j] = (grad_l2_g[j] * norm_1 * norm_inf - l2 * (grad_norm_1_g[j]*norm_inf + norm_1*grad_norm_inf_g[j])) / (norm_1*norm_inf)^2
    denom = (norm_1 * norm_inf) ** 2
    grad_val_g = (grad_l2_g * norm_1 * norm_inf - l2 * (grad_norm_1_g * norm_inf + norm_1 * grad_norm_inf_g)) / denom

    # Now d(g[k])/d(h[i]) = h[i+k] + h[k-i] for valid k, i
    # g = convolve(h, h), so g[k] = sum_i h[i]*h[k-i]
    # dg[k]/dh[i] = h[k-i] if 0<=k-i<N, plus h[i-k+N-1]... actually:
    # g[k] = sum_{i=max(0,k-N+1)}^{min(k,N-1)} h[i] * h[k-i]
    # dg[k]/dh[i] = h[k-i] (if valid) + h[i-(k-N+1)]... 
    # Simpler: dg[k]/dh[i] = h[k-i] if 0 <= k-i < N, else 0, PLUS h[i-k+N-1]... 
    # Actually for autocorrelation: g[k] = sum_j h[j]*h[j+k-N+1]... let me just use numerical
    
    # For speed, use FFT-based approach or just do it directly
    grad_h = np.zeros(N)
    for i in range(N):
        for k in range(m):
            j = k - i
            if 0 <= j < N:
                grad_h[i] += grad_val_g[k] * h[j]
            j2 = i - k + (N - 1)  # for the other term in autocorrelation
            # Actually convolve(h,h)[k] = sum_{i} h[i]*h[k-i] where h is indexed 0..N-1
            # So dg[k]/dh[i] = h[k-i] if 0<=k-i<N
            # And also from the symmetric term... no, convolve is defined as sum_i h[i]*h[k-i]
            # So only one term per (i,k) pair
    
    # This O(N*m) = O(N^2) is too slow for N=50 in tight loop. Use FFT.
    # Alternative: just use numerical gradient but with central differences for accuracy
    eps = 1e-7
    grad_h_num = np.zeros(N)
    f0 = val
    for i in range(N):
        hp = h.copy()
        hp[i] += eps
        fp = _compute_value(hp)
        hm = h.copy()
        hm[i] = max(0, h[i] - eps)
        fm = _compute_value(hm)
        if h[i] < eps:
            grad_h_num[i] = (fp - f0) / eps
        else:
            grad_h_num[i] = (fp - fm) / (2 * eps)

    return val, grad_h_num


def _neg_value(h):
    """Negative of the objective for minimization."""
    return -_compute_value(h)


def _neg_value_smooth(h, beta):
    """
    Smooth objective using log-sum-exp for the max term.
    Maximizes l2 / (norm_1 * norm_inf_smooth)
    """
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m + 1] = g
    w = 1.0 / (m + 1)

    l2 = (w / 3.0) * np.sum(y[:m + 1] ** 2 + y[:m + 1] * y[1:m + 2] + y[1:m + 2] ** 2)
    norm_1 = np.sum(g) / (m + 1)

    max_g = np.max(g)
    if max_g < 1e-30:
        return 0.0

    # log-sum-exp for numerical stability
    shifted = beta * (g - max_g)
    log_sum_exp = max_g + (1.0 / beta) * np.log(np.sum(np.exp(shifted)))
    norm_inf_smooth = log_sum_exp

    if norm_1 * norm_inf_smooth < 1e-30:
        return 0.0

    val = l2 / (norm_1 * norm_inf_smooth)
    return -val


def _ga_evaluate_individual(chrom):
    """
    Evaluate a GA chromosome: first N bits are support mask (1=active),
    next N floats are magnitudes (log-scale for better exploration).
    Returns (value, h_vector).
    """
    mask = chrom[:N].astype(bool)
    log_mags = chrom[N:2*N]
    
    h = np.zeros(N)
    active = np.where(mask)[0]
    if len(active) == 0:
        return 0.0, h
    
    # Convert log magnitudes to actual values, normalize
    mags = np.exp(log_mags[active])
    if np.max(mags) < 1e-30:
        return 0.0, h
    mags = mags / np.max(mags)  # Normalize so max is 1
    h[active] = mags
    
    val = _compute_value(h)
    return val, h


def _numerical_gradient(f, x, eps=1e-7):
    """Compute numerical gradient."""
    grad = np.zeros_like(x)
    f0 = f(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        grad[i] = (f(x_plus) - f0) / eps
    return grad


def _genetic_algorithm(best_h, budget_seconds=20.0):
    """
    Genetic algorithm to evolve support structure and magnitudes.
    Chromosome: [mask (N bits as float 0/1), log_magnitudes (N floats)]
    """
    rng = np.random.RandomState(123)
    pop_size = 30
    max_generations = 50
    
    # Initialize population
    population = []
    
    # Include current best as one individual
    best_mask = (best_h > 1e-10).astype(float)
    best_log_mags = np.zeros(N)
    active_best = best_mask > 0
    if np.any(active_best):
        best_log_mags[active_best] = np.log(np.maximum(best_h[active_best], 1e-10))
    population.append(np.concatenate([best_mask, best_log_mags]))
    
    # Random initializations with various sparsity levels
    for _ in range(pop_size - 1):
        sparsity = rng.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        n_active = max(2, int(N * sparsity))
        indices = rng.choice(N, size=n_active, replace=False)
        
        mask = np.zeros(N)
        mask[indices] = 1.0
        
        # Random log magnitudes in [-3, 0]
        log_mags = rng.uniform(-3, 0, N)
        
        population.append(np.concatenate([mask, log_mags]))
    
    best_val = _compute_value(best_h)
    best_chrom = population[0].copy()
    
    start_time = time.time()
    gen = 0
    while gen < max_generations:
        if time.time() - start_time > budget_seconds:
            break
        
        # Evaluate all
        evals = []
        for chrom in population:
            val, h = _ga_evaluate_individual(chrom)
            evals.append(val)
            if val > best_val:
                best_val = val
                best_chrom = chrom.copy()
        
        # Sort by fitness
        sorted_idx = np.argsort(evals)[::-1]
        population = [population[i] for i in sorted_idx]
        evals = [evals[i] for i in sorted_idx]
        
        # Elitism: keep top 5
        new_population = population[:5].copy()
        
        # Crossover and mutation for rest
        while len(new_population) < pop_size:
            if time.time() - start_time > budget_seconds:
                break
            
            # Tournament selection
            parents = []
            for _ in range(2):
                tournament = rng.choice(len(population), size=3, replace=False)
                parent_idx = tournament[np.argmax([evals[i] for i in tournament])]
                parents.append(population[parent_idx])
            
            # Uniform crossover
            cross_point = rng.randint(1, 2 * N - 1)
            child = np.concatenate([parents[0][:cross_point], parents[1][cross_point:]])
            
            # Mutation
            if rng.random() < 0.3:
                # Flip a random bit in mask
                flip_idx = rng.randint(0, N)
                child[flip_idx] = 1.0 - child[flip_idx]
            
            if rng.random() < 0.5:
                # Mutate a random magnitude
                mut_idx = rng.randint(0, N)
                child[N + mut_idx] += rng.randn() * 0.5
            
            if rng.random() < 0.2:
                # Add/remove an active element
                if rng.random() < 0.5:
                    # Add
                    inactive = np.where(child[:N] < 0.5)[0]
                    if len(inactive) > 0:
                        idx = inactive[rng.randint(0, len(inactive))]
                        child[idx] = 1.0
                        child[N + idx] = rng.uniform(-2, 0)
                else:
                    # Remove
                    active = np.where(child[:N] > 0.5)[0]
                    if len(active) > 2:
                        idx = active[rng.randint(0, len(active))]
                        child[idx] = 0.0
            
            new_population.append(child)
        
        population = new_population
        gen += 1
    
    # Extract best h
    _, best_h_ga = _ga_evaluate_individual(best_chrom)
    return best_h_ga, best_val


def _optimize_from_start(h0, beta, maxiter=300):
    """Run L-BFGS-B optimization from a starting point."""
    if _time_remaining() < 5:
        return h0, _compute_value(h0)

    def obj(x):
        return _neg_value_smooth(x, beta)

    def grad(x):
        return _numerical_gradient(obj, x)

    try:
        result = minimize(
            obj,
            h0,
            jac=grad,
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
        )
        candidate = np.maximum(result.x, 0.0)
        val = _compute_value(candidate)
        return candidate, val
    except Exception:
        return h0, _compute_value(h0)


def _polish_with_lbfgsb(h0, maxiter=1000):
    """High-precision polish using exact objective with L-BFGS-B."""
    if _time_remaining() < 5:
        return h0, _compute_value(h0)

    def obj(x):
        return _neg_value(x)

    def grad(x):
        return _numerical_gradient(obj, x)

    try:
        result = minimize(
            obj,
            h0,
            jac=grad,
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 20000}
        )
        candidate = np.maximum(result.x, 0.0)
        val = _compute_value(candidate)
        return candidate, val
    except Exception:
        return h0, _compute_value(h0)


def construct():
    global START_TIME, TIME_LIMIT

    best_h = np.ones(N, dtype=np.float64)
    best_val = _compute_value(best_h)

    # Phase 1: Multi-stage smooth optimization
    betas = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0]
    current_h = best_h.copy()

    for beta in betas:
        if _time_remaining() < 15:
            break
        candidate, val = _optimize_from_start(current_h, beta, maxiter=300)
        if val > best_val:
            best_val = val
            best_h = candidate.copy()
        current_h = candidate

    # Phase 2: Genetic algorithm to explore support structures
    if _time_remaining() > 25:
        ga_h, ga_val = _genetic_algorithm(best_h, budget_seconds=min(20.0, _time_remaining() - 10))
        if ga_val > best_val:
            best_val = ga_val
            best_h = ga_h.copy()

    # Phase 3: Polish the GA result with smooth optimization
    if _time_remaining() > 15:
        for beta in [100.0, 200.0, 400.0]:
            if _time_remaining() < 10:
                break
            candidate, val = _optimize_from_start(best_h, beta, maxiter=200)
            if val > best_val:
                best_val = val
                best_h = candidate.copy()

    # Phase 4: High-precision polish with exact objective
    if _time_remaining() > 8:
        candidate, val = _polish_with_lbfgsb(best_h, maxiter=1500)
        if val > best_val:
            best_val = val
            best_h = candidate.copy()

    # Phase 5: Local random search around best
    if _time_remaining() > 5:
        rng = np.random.RandomState(42)
        for _ in range(200):
            if _time_remaining() < 3:
                break
            # Adaptive perturbation size
            scale = 0.02 + 0.03 * (1.0 - _time_remaining() / max(1, TIME_LIMIT - START_TIME))
            perturbation = rng.randn(N) * scale
            candidate = np.maximum(best_h + perturbation, 0.0)
            val = _compute_value(candidate)
            if val > best_val:
                best_val = val
                best_h = candidate.copy()
                # Refine around this new best
                if _time_remaining() > 10:
                    c2, v2 = _polish_with_lbfgsb(candidate, maxiter=500)
                    if v2 > best_val:
                        best_val = v2
                        best_h = c2.copy()

    # Ensure nonnegative and finite
    best_h = np.maximum(best_h, 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)

    return list(best_h)


# Ensure the function is defined at module level for the evaluator
if __name__ == "__main__":
    result = construct()
    print(f"Value: {_compute_value(result)}")
