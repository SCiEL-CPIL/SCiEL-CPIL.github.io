"""
Signed step function minimizing max|autoconvolution| / (sum f)^2.

Structural change: model the signal as a sparse polynomial (few non-zero
terms) and use differential evolution to optimize positions, signs, and
magnitudes. This drastically reduces the effective dimensionality and
allows targeted spectral null construction.
"""
import numpy as np
import time
import random

N = 400
_DEADLINE = 145.0
_T0 = time.monotonic()


def _remaining():
    return _DEADLINE - (time.monotonic() - _T0)


def _autoconv_fft(f):
    """Compute linear autoconvolution via FFT."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.rfft(f, size)
    ac = np.fft.irfft(F * F, size)
    return ac[:2 * n - 1]


def _score(f):
    """Compute max|autoconv| / (sum f)^2. Returns inf if sum f == 0."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv_fft(f)
    return np.max(np.abs(ac)) / (s * s)


def _sparse_to_dense(positions, magnitudes, N):
    """Convert sparse representation (positions, magnitudes) to dense vector."""
    f = np.zeros(N)
    for pos, mag in zip(positions, magnitudes):
        if 0 <= pos < N:
            f[int(pos)] = mag
    return f


def _evaluate_sparse(positions, magnitudes, N):
    """Evaluate score for a sparse representation."""
    f = _sparse_to_dense(positions, magnitudes, N)
    return _score(f)


def _differential_evolution_sparse(objective, bounds, pop_size=50, max_iter=100,
                                    F=0.8, CR=0.9, mutation='rand1bin',
                                    seed=42):
    """
    Differential evolution for sparse polynomial optimization.
    
    Parameters:
    - objective: function taking a vector of parameters and returning score
    - bounds: list of (low, high) for each parameter
    - pop_size: population size
    - max_iter: maximum iterations
    - F: scale factor
    - CR: crossover rate
    - seed: random seed
    
    Returns: (best_params, best_score)
    """
    rng = np.random.RandomState(seed)
    dim = len(bounds)
    bounds = np.array(bounds)
    low, high = bounds[:, 0], bounds[:, 1]
    
    # Initialize population
    pop = low + (high - low) * rng.rand(pop_size, dim)
    scores = np.array([objective(ind) for ind in pop])
    
    best_idx = np.argmin(scores)
    best_params = pop[best_idx].copy()
    best_score = scores[best_idx]
    
    for it in range(max_iter):
        if _remaining() < 15.0:
            break
        
        new_pop = []
        new_scores = []
        
        for i in range(pop_size):
            # Select three distinct random individuals
            a, b, c = rng.choice(pop_size, size=3, replace=False)
            while a == b or a == c or b == c:
                a, b, c = rng.choice(pop_size, size=3, replace=False)
            
            # Differential mutation
            v = pop[a] + F * (pop[b] - pop[c])
            
            # Binomial crossover
            trial = pop[i].copy()
            j_rand = rng.randint(dim)
            for j in range(dim):
                if rng.rand() < CR or j == j_rand:
                    trial[j] = v[j]
            
            # Clip to bounds
            trial = np.clip(trial, low, high)
            
            # Evaluate
            trial_score = objective(trial)
            
            # Selection
            if trial_score < scores[i]:
                new_pop.append(trial)
                new_scores.append(trial_score)
            else:
                new_pop.append(pop[i])
                new_scores.append(scores[i])
        
        pop = np.array(new_pop)
        scores = np.array(new_scores)
        
        # Update best
        best_idx = np.argmin(scores)
        if scores[best_idx] < best_score:
            best_score = scores[best_idx]
            best_params = pop[best_idx].copy()
    
    return best_params, best_score


def _local_polish(f, max_iters=5000):
    """Local polish: try single-bit flips to improve score."""
    ac = _autoconv_fft(f)
    s = np.sum(f)
    if abs(s) < 1e-12:
        return f, np.inf
    cur_score = np.max(np.abs(ac)) / (s * s)
    
    rng = np.random.RandomState(123)
    for _ in range(max_iters):
        if _remaining() < 5.0:
            break
        
        idx = rng.randint(N)
        old_val = f[idx]
        f[idx] = -f[idx]
        
        ac_new = _autoconv_fft(f)
        s_new = np.sum(f)
        if abs(s_new) < 1e-12:
            f[idx] = old_val
            continue
        new_score = np.max(np.abs(ac_new)) / (s_new * s_new)
        
        if new_score < cur_score:
            cur_score = new_score
            ac = ac_new
            s = s_new
        else:
            f[idx] = old_val
    
    return f, cur_score


def construct():
    """
    Returns a signed step height vector of length N minimizing
    max|linear_autoconvolution| / (sum f)^2.
    
    Uses sparse polynomial representation with differential evolution.
    """
    rng = np.random.RandomState(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # Random sequences
    for _ in range(5):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Golay-like complementary sequence construction
    def golay_pair(n):
        if n == 1:
            return np.array([1.0]), np.array([1.0])
        half = n // 2
        a, b = golay_pair(half)
        a2 = np.concatenate([a, b])
        b2 = np.concatenate([a, -b])
        return a2, b2
    
    golay_len = 1
    while golay_len < N:
        golay_len *= 2
    a, b = golay_pair(golay_len)
    if golay_len >= N:
        candidates.append(a[:N].copy())
        candidates.append(b[:N].copy())
        candidates.append(np.concatenate([a[:N//2], b[:N//2]])[:N])
    else:
        reps = (N + golay_len - 1) // golay_len
        candidates.append(np.tile(a, reps)[:N])
        candidates.append(np.tile(b, reps)[:N])
        candidates.append(np.tile(np.concatenate([a, b]), reps)[:N])
    
    # Barker-like sequences
    barker13 = np.array([1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1], dtype=float)
    reps = (N + 12) // 13
    candidates.append(np.tile(barker13, reps)[:N])
    
    # Alternating patterns
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # Evaluate all candidates
    best_f = None
    best_score = np.inf
    for c in candidates:
        s = _score(c)
        if s < best_score:
            best_score = s
            best_f = c.copy()
    
    # Sparse polynomial optimization via differential evolution
    # Try different sparsity levels
    for n_sparse in [10, 20, 30, 40]:
        if _remaining() < 20.0:
            break
        
        # Parameters: positions (n_sparse), magnitudes (n_sparse)
        # Positions in [0, N-1], magnitudes in [-1, 1]
        bounds = []
        for _ in range(n_sparse):
            bounds.append((0, N - 1))  # position
        for _ in range(n_sparse):
            bounds.append((-1.0, 1.0))  # magnitude
        
        def objective(params):
            positions = params[:n_sparse].astype(int)
            magnitudes = params[n_sparse:]
            return _evaluate_sparse(positions, magnitudes, N)
        
        best_params, best_sp_score = _differential_evolution_sparse(
            objective, bounds, pop_size=30, max_iter=50,
            F=0.8, CR=0.9, seed=42
        )
        
        # Convert to dense and polish
        positions = best_params[:n_sparse].astype(int)
        magnitudes = best_params[n_sparse:]
        f_sparse = _sparse_to_dense(positions, magnitudes, N)
        
        # Project to binary
        f_binary = np.sign(f_sparse)
        f_binary[f_binary == 0] = 1.0
        
        # Local polish
        f_polished, sp_score = _local_polish(f_binary, max_iters=2000)
        
        if sp_score < best_score:
            best_score = sp_score
            best_f = f_polished.copy()
    
    # Final greedy pass
    f = best_f.copy()
    ac = _autoconv_fft(f)
    s = np.sum(f)
    if abs(s) < 1e-12:
        return best_f
    cur_score = np.max(np.abs(ac)) / (s * s)
    
    improved = True
    while improved and _remaining() > 2.0:
        improved = False
        for idx in range(N):
            f[idx] = -f[idx]
            ac_new = _autoconv_fft(f)
            s_new = np.sum(f)
            if abs(s_new) < 1e-12:
                f[idx] = -f[idx]
                continue
            new_score = np.max(np.abs(ac_new)) / (s_new * s_new)
            if new_score < cur_score:
                cur_score = new_score
                ac = ac_new
                s = s_new
                improved = True
            else:
                f[idx] = -f[idx]
        if cur_score < best_score:
            best_score = cur_score
            best_f = f.copy()
    
    return best_f
