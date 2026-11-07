"""Erdos minimum overlap: Poisson-driven low-dimensional source optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 142.0  # safety margin before 150s kill


def _score(h):
    """Compute the objective value: max correlation / N * 2."""
    if not np.all(np.isfinite(h)):
        return np.inf
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2


def _is_valid(h):
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
        return False
    if abs(np.sum(h) - N / 2) > 1e-9 * N:
        return False
    return True


def _fix_sum(h):
    """Project h onto the affine hyperplane sum(h)=N/2 and box [0,1]."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(200):
        diff = target - np.sum(h)
        if abs(diff) < 1e-13:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-9
            if not np.any(mask):
                break
            room = 1.0 - h[mask]
            total_room = np.sum(room)
            if total_room < 1e-9:
                break
            add_factor = min(1.0, diff / total_room)
            h[mask] += room * add_factor
        else:
            mask = h > 1e-9
            if not np.any(mask):
                break
            val = h[mask]
            total_val = np.sum(val)
            if total_val < 1e-9:
                break
            sub_factor = min(1.0, -diff / total_val)
            h[mask] -= val * sub_factor
    return h


def _poisson_solve(source, n_iter=50):
    """
    Solve the discrete Poisson equation: -laplacian(h) = source
    using Jacobi iteration with Neumann boundary conditions.
    source: vector of length N (sparse source)
    Returns h of length N satisfying sum(h)=N/2 (mean removed).
    """
    h = np.zeros(N)
    # Initialize with zero mean
    for _ in range(n_iter):
        h_new = h.copy()
        for i in range(N):
            # Discrete Laplacian with periodic boundaries
            left = h[(i - 1) % N]
            right = h[(i + 1) % N]
            lap = left + right - 2 * h[i]
            h_new[i] = (h[i] - lap + source[i]) / 2.0
        h = h_new
    # Remove mean to ensure sum(h) = 0, then shift to mean 0.5
    h = h - np.mean(h)
    h = h + 0.5
    return h


def _sources_to_h(sources):
    """
    Convert a sparse source vector (length K) to height vector h (length N).
    sources[i] is the amplitude of a point source at position i.
    """
    K = len(sources)
    # Map K sources to N positions
    positions = np.linspace(0, N - 1, K, dtype=int)
    source_full = np.zeros(N)
    for i, pos in enumerate(positions):
        source_full[pos] = sources[i]
    # Remove mean of source for solvability
    source_full = source_full - np.mean(source_full)
    h = _poisson_solve(source_full)
    return _fix_sum(h)


def _objective_sources(sources):
    """
    Objective function for source-based optimization.
    sources: vector of length K
    """
    h = _sources_to_h(sources)
    if not _is_valid(h):
        return 1e6
    return _score(h)


def construct():
    start_time = time.time()

    def deadline_check():
        return time.time() > start_time + DEADLINE

    best_h = np.full(N, 0.5)
    best_score = _score(best_h)

    rng = np.random.RandomState(42)

    # Strategy:
    # 1. Use low-dimensional sparse source representation (K << N)
    # 2. Optimize sources via SLSQP to minimize max correlation
    # 3. Polish with coordinate descent on full h

    # Try different numbers of sources
    K_values = [5, 8, 10, 15, 20]

    for K in K_values:
        if deadline_check():
            break

        # Multiple random initializations for K sources
        for init_idx in range(5):
            if deadline_check():
                break

            # Random source amplitudes
            sources0 = rng.uniform(-0.5, 0.5, K)

            try:
                res = minimize(
                    _objective_sources,
                    sources0,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
                )

                if res.success:
                    h_curr = _sources_to_h(res.x)
                    if _is_valid(h_curr):
                        s_curr = _score(h_curr)
                        if s_curr < best_score:
                            best_score = s_curr
                            best_h = h_curr.copy()

            except Exception:
                pass

        # Also try structured initializations
        # Alternating signs
        sources_alt = np.array([0.5 if i % 2 == 0 else -0.5 for i in range(K)])
        try:
            res = minimize(
                _objective_sources,
                sources_alt,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.success:
                h_curr = _sources_to_h(res.x)
                if _is_valid(h_curr):
                    s_curr = _score(h_curr)
                    if s_curr < best_score:
                        best_score = s_curr
                        best_h = h_curr.copy()
        except Exception:
            pass

    # Polish the best result with coordinate descent on full h
    if not deadline_check():
        h_cd = best_h.copy()
        current_score = _score(h_cd)
        
        # Coordinate descent: try small perturbations on each coordinate
        for iteration in range(50):
            if deadline_check():
                break
            
            improved = False
            for i in range(N):
                if deadline_check():
                    break
                
                # Try several perturbations
                for delta in [-0.1, -0.05, 0.05, 0.1]:
                    h_new = h_cd.copy()
                    h_new[i] += delta
                    h_new = _fix_sum(h_new)
                    
                    if _is_valid(h_new):
                        s_new = _score(h_new)
                        if s_new < current_score - 1e-10:
                            h_cd = h_new
                            current_score = s_new
                            improved = True
                            break
            
            if not improved:
                break
        
        if current_score < best_score:
            best_score = current_score
            best_h = h_cd.copy()

    # Final Simulated Annealing polish
    if not deadline_check():
        h_sa = best_h.copy()
        current_score = _score(h_sa)
        T = 0.005
        step_size = 0.03
        
        for _ in range(3000):
            if deadline_check():
                break
            
            num_changes = rng.randint(1, 4)
            indices = rng.choice(N, num_changes, replace=False)
            deltas = rng.uniform(-step_size, step_size, num_changes)
            
            h_new = h_sa.copy()
            for idx, delta in zip(indices, deltas):
                h_new[idx] += delta
            
            h_new = _fix_sum(h_new)
            
            if not _is_valid(h_new):
                continue
            
            s_new = _score(h_new)
            delta_s = s_new - current_score
            
            if delta_s < 0 or rng.random() < np.exp(-delta_s / T):
                h_sa = h_new
                current_score = s_new
                if current_score < best_score:
                    best_score = current_score
                    best_h = h_sa.copy()
            
            T *= 0.995
            if T < 1e-5:
                break

    # Final validation
    if not _is_valid(best_h):
        best_h = np.full(N, 0.5)

    return best_h
