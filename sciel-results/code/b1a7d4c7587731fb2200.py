"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START
def _autocorr_fft(h):
    """Compute circular autocorrelation using FFT. Returns array of length N where
    result[k] = sum_i h[i] * h[(i-k) % N] for k=0..N-1."""
    n = len(h)
    # Use FFT to compute circular autocorrelation
    H = np.fft.fft(h)
    acf = np.fft.ifft(H * np.conj(H)).real
    # acf[k] = sum_i h[i] * h[(i+k) % N] for k=0..N-1
    # We want sum_i h[i] * h[(i-k) % N] which is acf[N-k] for k>0, acf[0] for k=0
    # Actually: sum_i h[i]*h[(i-k)%N] = sum_i h[i+k]*h[i] (by substitution) = acf[k]
    # Wait, let me be careful.
    # np.fft.ifft(H * conj(H))[k] = sum_i h[i] * h[(i-k) % N]
    # That's exactly what we want.
    return acf


def _max_overlap(h):
    """Compute max_k sum_i h[i] * (1 - h[(i-k) % N]) = max_k (N/2 - autocorr[k])."""
    acf = _autocorr_fft(h)
    # overlap[k] = sum_i h[i] * (1 - h[(i-k)%N]) = sum_i h[i] - sum_i h[i]*h[(i-k)%N]
    #           = N/2 - acf[k]
    # We want max over k of (N/2 - acf[k]) = N/2 - min(acf)
    return N / 2.0 - np.min(acf)


def _project_to_feasible(h):
    """Project h onto the feasible set: 0 <= h[i] <= 1, sum(h) = N/2."""
    h = h.copy()
    h = np.clip(h, 0, 1)
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-12:
        if diff > 0:
            # Need to add mass; distribute to elements not at upper bound
            room = 1.0 - h
            total_room = np.sum(room)
            if total_room > 1e-12:
                h += diff * room / total_room
            else:
                h += diff / N
        else:
            # Need to remove mass; distribute from elements not at lower bound
            mass = h
            total_mass = np.sum(mass)
            if total_mass > 1e-12:
                h += diff * mass / total_mass
            else:
                h += diff / N
    h = np.clip(h, 0, 1)
    # Final correction
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-12:
        h += diff / N
        h = np.clip(h, 0, 1)
    return h


def _objective(x):
    """Objective: minimize t (max overlap) with epigraph.
    x = [h_0, ..., h_{N-1}, t]
    Returns [t, constraint_violation] but for SLSQP we use constraints."""
    h = x[:N]
    t = x[N]
    acf = _autocorr_fft(h)
    overlaps = N / 2.0 - acf
    # We want t >= overlaps[k] for all k, and minimize t
    # For the objective, just return t
    return t


def _constraint_sum(x):
    """Constraint: sum(h) = N/2"""
    return np.sum(x[:N]) - N / 2.0


def _constraints(x):
    """Constraints: t - overlaps[k] >= 0 for all k, i.e., overlaps[k] - t <= 0"""
    h = x[:N]
    t = x[N]
    acf = _autocorr_fft(h)
    overlaps = N / 2.0 - acf
    # Constraint: overlaps[k] - t <= 0 for all k
    return overlaps - t


def _optimize_from_start(h0, max_time=30.0):
    """Run SLSQP optimization from a starting point h0."""
    start_time = time.time()
    h0 = _project_to_feasible(h0)
    x0 = np.concatenate([h0, [N / 2.0]])  # Initial t = N/2 (worst case)
    
    bounds = [(0, 1)] * N + [(0, N)]  # h[i] in [0,1], t in [0, N]
    
    # Equality constraint: sum(h) = N/2
    cons = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[:N]) - N / 2.0},
        # Inequality constraints: overlaps[k] - t <= 0 for all k
        # We'll use a single constraint function that returns an array
    ]
    
    # SLSQP can handle multiple inequality constraints via a single function returning array
    # But scipy's SLSQP expects each constraint to return scalar or we use 'ineq' type
    # Let's use the approach where we define all inequality constraints
    
    def ineq_constraints(x):
        """Returns array of values that must be >= 0 (scipy uses >= 0 for 'ineq')."""
        h = x[:N]
        t = x[N]
        acf = _autocorr_fft(h)
        overlaps = N / 2.0 - acf
        # We need t - overlaps[k] >= 0 for all k
        return t - overlaps
    
    cons = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[:N]) - N / 2.0},
        {'type': 'ineq', 'fun': ineq_constraints}
    ]
    
    result = minimize(
        _objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
    )
    
    elapsed = time.time() - start_time
    if result.success and time.time() - start_time < max_time:
        h_opt = _project_to_feasible(result.x[:N])
        score = _max_overlap(h_opt)
        return h_opt, score, elapsed
    else:
        # Return the best feasible point found
        h_opt = _project_to_feasible(result.x[:N])
        score = _max_overlap(h_opt)
        return h_opt, score, elapsed


def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5 second safety margin
    np.random.seed(42)
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Generate diverse starting points
    start_points = []
    
    # Uniform
    start_points.append(np.full(N, 0.5))
    
    # Sinusoidal candidates
    for freq in range(1, 15):
        for phase in [0.0, np.pi/2, np.pi, np.pi/4, 3*np.pi/4]:
            x = np.linspace(0, 2*np.pi*freq, N, endpoint=False)
            h = 0.5 + 0.5*np.sin(x + phase)
            h = _project_to_feasible(h)
            start_points.append(h)
    
    # Random candidates
    for _ in range(30):
        h = np.random.uniform(0, 1, N)
        h = _project_to_feasible(h)
        start_points.append(h)
    
    # Binary-like candidates
    for _ in range(20):
        h = np.zeros(N)
        mask = np.random.random(N) < 0.5
        h[mask] = 1.0
        h = _project_to_feasible(h)
        start_points.append(h)
    
    # Evaluate starting points and pick the best few to optimize
    scored_starts = []
    for h in start_points:
        if time.time() > deadline:
            break
        score = _max_overlap(h)
        scored_starts.append((score, h))
    
    scored_starts.sort(key=lambda x: x[0])
    
    # Try optimization from the top starting points
    for score, h0 in scored_starts[:20]:
        if time.time() > deadline - 10:
            break
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        h_opt, opt_score, elapsed = _optimize_from_start(h0, max_time=min(remaining - 1.0, 30.0))
        if opt_score < best_score:
            best_score = opt_score
            best_h = h_opt.copy()
    
    # Final check
    if time.time() < deadline:
        final_score = _max_overlap(best_h)
        if final_score < best_score:
            best_score = final_score
    
    return best_h
# EVOLVE-BLOCK-END
