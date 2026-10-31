"""Search-based program for Erdos minimum overlap task."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _max_overlap(h):
    """Compute max over shifts of sum(h[i]*(1-h[(i+s)%N]))."""
    N = len(h)
    inv_h = 1.0 - h
    best = 0.0
    for s in range(N):
        ov = np.dot(h, inv_h[np.roll(np.arange(N), -s)])
        if ov > best:
            best = ov
    return best

def _score(h):
    s = h.sum()
    if abs(s - N/2.0) > 1e-8:
        return float('inf')
    if h.min() < -1e-8 or h.max() > 1.0 + 1e-8:
        return float('inf')
    return _max_overlap(h)

def _project_to_feasible(h):
    """Project h to satisfy sum=N/2 and 0<=h<=1."""
    target = N / 2.0
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if s < 1e-10:
        h = np.full(N, 0.5)
        return h
    h = h * (target / s)
    h = np.clip(h, 0.0, 1.0)
    # Fix residual by distributing evenly
    diff = target - h.sum()
    if abs(diff) > 1e-12:
        # Find indices with room to adjust
        if diff > 0:
            idxs = np.argsort(h)
        else:
            idxs = np.argsort(-h)
        for idx in idxs:
            room = (1.0 - h[idx]) if diff > 0 else h[idx]
            adjust = min(room, abs(diff))
            h[idx] += adjust * (1 if diff > 0 else -1)
            diff -= adjust * (1 if diff > 0 else -1)
            if abs(diff) < 1e-12:
                break
    return h

def _objective_and_grad(x):
    """Objective: t + penalties, where x = [h_0, ..., h_{N-1}, t]"""
    h = x[:N]
    t = x[N]
    
    # Compute all overlaps
    overlaps = np.zeros(N)
    for s in range(N):
        overlaps[s] = np.dot(h, 1.0 - h[np.roll(np.arange(N), -s)])
    
    # Objective: t + max(0, overlaps - t) penalty
    excess = np.maximum(0, overlaps - t)
    obj = t + np.sum(excess)
    
    # Gradient
    grad_h = np.zeros(N)
    grad_t = 1.0
    
    # d(excess)/dt = -count of s where overlaps[s] > t
    active = overlaps > t
    grad_t -= np.sum(active)
    
    # d(overlaps[s])/d(h[j]) = (1-h[(j+s)%N]) - h[(j-s)%N]
    for s in range(N):
        if active[s]:
            grad_h += (1.0 - h[(np.arange(N) + s) % N]) - h[(np.arange(N) - s) % N]
    
    return obj, np.concatenate([grad_h, [grad_t]])

def _constraints(x):
    """Constraints: sum(h) = N/2, 0<=h<=1, t>=0"""
    h = x[:N]
    t = x[N]
    
    cons = []
    # Sum constraint
    cons.append({'type': 'eq', 'fun': lambda x: np.sum(x[:N]) - N/2.0, 
                 'jac': lambda x: np.concatenate([np.ones(N), [0.0]])})
    # t >= 0
    cons.append({'type': 'ineq', 'fun': lambda x: x[N],
                 'jac': lambda x: np.concatenate([np.zeros(N), [1.0]])})
    # h[i] >= 0
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i],
                     'jac': lambda x, i=i: np.concatenate([[1.0 if j==i else 0.0 for j in range(N)], [0.0]])})
    # h[i] <= 1
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[i],
                     'jac': lambda x, i=i: np.concatenate([[-1.0 if j==i else 0.0 for j in range(N)], [0.0]])})
    
    return cons

def _generate_initial_guesses(rng):
    """Generate diverse initial guesses."""
    guesses = []
    
    # Uniform
    guesses.append(np.full(N, 0.5))
    
    # Sinusoidal patterns
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            x = np.arange(N)
            h = 0.5 + 0.4 * np.sin(2*np.pi*freq*x/N + phase)
            h = _project_to_feasible(h)
            guesses.append(h.copy())
    
    # Block patterns
    for block in [1, 2, 3, 4, 5, 10, 25]:
        h = np.zeros(N)
        for i in range(N):
            if (i // block) % 2 == 0:
                h[i] = 1.0
        h = _project_to_feasible(h)
        guesses.append(h.copy())
    
    # Random
    for _ in range(30):
        h = rng.uniform(0.0, 1.0, N)
        h = _project_to_feasible(h)
        guesses.append(h.copy())
    
    # Thue-Morse-like pattern
    tm = np.zeros(N)
    for i in range(N):
        # Count bits
        c = 0
        n = i
        while n:
            c += n & 1
            n >>= 1
        tm[i] = 0.0 if c % 2 == 0 else 1.0
    tm = _project_to_feasible(tm)
    guesses.append(tm.copy())
    
    # Pseudo-random with good properties
    for _ in range(10):
        h = (rng.random(N) > 0.5).astype(float)
        h = _project_to_feasible(h)
        guesses.append(h.copy())
    
    return guesses

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    rng = np.random.default_rng(42)
    
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # Generate initial guesses
    guesses = _generate_initial_guesses(rng)
    
    # First pass: quick SLSQP from each guess with limited iterations
    for guess in guesses:
        if time.time() > deadline - 30:
            break
        
        h0 = guess.copy()
        t0 = _max_overlap(h0) + 0.1
        x0 = np.concatenate([h0, [t0]])
        
        try:
            res = minimize(
                _objective_and_grad,
                x0,
                method='SLSQP',
                jac=True,
                bounds=[(0, 1)] * N + [(0, 2)],
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            h_opt = _project_to_feasible(res.x[:N])
            sc = _score(h_opt)
            if sc < best_score:
                best_score = sc
                best_h = h_opt.copy()
        except Exception:
            continue
    
    # Second pass: refine best with more iterations
    if time.time() < deadline - 20:
        h0 = best_h.copy()
        t0 = _max_overlap(h0) + 0.01
        x0 = np.concatenate([h0, [t0]])
        
        try:
            res = minimize(
                _objective_and_grad,
                x0,
                method='SLSQP',
                jac=True,
                bounds=[(0, 1)] * N + [(0, 2)],
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            h_opt = _project_to_feasible(res.x[:N])
            sc = _score(h_opt)
            if sc < best_score:
                best_score = sc
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Local search refinement
    h = best_h.copy()
    for _ in range(5000):
        if time.time() > deadline:
            break
        h_trial = h.copy()
        idxs = rng.choice(N, size=2, replace=False)
        for i in idxs:
            h_trial[i] = np.clip(h_trial[i] + rng.uniform(-0.02, 0.02), 0.0, 1.0)
        h_trial = _project_to_feasible(h_trial)
        sc = _score(h_trial)
        if sc < best_score - 1e-12:
            best_score = sc
            best_h = h_trial.copy()
            h = h_trial.copy()
    
    return best_h
# EVOLVE-BLOCK-END
