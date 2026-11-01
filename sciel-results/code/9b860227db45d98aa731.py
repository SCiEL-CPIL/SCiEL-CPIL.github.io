"""Rewritten program: minimize (max/min pairwise distance)^2 for 14 points in 3D."""
import time
import numpy as np
from scipy.optimize import minimize

N = 14
D = 3

def _pairwise_dists_sq(P):
    # P: (N, D)
    diff = P[:, None, :] - P[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    return d2

def _score(P):
    d2 = _pairwise_dists_sq(P)
    iu = np.triu_indices(N, k=1)
    vals = d2[iu]
    mn = vals.min()
    mx = vals.max()
    if mn <= 0:
        return np.inf
    return (mx / mn) ** 2

def _objective_flat(x):
    P = x.reshape(N, D)
    return _score(P)

def _try_optimize(x0, time_budget):
    """Run L-BFGS-B from x0 with a time budget. Returns (P, score)."""
    t_start = time.time()
    x0 = np.asarray(x0, dtype=float).flatten()
    
    # Ensure distinctness
    P0 = x0.reshape(N, D)
    # Check for duplicates
    d2 = _pairwise_dists_sq(P0)
    iu = np.triu_indices(N, k=1)
    if d2[iu].min() < 1e-12:
        # Perturb
        x0 = x0 + 0.01 * np.random.randn(*x0.shape)
    
    best_x = x0.copy()
    best_score = _objective_flat(best_x)
    
    # Use L-BFGS-B without gradient (numerical) for simplicity, but that's slow.
    # Let's use BFGS or L-BFGS-B with numerical gradient.
    # Actually, for 42 variables, numerical gradient is fine.
    
    def obj(x):
        return _objective_flat(x)
    
    # We need a callback to check time and update best
    state = {'done': False}
    
    def callback(xk):
        if time.time() - t_start > time_budget:
            state['done'] = True
        # Update best if better
        sc = _objective_flat(xk)
        if sc < best_score:
            nonlocal best_score, best_x
            best_score = sc
            best_x = xk.copy()
        return state['done']
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}, callback=callback)
    except Exception:
        pass
    
    # Final check
    sc = _objective_flat(best_x)
    if sc < best_score:
        best_score = sc
    
    return best_x.reshape(N, D), best_score

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin from 150s limit
    
    # Generate diverse initial guesses
    rng = np.random.RandomState(42)
    
    initial_guesses = []
    
    # 1. Random uniform in unit cube
    for _ in range(10):
        initial_guesses.append(rng.rand(N, D) * 2 - 1)
    
    # 2. Spherical distribution (Fibonacci-like)
    for _ in range(5):
        y = np.linspace(-1, 1, N)
        r = np.sqrt(1 - y**2)
        theta = np.linspace(0, 2*np.pi, N)
        P = np.column_stack([r*np.cos(theta), r*np.sin(theta), y])
        P = P * (0.5 + 0.5*rng.rand())  # vary radius slightly
        initial_guesses.append(P)
    
    # 3. Icosahedron-like + jitter
    # 12 icosahedron vertices + 2 center-ish
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, 0, phi], [1, 0, phi], [-1, 0, -phi], [1, 0, -phi],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [0, phi, -1], [0, phi, 1], [0, -phi, -1], [0, -phi, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    # Add 2 more points
    extra = rng.randn(2, D)
    extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
    P_ico = np.vstack([ico, extra])
    for _ in range(3):
        P_jit = P_ico + 0.1 * rng.randn(N, D)
        initial_guesses.append(P_jit)
    
    # 4. Ring configurations
    for _ in range(3):
        # Two rings
        n1 = N // 2
        n2 = N - n1
        th1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
        th2 = np.linspace(0, 2*np.pi, n2, endpoint=False)
        r1, r2 = 1.0, 0.8
        z1, z2 = 0.5, -0.5
        P1 = np.column_stack([r1*np.cos(th1), r1*np.sin(th1), np.full(n1, z1)])
        P2 = np.column_stack([r2*np.cos(th2), r2*np.sin(th2), np.full(n2, z2)])
        P = np.vstack([P1, P2])
        P = P + 0.05 * rng.randn(N, D)
        initial_guesses.append(P)
    
    # 5. Grid-based with perturbation
    for _ in range(3):
        # 2x2x4 grid
        grid = np.array(list(zip(
            [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1][:N],
            [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1][:N],
            [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1][:N]
        )), dtype=float)
        grid = grid - 0.5
        P = grid + 0.05 * rng.randn(N, D)
        initial_guesses.append(P)
    
    best_P = None
    best_score = np.inf
    
    # Allocate time per initial guess
    n_guesses = len(initial_guesses)
    time_per_guess = (deadline - t_start) / max(1, n_guesses)
    
    for i, P0 in enumerate(initial_guesses):
        if time.time() > deadline - 5.0:
            break
        
        time_budget = min(time_per_guess, deadline - time.time() - 5.0)
        if time_budget < 1.0:
            break
        
        P_opt, score = _try_optimize(P0, time_budget)
        
        if score < best_score:
            best_score = score
            best_P = P_opt.copy()
    
    # Final polish on the best solution found
    if best_P is not None and time.time() < deadline - 10.0:
        time_budget = deadline - time.time() - 5.0
        if time_budget > 5.0:
            P_polished, score_polished = _try_optimize(best_P, time_budget)
            if score_polished < best_score:
                best_P = P_polished
                best_score = score_polished
    
    if best_P is None:
        # Fallback
        best_P = np.random.rand(N, D)
    
    # Verify distinctness
    d2 = _pairwise_dists_sq(best_P)
    iu = np.triu_indices(N, k=1)
    if d2[iu].min() < 1e-10:
        # Perturb
        best_P = best_P + 0.01 * np.random.randn(N, D)
    
    return best_P
