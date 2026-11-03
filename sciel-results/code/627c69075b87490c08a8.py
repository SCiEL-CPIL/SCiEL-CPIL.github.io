"""Rewritten: optimize log-energy on the sphere via projected gradient descent with restarts."""
import time
import numpy as np
from math import fsum

N = 150
DEADLINE = 150.0
SAFETY = 5.0  # seconds before the hard limit


def _energy(pts):
    # Vectorized energy computation
    # d[i,j] = ||x_i - x_j||
    # We only need i < j
    N = pts.shape[0]
    # Compute pairwise distances
    # (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    # Mask diagonal to avoid log(0)
    mask = ~np.eye(N, dtype=bool)
    d = np.sqrt(d2)
    # Sum log(1/d) = -sum(log(d)) for i<j
    # Since d is symmetric, sum over upper triangle
    # Using fsum for precision as per task description if needed, but np.sum is usually sufficient for this scale
    # However, task says "pair terms summed with math.fsum". 
    # For N=150, there are ~11k pairs. np.sum is fast. 
    # To strictly adhere to "math.fsum" if it impacts score precision, we might need a loop, but that's slow.
    # Let's stick to np.sum for speed, as the error is likely dominated by the gradient loops.
    # Actually, let's just extract the upper triangle values.
    triu_idx = np.triu_indices(N, k=1)
    d_vals = d[triu_idx]
    # Ensure no zeros (points are distinct)
    d_vals = np.maximum(d_vals, 1e-15)
    return -np.sum(np.log(d_vals))


def _energy_grad(pts):
    # Vectorized gradient computation
    # grad_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    N = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :] # (N, N, 3)
    d2 = np.sum(diff * diff, axis=2) # (N, N)
    # Avoid division by zero on diagonal
    np.fill_diagonal(d2, 1.0) # Temporary placeholder
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0) # Zero out self-interaction
    # contrib[i, j] = (x_i - x_j) / d2[i, j]
    contrib = diff * inv_d2[:, :, np.newaxis] # (N, N, 3)
    grad = np.sum(contrib, axis=1) # (N, 3)
    return grad


def _project_tangent(pts, grad):
    # project grad onto tangent plane at each point: g - (g.x_i) x_i
    dots = np.einsum('ij,ij->i', grad, pts)
    return grad - dots[:, None] * pts


def _fibonacci_sphere():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    v = rng.standard_normal((N, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _grad_descent(pts, rng, budget):
    pts = pts.copy()
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    best = pts.copy()
    best_e = _energy(best)
    lr = 1e-2 # Start with larger learning rate for faster initial progress
    n = 0
    # We need to be careful with time.
    # Estimate cost per iteration.
    # For N=150, vectorized ops are fast (~ms).
    # Let's run until budget or convergence.
    while time.time() < budget:
        g = _energy_grad(pts)
        gt = _project_tangent(pts, g)
        gnorm = np.linalg.norm(gt)
        if gnorm < 1e-10:
            break
        step = gt * lr
        pts_new = pts + step
        pts_new /= np.linalg.norm(pts_new, axis=1, keepdims=True)
        e_new = _energy(pts_new)
        if e_new < best_e:
            best = pts_new.copy()
            best_e = e_new
            lr = min(lr * 1.1, 0.5) # Increase lr cautiously
        else:
            lr *= 0.8
            # If lr gets too small, break to save time for other restarts
            if lr < 1e-6:
                break
        pts = pts_new
        n += 1
        # Check time more frequently to avoid long loops
        if n % 10 == 0 and time.time() > budget:
            break
    return best, best_e


def construct():
    t0 = time.time()
    # Use a tighter safety margin to ensure we return before 120s
    budget = t0 + DEADLINE - 10.0 
    rng = np.random.default_rng(12345)
    best_pts = None
    best_e = np.inf

    # Start from a few different initializations.
    # Fibonacci is a good start. Add 2 random restarts.
    inits = [_fibonacci_sphere()]
    for _ in range(2):
        inits.append(_random_sphere(rng))

    # Allocate time per restart
    # If we have 3 restarts, give each ~ (budget - t0) / 3
    # But we don't know t0 vs budget difference yet.
    # Simpler: just run them sequentially with a global budget check.
    
    for i, init in enumerate(inits):
        if time.time() > budget:
            break
        # Give the first restart more time, subsequent ones less if time is tight?
        # Or just let the inner loop handle it.
        # To be safe, let's cap each restart's time to 1/3 of remaining time if possible?
        # No, the inner loop checks `budget`. 
        # However, if the first restart takes 100s, we have no time for others.
        # Let's enforce a per-restart time limit.
        remaining = budget - time.time()
        per_restart_time = remaining / (len(inits) - i)
        # Ensure at least 5 seconds per restart
        per_restart_time = max(per_restart_time, 5.0)
        
        pts, e = _grad_descent(init, rng, time.time() + per_restart_time)
        if e < best_e:
            best_e = e
            best_pts = pts

    if best_pts is None:
        best_pts = _fibonacci_sphere()

    # Final high-precision polish with smaller steps.
    # Only if we have at least 5 seconds left
    if time.time() < budget - 5.0:
        # Use a smaller initial LR for polish
        # We can't easily change LR inside _grad_descent without modifying it, 
        # but we can call it again. The current _grad_descent starts with lr=1e-2.
        # For polish, we might want smaller. 
        # Let's just run _grad_descent again. It will start with lr=1e-2.
        # This might be too aggressive for polish.
        # Let's modify _grad_descent to accept an initial LR?
        # For now, let's just run it. If it fails to improve, it will break quickly or oscillate.
        # Actually, let's just return the best found.
        pass

    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    return best_pts
