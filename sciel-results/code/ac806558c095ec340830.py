"""Rewritten program for Erdos minimum overlap task."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START

def _compute_overlap(h, shift):
    """Compute the overlap sum: sum_i h[i] * (1 - h[(i+shift) % N])"""
    shifted = np.roll(h, shift)
    return np.sum(h * (1.0 - shifted))

def _max_overlap(h):
    """Compute the maximum overlap over all non-zero shifts."""
    max_ov = 0.0
    for s in range(1, N):
        ov = _compute_overlap(h, s)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _project_to_simplex_constraint(h, target_sum=N/2.0):
    """Project h onto [0,1]^N with sum exactly target_sum.
    Uses a simple iterative projection."""
    h = np.clip(h, 0.0, 1.0)
    # Adjust to hit the sum constraint
    diff = target_sum - np.sum(h)
    if abs(diff) < 1e-12:
        return h
    # Distribute the difference proportionally to available capacity
    if diff > 0:
        # Need to increase; add to elements with room
        room = 1.0 - h
        mask = room > 1e-10
        if not np.any(mask):
            return h
        total_room = np.sum(room[mask])
        if total_room < 1e-10:
            return h
        h[mask] += diff * room[mask] / total_room
    else:
        # Need to decrease; take from elements above 0
        room = h
        mask = room > 1e-10
        if not np.any(mask):
            return h
        total_room = np.sum(room[mask])
        if total_room < 1e-10:
            return h
        h[mask] += diff * room[mask] / total_room
    return np.clip(h, 0.0, 1.0)

def _grad_max_overlap(h, alpha=50.0):
    """Approximate gradient of the max overlap using soft-max over shifts.
    Uses log-sum-exp style weighting to get a smooth surrogate."""
    overlaps = np.zeros(N - 1)
    for s in range(1, N):
        overlaps[s-1] = _compute_overlap(h, s)
    
    # Soft-max weights
    max_ov = np.max(overlaps)
    exps = np.exp(alpha * (overlaps - max_ov))
    weights = exps / np.sum(exps)
    
    # Gradient of each overlap w.r.t. h
    grad = np.zeros(N)
    for idx, s in enumerate(range(1, N)):
        w = weights[idx]
        if w < 1e-10:
            continue
        # d/dh[i] of sum_j h[j]*(1-h[(j+s)%N])
        # = sum_j [delta_{i,j}*(1-h[(j+s)%N]) + h[j]*(-delta_{i,(j+s)%N})]
        # = (1-h[(i+s)%N]) - sum_{j: (j+s)%N == i} h[j]
        # = (1-h[(i+s)%N]) - h[(i-s)%N]
        shifted_plus = np.roll(h, s)
        shifted_minus = np.roll(h, -s)
        grad += w * (1.0 - shifted_plus - shifted_minus)
    
    return grad, max_ov

def _gradient_descent(h_init, max_iters=5000, lr=0.01, tol=1e-8, deadline=None):
    """Projected gradient descent on the soft-max surrogate of max overlap."""
    h = h_init.copy()
    h = _project_to_simplex_constraint(h)
    
    best_h = h.copy()
    best_val = _max_overlap(h)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        grad, soft_max = _grad_max_overlap(h)
        
        # Gradient step
        h_new = h - lr * grad
        h_new = _project_to_simplex_constraint(h_new)
        
        # Check improvement
        val = _max_overlap(h_new)
        if val < best_val - 1e-10:
            best_h = h_new.copy()
            best_val = val
        
        h = h_new
        
        # Convergence check
        if it > 100 and abs(grad).max() < tol:
            break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_h = None
    best_val = float('inf')
    
    # Generate diverse initializations
    inits = []
    
    # 1. Constant 0.5
    inits.append(np.full(N, 0.5))
    
    # 2. Alternating pattern
    alt = np.zeros(N)
    alt[0::2] = 0.8
    alt[1::2] = 0.2
    inits.append(alt)
    
    # 3. Cosine pattern
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    cos_pat = 0.5 + 0.4 * np.cos(t)
    inits.append(cos_pat)
    
    # 4. Sine pattern
    sin_pat = 0.5 + 0.4 * np.sin(t)
    inits.append(sin_pat)
    
    # 5. Block pattern (high-low-high-low in blocks)
    block = np.zeros(N)
    for i in range(N):
        block[i] = 0.9 if (i // 5) % 2 == 0 else 0.1
    inits.append(block)
    
    # 6. Random with mean 0.5
    np.random.seed(42)
    for seed in range(5):
        np.random.seed(seed)
        r = np.random.rand(N)
        r = 0.5 + 0.3 * (r - 0.5) * 2  # scale to [0.2, 0.8]
        inits.append(r)
    
    # 7. Two-basis convex combination approach
    # Basis 1: balanced (all 0.5)
    # Basis 2: high-contrast alternating
    basis1 = np.full(N, 0.5)
    basis2 = np.zeros(N)
    basis2[0::2] = 1.0
    basis2[1::2] = 0.0
    # Mix: h = w*basis1 + (1-w)*basis2, project to [0,1] and sum=N/2
    for w in [0.1, 0.3, 0.5, 0.7, 0.9]:
        mix = w * basis1 + (1.0 - w) * basis2
        inits.append(mix)
    
    # Run gradient descent on each initialization
    for idx, h_init in enumerate(inits):
        if time.time() > deadline:
            break
        
        h_init = _project_to_simplex_constraint(h_init)
        
        h_result, val = _gradient_descent(
            h_init, 
            max_iters=3000, 
            lr=0.005, 
            tol=1e-9,
            deadline=deadline
        )
        
        if val < best_val:
            best_val = val
            best_h = h_result
    
    # Final polish: run a few more iterations with smaller learning rate on the best
    if best_h is not None and time.time() < deadline - 10:
        h_polished, val_polished = _gradient_descent(
            best_h,
            max_iters=5000,
            lr=0.001,
            tol=1e-10,
            deadline=deadline
        )
        if val_polished < best_val:
            best_h = h_polished
            best_val = val_polished
    
    # Ensure exact sum
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _project_to_simplex_constraint(best_h)
    
    # Verify sum is exactly N/2
    current_sum = np.sum(best_h)
    if abs(current_sum - N/2.0) > 1e-10:
        diff = N/2.0 - current_sum
        # Distribute evenly
        best_h += diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
        # Re-project
        best_h = _project_to_simplex_constraint(best_h)
    
    return best_h

# EVOLVE-BLOCK-END
