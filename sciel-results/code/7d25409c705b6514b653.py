"""Rewritten program: Subgradient Optimization for Erdos minimum overlap (Linear Shifts)."""
import numpy as np
import time

N = 100
DEADLINE = 110  # seconds, safety margin before 120s kill


def _max_overlap_linear(h):
    """
    Compute the maximum overlap of h with shifted (1-h) over all linear shifts.
    C_k = sum_{i=0}^{N-k-1} h_i * (1 - h_{i+k})
    """
    h = np.asarray(h, dtype=np.float64)
    max_val = 0.0
    # We can compute this efficiently or just loop since N is small (100)
    # N=100, shifts=100, ops=10000. Very fast.
    for k in range(N):
        if k == 0:
            # C_0 = sum h_i * (1-h_i)
            c = np.sum(h * (1.0 - h))
        else:
            # C_k = sum h_i * (1-h_{i+k}) for i=0..N-k-1
            part1 = h[:N-k]
            part2 = 1.0 - h[k:]
            c = np.sum(part1 * part2)
        if c > max_val:
            max_val = c
    return max_val


def _subgrad_linear(h):
    """
    Compute the subgradient of the max overlap.
    Find the k that achieves the max, then compute the gradient of C_k.
    """
    h = np.asarray(h, dtype=np.float64)
    max_val = -1.0
    active_k = 0
    
    # Find active k
    for k in range(N):
        if k == 0:
            c = np.sum(h * (1.0 - h))
        else:
            part1 = h[:N-k]
            part2 = 1.0 - h[k:]
            c = np.sum(part1 * part2)
        if c > max_val:
            max_val = c
            active_k = k
            
    # Compute gradient of C_{active_k}
    g = np.zeros(N)
    k = active_k
    
    if k == 0:
        # C_0 = sum h_i - h_i^2
        # dC/dh_i = 1 - 2*h_i
        g = 1.0 - 2.0 * h
    else:
        # C_k = sum_{i=0}^{N-k-1} (h_i - h_i h_{i+k})
        # Term 1: sum h_i -> gradient is 1 for i in [0, N-k-1]
        g[:N-k] += 1.0
        
        # Term 2: - sum h_i h_{i+k}
        # For a fixed j, h_j appears in h_i if i=j, and in h_{i+k} if i+k=j -> i=j-k.
        # If i=j: term is -h_{j+k}. Valid if j < N-k and j+k < N.
        # If i=j-k: term is -h_{j-k}. Valid if j-k >= 0 and j-k < N-k -> j < N.
        
        # Contribution from h_i (where i=j):
        # -h_{j+k} for j in [0, N-k-1]
        if N - k > 0:
            g[:N-k] -= h[k:]
            
        # Contribution from h_{i+k} (where i+k=j):
        # -h_{j-k} for j in [k, N-1]
        g[k:] -= h[:N-k]
        
    return g, max_val


def _project(h):
    """
    Project h onto the feasible set: 0 <= h_i <= 1 and sum(h) = N/2.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    
    # Iterative projection
    for _ in range(100):
        # Clip to [0, 1]
        h = np.clip(h, 0.0, 1.0)
        
        # Adjust sum
        current_sum = np.sum(h)
        target_sum = N / 2.0
        diff = target_sum - current_sum
        
        if abs(diff) < 1e-10:
            break
            
        # Distribute difference
        # To maintain bounds, we can add diff/N to all, but that might violate bounds.
        # Better: add to those with room, subtract from those with room.
        if diff > 0:
            # Add to elements < 1
            room = 1.0 - h
            total_room = np.sum(room)
            if total_room < 1e-10:
                break
            h += (diff / total_room) * room
        else:
            # Subtract from elements > 0
            room = h
            total_room = np.sum(room)
            if total_room < 1e-10:
                break
            h += (diff / total_room) * room
            
    # Final clip and exact sum fix
    h = np.clip(h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-12:
        # Simple distribution
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        # One more correction
        diff2 = N / 2.0 - np.sum(h)
        h += diff2 / N
        h = np.clip(h, 0.0, 1.0)

    return h


def construct():
    t0 = time.time()
    
    # Initialize with a non-uniform pattern
    # Try a few inits
    best_h = None
    best_val = float('inf')
    
    inits = [
        np.full(N, 0.5),
        np.sin(np.arange(N) * 2 * np.pi / 10) * 0.4 + 0.5,
        np.random.uniform(0, 1, N)
    ]
    
    for h0 in inits:
        h0 = _project(h0)
        val = _max_overlap_linear(h0)
        if val < best_val:
            best_val = val
            best_h = h0.copy()
            
        if time.time() - t0 > DEADLINE - 10:
            break
            
    if best_h is None:
        best_h = np.full(N, 0.5)
        best_val = _max_overlap_linear(best_h)

    # Subgradient Descent
    h = best_h.copy()
    current_val = best_val
    step_size = 0.1
    min_step = 1e-6
    
    # Annealing of step size
    decay = 0.99
    
    while time.time() - t0 < DEADLINE - 2:
        g, val = _subgrad_linear(h)
        
        # Update
        h_new = h - step_size * g
        h_new = _project(h_new)
        
        new_val = _max_overlap_linear(h_new)
        
        # Accept if better, or with probability if worse (simple restart if stuck)
        if new_val < current_val:
            h = h_new
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_h = h.copy()
        else:
            # Reduce step size if stuck
            step_size *= decay
            
        if step_size < min_step:
            break

    # Final projection
    final_h = _project(best_h)
    
    return final_h
