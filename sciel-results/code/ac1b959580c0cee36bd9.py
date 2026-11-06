"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_ap_matrix(n=256):
    """Build matrix A where each row represents one arithmetic progression.
    Row i has 1s at positions corresponding to the AP indices (0-indexed)."""
    rows = []
    for start in range(n):
        for step in range(1, n - start):
            # AP: start, start+step, start+2*step, ... all < n
            idx = list(range(start, n, step))
            if len(idx) < 2:
                continue
            row = np.zeros(n)
            row[idx] = 1.0
            rows.append(row)
    # Also include singletons (each single element is an AP of length 1)
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        rows.append(row)
    
    A = np.array(rows)
    return A

def _compute_max_ap_sum(signs, A):
    """Compute max absolute AP sum for given sign vector."""
    sums = A @ signs
    return np.max(np.abs(sums))

def construct():
    t0 = time.time()
    deadline = t0 + 80  # 80s safety margin for 90s kill
    
    n = 256
    rng = np.random.RandomState(42)
    
    # Build the AP matrix (this is the expensive part, done once)
    A = _build_ap_matrix(n)
    num_aps = A.shape[0]
    
    # Initialize with random signs
    signs = rng.choice([-1.0, 1.0], size=n)
    best_signs = signs.copy()
    best_score = _compute_max_ap_sum(signs, A)
    
    # Gradient-based optimization using soft-max surrogate
    # Loss: sum of (max(0, |s| - threshold))^2 for all AP sums
    # This is differentiable and encourages all AP sums to be small
    
    def soft_loss(x, A, threshold=1.0, power=2):
        """Differentiable surrogate: sum of (|A@x| - threshold)^2 for |A@x| > threshold"""
        sums = A @ x
        abs_sums = np.abs(sums)
        # Only penalize sums exceeding threshold
        excess = np.maximum(abs_sums - threshold, 0.0)
        return np.sum(excess ** power)
    
    def soft_grad(x, A, threshold=1.0, power=2):
        """Gradient of soft_loss"""
        sums = A @ x
        abs_sums = np.abs(sums)
        excess = np.maximum(abs_sums - threshold, 0.0)
        # d/dx sum((|A@x|-t)^2) for |A@x|>t
        # = 2 * (|A@x|-t) * sign(A@x) * A^T
        grad = 2.0 * power * excess * np.sign(sums)
        return A.T @ grad
    
    # SGD with momentum
    lr = 0.1
    momentum = 0.9
    velocity = np.zeros(n)
    x = signs.copy()
    
    # Learning rate schedule
    num_steps = 5000
    for step in range(num_steps):
        if time.time() > deadline - 5:
            break
        
        # Compute gradient
        g = soft_grad(x, A, threshold=1.0, power=2)
        
        # Momentum update
        velocity = momentum * velocity - lr * g
        x_new = x + velocity
        
        # Project back to [-1, 1] box (or keep continuous for now)
        # For gradient-based, we can keep continuous and discretize at end
        # But let's also try hard projection every few steps
        if step % 10 == 0:
            x_new = np.clip(x_new, -1.0, 1.0)
        
        x = x_new
        
        # Evaluate and track best
        # Discretize to check actual performance
        signs_disc = np.sign(x)
        signs_disc[signs_disc == 0] = 1.0
        score = _compute_max_ap_sum(signs_disc, A)
        
        if score < best_score:
            best_score = score
            best_signs = signs_disc.copy()
    
    # Final local search: try flipping individual signs
    final_signs = best_signs.copy()
    final_score = best_score
    
    # Greedy local search
    for i in range(n):
        if time.time() > deadline - 2:
            break
        # Try flipping sign i
        final_signs[i] *= -1
        new_score = _compute_max_ap_sum(final_signs, A)
        if new_score < final_score:
            final_score = new_score
        else:
            final_signs[i] *= -1  # revert
    
    # Convert to Python integers
    result = [int(s) for s in final_signs]
    return result
# EVOLVE-BLOCK-END
