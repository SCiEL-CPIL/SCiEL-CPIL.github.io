import time
import random
import numpy as np


def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 85  # safety margin before 90s kill

    # Precompute all arithmetic progressions as index arrays
    progressions = []
    for step in range(1, n):
        for start in range(step):
            max_len = (n - 1 - start) // step + 1
            for length in range(1, max_len + 1):
                indices = np.arange(start, start + length * step, step)
                progressions.append(indices)

    num_progs = len(progressions)

    # Build sparse representation: for each progression, store indices and signs pattern
    # We'll use a matrix A where A[p, i] = 1 if index i is in progression p
    # But this matrix is too large, so we'll use a list of index arrays

    # Precompute progression index arrays as numpy arrays for fast dot products
    prog_indices = [np.asarray(p, dtype=np.int64) for p in progressions]

    def compute_max_abs(signed_vec):
        """Compute max absolute sum over all progressions for a given sign vector."""
        max_abs = 0
        for pi in prog_indices:
            s = np.dot(signed_vec[pi], np.ones(len(pi)))
            if abs(s) > max_abs:
                max_abs = abs(s)
        return max_abs

    def compute_grad_softmax(signed_vec, beta):
        """Compute gradient of softmax-max objective."""
        # Compute all progression sums
        sums = np.empty(num_progs)
        for p, pi in enumerate(prog_indices):
            sums[p] = np.dot(signed_vec[pi], np.ones(len(pi)))
        
        abs_sums = np.abs(sums)
        # Softmax approximation of max
        max_val = np.max(abs_sums)
        weights = np.exp(beta * (abs_sums - max_val))
        weights /= np.sum(weights)
        
        # Gradient: d/ds_i sum_p w_p * |sum_p| = sum_p w_p * sign(sum_p) * [i in p]
        grad = np.zeros(n)
        for p, pi in enumerate(prog_indices):
            if abs_sums[p] > 0:
                sign_p = np.sign(sums[p])
                grad[pi] += weights[p] * sign_p
        return grad, np.max(abs_sums)

    # Initialize with Thue-Morse sequence (known low-discrepancy)
    def thue_morse(n):
        tm = np.zeros(n, dtype=np.int8)
        for i in range(n):
            # Count bits in i
            tm[i] = 1 if bin(i).count('1') % 2 == 0 else -1
        return tm

    best_signs = None
    best_max = float('inf')

    # Try multiple initializations
    initializations = []
    
    # Thue-Morse
    initializations.append(thue_morse(n))
    
    # Random with different seeds
    for seed in [42, 123, 456]:
        rng = np.random.RandomState(seed)
        initializations.append(rng.choice([-1, 1], size=n).astype(np.int8))

    for init_signs in initializations:
        if time.time() > deadline - 5:
            break

        # Continuous optimization: projected gradient descent
        # Work in continuous space, then round to ±1
        x = init_signs.astype(np.float64)
        
        # Learning rate schedule
        lr = 0.1
        beta = 5.0  # softmax sharpness
        
        current_max = compute_max_abs(x)
        
        # Gradient descent with projection to hypercube [-1, 1]
        steps = 0
        max_steps = 5000
        step_time_budget = 15.0  # seconds per initialization
        
        while steps < max_steps:
            if time.time() > deadline - 5:
                break
            if time.time() > t0 + step_time_budget * (initializations.index(init_signs) + 1) + 10:
                break
            
            # Compute gradient
            grad, current_max = compute_grad_softmax(x, beta)
            
            # Update
            x -= lr * grad
            x = np.clip(x, -1.0, 1.0)
            
            # Occasionally round and check
            if steps % 50 == 0:
                rounded = np.sign(x)
                rounded[rounded == 0] = 1  # handle zeros
                rounded_max = compute_max_abs(rounded)
                if rounded_max < best_max:
                    best_max = rounded_max
                    best_signs = rounded.copy()
            
            # Adapt learning rate
            lr *= 0.999
            beta = min(beta * 1.001, 50.0)
            steps += 1

        # Final round
        rounded = np.sign(x)
        rounded[rounded == 0] = 1
        rounded_max = compute_max_abs(rounded)
        if rounded_max < best_max:
            best_max = rounded_max
            best_signs = rounded.copy()

    # Local search refinement on best found
    if best_signs is not None and time.time() < deadline - 3:
        # Single flip local search
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            best_signs = best_signs.copy()
            current_max = compute_max_abs(best_signs)
            
            for i in range(n):
                if time.time() > deadline - 3:
                    break
                # Try flipping sign at position i
                old_sign = best_signs[i]
                best_signs[i] = -old_sign
                new_max = compute_max_abs(best_signs)
                if new_max < current_max:
                    current_max = new_max
                    improved = True
                else:
                    best_signs[i] = old_sign

        if best_max > current_max:
            best_max = current_max

    # Ensure we have a valid result
    if best_signs is None:
        rng = random.Random(42)
        best_signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)

    return [int(x) for x in best_signs]
