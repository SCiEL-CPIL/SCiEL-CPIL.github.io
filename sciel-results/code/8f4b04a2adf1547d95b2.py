"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Initialization: Rank-matching heuristic
    # Rows of A with largest sums go to rows of B with smallest sums
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Calculate initial cost
    # Cost = sum a[i,j] * b[p[i], p[j]]
    # This is equivalent to sum a[i,j] * b[p[i], p[j]]
    # We can compute this efficiently using matrix multiplication if we construct the permutation matrix,
    # but for n=150, a double loop or vectorized sum is fine.
    # Let's use a vectorized approach:
    # B_perm[i, j] = b[p[i], p[j]]
    # Cost = sum(a * B_perm)
    
    def compute_cost(p):
        # b[p[:, None], p[None, :]] creates the permuted matrix
        return np.sum(a * b[p[:, None], p[None, :]])
        
    current_cost = compute_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Simulated Annealing
    # Parameters
    T = 1000.0  # Initial temperature, needs tuning relative to cost magnitude
    # Cost magnitude is ~8e6. Delta C is usually small.
    # Let's estimate typical delta.
    # If we swap, delta is sum of ~150 terms.
    # Let's start T high enough to accept bad moves.
    
    # A better way to set T:
    # Sample a few random swaps to estimate average |delta|
    deltas = []
    for _ in range(100):
        i, j = random.sample(range(n), 2)
        u, v = p[i], p[j]
        delta = np.dot(a[i] - a[j], b[u, p] - b[v, p])
        deltas.append(abs(delta))
    avg_delta = np.mean(deltas) if deltas else 1.0
    T = avg_delta * 10.0
    T_min = 1e-5
    alpha = 0.999  # Cooling rate
    
    iterations = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        i, j = random.sample(range(n), 2)
        u, v = p[i], p[j]
        
        # Calculate delta cost
        # delta = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[u, p[k]] - b[v, p[k]])
        # As derived, dot(a[i]-a[j], b[u,p]-b[v,p]) works for all k including i,j
        delta = np.dot(a[i] - a[j], b[u, p] - b[v, p])
        
        if delta < 0:
            # Accept
            p[i], p[j] = v, u
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Probabilistic acceptance
            if T > 0:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    p[i], p[j] = v, u
                    current_cost += delta
                    no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        # Cool down
        T *= alpha
        
        # Restart if stuck for too long
        if no_improve_count > 1000:
            # Try to restart with a perturbation or reheat
            # Reheat
            T = avg_delta * 5.0
            no_improve_count = 0
            # Maybe perturb the solution
            # p = best_p.copy()
            # current_cost = best_cost
            # Do a few random swaps
            for _ in range(10):
                i2, j2 = random.sample(range(n), 2)
                u2, v2 = p[i2], p[j2]
                delta2 = np.dot(a[i2] - a[j2], b[u2, p] - b[v2, p])
                p[i2], p[j2] = v2, u2
                current_cost += delta2
            # If this perturbation made it worse, maybe revert?
            # For now, just keep going.
            
        iterations += 1
        
        # Check time more frequently in the inner loop? 
        # The while loop condition is enough, but np.dot is fast.
        # 150 elements dot product is very fast.
        # 115 seconds / (time per iteration)
        # Time per iteration: ~100ns for dot, ~50ns for overhead. ~150ns total?
        # 115s / 150ns = 7.6e8 iterations. That's a lot.
        # Python overhead is higher. Maybe 1-2 microseconds per iteration.
        # 115s / 2e-6 = 57.5 million iterations.
        # This should be enough to find a good solution.

    return best_p.tolist()

# EVOLVE-BLOCK-END
