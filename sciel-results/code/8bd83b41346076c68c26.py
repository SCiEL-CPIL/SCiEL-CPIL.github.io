"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 148.0  # Leave 2s safety margin for 150s limit

    # Ensure arrays are int64 for exact integer arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)

    # Precompute row sums for initial heuristic
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)

    # 1. Initial Solution: Rank Matching
    # Heavy flows (high row sums in a) should go to central locations (low row sums in b)
    heavy_idx = np.argsort(-a_sums, kind="stable")
    central_idx = np.argsort(b_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_idx] = central_idx

    # Calculate initial cost
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Vectorized: sum over i of (a[i] @ b[p[i], :])
    # b[p] gives a matrix where row i is b[p[i]]
    # So (a * b[p]).sum() computes the cost
    cost = np.sum(a * b[p])
    
    best_cost = cost
    best_p = p.copy()

    # 2. Local Search: 2-Opt (Pairwise Swap)
    # We iterate until no improvement is found in a full pass or time runs out.
    # We use a randomized restart strategy to escape local optima if time permits.
    
    # Precompute b_rows for fast access? b[p] is already a view/copy.
    # Let's use a helper to compute delta efficiently.
    # delta = sum_k (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]]) + 2*a[i,j]*b[u,v]
    # where u = p[i], v = p[j]
    
    # To speed up, we can precompute the current "interaction" matrix?
    # No, n=150 is small. O(n^2) per step is fine, but we do many steps.
    # Let's try to optimize the inner loop.
    
    def get_delta(i, j, p, a, b):
        u = p[i]
        v = p[j]
        # diff_a = a[i] - a[j]
        # diff_b = b[v, p] - b[u, p]
        # dot = np.dot(diff_a, diff_b)
        # term = 2 * a[i,j] * b[u,v]
        
        # Optimization: 
        # sum_k a[i,k]b[v,p[k]] - a[i,k]b[u,p[k]] - a[j,k]b[v,p[k]] + a[j,k]b[u,p[k]]
        # = a[i] @ b[v,p] - a[i] @ b[u,p] - a[j] @ b[v,p] + a[j] @ b[u,p]
        # = (a[i] - a[j]) @ (b[v,p] - b[u,p])
        
        # Let's just compute it directly with numpy vectors
        # b[p] is the matrix B_p where B_p[k] = b[p[k]]
        # We need b[v, p] and b[u, p] which are rows of b indexed by p.
        # This is b[v, p] and b[u, p].
        
        # Actually, computing b[v, p] involves gathering.
        # Let's stick to the dot product of differences.
        
        diff_a = a[i] - a[j]
        # b[v, p] is a vector of length n
        # b[u, p] is a vector of length n
        # diff_b = b[v, p] - b[u, p]
        
        # Using advanced indexing might be slow in a tight loop.
        # Let's try to precompute nothing but use efficient numpy ops.
        
        # Alternative:
        # delta = 2 * a[i,j] * b[u,v] + sum_k (a[i,k] - a[j,k]) * (b[v,p[k]] - b[u,p[k]])
        
        # Let's compute:
        # part1 = 2 * a[i,j] * b[u,v]
        # part2 = (a[i] - a[j]) @ (b[v, p] - b[u, p])
        
        # b[v, p] is equivalent to b[v][:, p]? No, b is 2D. b[v, p] is a 1D array.
        
        # Let's just do:
        # row_v_p = b[v, p]
        # row_u_p = b[u, p]
        # diff_b = row_v_p - row_u_p
        # delta = np.dot(diff_a, diff_b) + part1
        
        # This involves two gather operations of size n.
        # For n=150, this is very fast.
        
        u_val = p[i]
        v_val = p[j]
        
        diff_a = a[i] - a[j]
        diff_b = b[v_val, p] - b[u_val, p]
        
        delta = np.dot(diff_a, diff_b) + 2 * a[i, j] * b[u_val, v_val]
        return delta

    # Main Loop
    # We will perform multiple passes. If no improvement in a pass, we might restart with perturbation.
    # But simple 2-opt often gets stuck. Let's add a "best improvement" or "first improvement" strategy.
    # First improvement is faster.
    
    max_passes = 10000
    pass_count = 0
    
    while True:
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        # Randomize the order of pairs to avoid bias? 
        # Or just sequential. Sequential is deterministic.
        # Let's do sequential first.
        
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                delta = get_delta(i, j, p, a, b)
                
                if delta < 0:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    
                    # Update best if better
                    if cost < best_cost:
                        best_cost = cost
                        best_p = p.copy()
                        
                    # Restart scan from beginning to ensure stability
                    # This is O(n^3) worst case but effective for QAP
                    break
            if improved:
                break
        
        if not improved:
            # Local optimum reached
            # Try a perturbation if we have time?
            # Simple perturbation: swap a random pair and continue?
            # Or just break and rely on the initial solution quality?
            # Given the time limit, let's try a few random restarts if we finish early.
            
            elapsed = time.time() - start_time
            if elapsed > time_limit - 5:
                break
            
            # Perturb: pick 2 random distinct indices and swap them
            # Then continue local search
            if random.random() < 0.5:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    p[i], p[j] = p[j], p[i]
                    # Recalculate cost? 
                    # It's easier to just recalculate from scratch for safety after perturbation
                    cost = np.sum(a * b[p])
                    if cost < best_cost:
                        best_cost = cost
                        best_p = p.copy()
                    # Continue loop to improve from here
                    continue
            else:
                break

    return best_p.tolist()

# EVOLVE-BLOCK-END
