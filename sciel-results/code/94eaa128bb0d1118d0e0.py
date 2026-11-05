import numpy as np
import time

def partition(adjacency):
    # Convert to numpy array if not already
    A = np.array(adjacency, dtype=np.float64)
    n = A.shape[0]
    
    # Precompute degrees
    deg = A.sum(axis=1)
    
    # Helper to evaluate cut size
    # Cut = 0.5 * (sum(deg) - sum(x * (A @ x))) ? 
    # Let's use the formula: Cut = 0.5 * (n_total_edges * 2 - sum(x_i * x_j * A_ij) * 2 ?)
    # Simpler: 
    # If x is 0/1, Cut = sum_{i<j, A_ij=1} (x_i != x_j)
    # This equals 0.5 * sum_{i,j} A_ij (x_i - x_j)^2 / 2 ? No.
    # (x_i - x_j)^2 is 1 if different, 0 if same.
    # So Cut = 0.5 * sum_{i,j} A_ij (x_i - x_j)^2 
    # = 0.5 * (sum_{i,j} A_ij (x_i^2 - 2x_i x_j + x_j^2))
    # Since x_i^2 = x_i for binary:
    # = 0.5 * (sum_{i,j} A_ij x_i - 2 sum_{i,j} A_ij x_i x_j + sum_{i,j} A_ij x_j)
    # = 0.5 * (sum_i x_i (sum_j A_ij) - 2 x^T A x + sum_j x_j (sum_i A_ij))
    # = 0.5 * (2 * x^T deg - 2 x^T A x)  (since A is symmetric, sum_i A_ij = deg_j)
    # = x^T deg - x^T A x
    # = sum(deg * x) - x @ A @ x
    
    def evaluate(x):
        return np.dot(deg, x) - np.dot(x, A.dot(x))
    
    best_x = None
    best_score = -1.0
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Initial best from a few quick random starts
    for _ in range(5):
        x = np.random.randint(0, 2, n).astype(np.float64)
        score = evaluate(x)
        if score > best_score:
            best_score = score
            best_x = x.copy()
            
    # Local Search Function
    def local_search(x):
        # x is 0/1 float array
        Ax = A.dot(x)
        current_score = evaluate(x)
        
        improved = True
        while improved:
            improved = False
            # Compute gains for flipping each node
            # Gain = deg - 2 * Ax
            gains = deg - 2.0 * Ax
            
            # Find node with max positive gain
            max_gain = np.max(gains)
            if max_gain <= 1e-9:
                break
            
            v = np.argmax(gains)
            
            # Flip v
            old_val = x[v]
            new_val = 1.0 - old_val
            x[v] = new_val
            
            # Update Ax
            # Ax_new = Ax_old + (new_val - old_val) * A[:, v]
            # new_val - old_val is +1 if 0->1, -1 if 1->0
            delta_val = new_val - old_val
            Ax += delta_val * A[:, v]
            
            # Update score incrementally or recompute?
            # Incremental is safer/faster
            # Score change:
            # Old contrib: x_v * deg_v - (x_v * (Ax_old)_v) ... wait, x^T A x is sum x_i (Ax)_i
            # Let's just recompute score if we want to be safe, but incremental is:
            # S = sum(deg*x) - x^T A x
            # Delta S = deg_v * delta_val - Delta(x^T A x)
            # x^T A x = sum_i x_i (Ax)_i
            # This is tricky to update incrementally without errors.
            # Given n=100, evaluating score is O(n^2) = 10k ops.
            # Local search steps might be ~n. Total O(n^3) per run? No, O(n^2) per step.
            # Actually, just recomputing `evaluate(x)` is fast enough in NumPy for n=100.
            # But we are in a tight loop. Let's stick to the gain logic for termination,
            # but recompute score to track best.
            
            current_score = evaluate(x)
            if current_score > best_score:
                best_score = current_score
                best_x = x.copy()
            
            improved = True
            # Note: We might oscillate if gains are equal? 
            # Gains are strictly determined by state. If max_gain > 0, we move.
            # Since finite states, we will terminate.
            
        return x, current_score

    # Main Loop: Random Restarts
    # We can also use "Kicks" (flip k random nodes) from the best solution to escape local optima
    # Strategy: Alternate between pure random restarts and kicks from best.
    
    while time.time() - start_time < time_limit:
        # Choose strategy: 50% Random, 50% Kick from Best
        if np.random.rand() < 0.5:
            x = np.random.randint(0, 2, n).astype(np.float64)
        else:
            if best_x is not None:
                x = best_x.copy()
                # Kick: flip 5 random nodes
                kick_nodes = np.random.choice(n, 5, replace=False)
                x[kick_nodes] = 1.0 - x[kick_nodes]
            else:
                x = np.random.randint(0, 2, n).astype(np.float64)
        
        x, score = local_search(x)
        
        if score > best_score:
            best_score = score
            best_x = x.copy()

    # Final Polish: Run local search on best_x again to ensure it's a local optimum
    if best_x is not None:
        x, score = local_search(best_x.copy())
        if score > best_score:
            best_score = score
            best_x = x.copy()

    # Return as 0/1 labels
    # The task requires 0/1 label per vertex.
    # best_x is float 0/1.
    return best_x.astype(int)
