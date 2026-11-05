"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for fast operations
    A = np.array(adjacency, dtype=np.float64)
    
    start_time = time.time()
    deadline = 145.0
    
    def calc_cut_size(labels):
        # labels are 0/1.
        # Cut size = sum_{i,j} A[i][j] * (labels[i] != labels[j]) / 2
        mask = labels[:, None] != labels[None, :]
        return 0.5 * np.sum(A * mask)

    def local_search(labels):
        # Steepest ascent local search (sequential, fast for n=100)
        x = labels.copy()
        x[x == 0] = -1
        x[x == 1] = 1
        
        improved = True
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            
            Ax = A @ x
            gains = x * Ax
            
            max_gain_idx = np.argmax(gains)
            max_gain = gains[max_gain_idx]
            
            if max_gain > 0:
                x[max_gain_idx] = -x[max_gain_idx]
                improved = True
            else:
                improved = False
                break
        
        labels_out = (x > 0).astype(np.int8)
        return labels_out

    def spectral_init():
        """
        Use the Fiedler vector (eigenvector corresponding to the second smallest eigenvalue)
        to create a high-quality initial partition. This is often much better than random.
        """
        try:
            # Use eigh for symmetric matrix
            eigvals, eigvecs = np.linalg.eigh(A)
            # Sort by eigenvalue
            idx = np.argsort(eigvals)
            eigvecs = eigvecs[:, idx]
            
            # The Fiedler vector is the eigenvector for the second smallest eigenvalue.
            # For a connected graph, lambda_0 = 0, lambda_1 is the Fiedler value.
            # If the graph is disconnected, there might be multiple zeros.
            # We generally want the vector that best separates the graph.
            # Taking the eigenvector with the largest absolute values or the one
            # corresponding to the first non-zero eigenvalue is standard.
            
            # Let's pick the eigenvector associated with the smallest non-zero eigenvalue.
            # If the graph is connected, this is index 1.
            # If not, we might want to try a few or just take index 1 (which will be 0 for components).
            # A robust choice for MaxCut initialization is often the eigenvector of the 
            # smallest eigenvalue of the Laplacian, but here we use A directly.
            # Actually, for MaxCut, the spectral method usually involves the eigenvector 
            # corresponding to the smallest eigenvalue of the adjacency matrix? No.
            # The standard spectral initialization for MaxCut uses the eigenvector 
            # corresponding to the smallest eigenvalue of the matrix (A - dI) or similar?
            # Actually, a common simple heuristic is to use the eigenvector corresponding 
            # to the smallest eigenvalue of the adjacency matrix A?
            # Let's check: Minimize x^T A x. The global minimum is the smallest eigenvalue.
            # The eigenvector corresponding to the smallest eigenvalue minimizes x^T A x 
            # subject to ||x||=1.
            # So we should use the eigenvector for the SMALLEST eigenvalue.
            
            v = eigvecs[:, 0]
            
            # Threshold at 0
            labels = (v >= 0).astype(np.int8)
            
            # Ensure it's not a trivial all-0 or all-1 solution (unlikely with eigenvectors)
            if np.all(labels == 0) or np.all(labels == 1):
                # Fallback to random if degenerate
                return np.random.randint(0, 2, size=n).astype(np.int8)
                
            return labels
        except Exception:
            return np.random.randint(0, 2, size=n).astype(np.int8)

    # 1. Get a strong initial solution via Spectral Init + Local Search
    init_labels = spectral_init()
    init_labels = local_search(init_labels)
    best_cut = calc_cut_size(init_labels)
    best_labels = init_labels.copy()
    
    # Also try a random start + local search as a backup
    rand_labels = np.random.randint(0, 2, size=n).astype(np.int8)
    rand_labels = local_search(rand_labels)
    rand_cut = calc_cut_size(rand_labels)
    if rand_cut > best_cut:
        best_cut = rand_cut
        best_labels = rand_labels.copy()

    # 2. Parallel Batch Simulated Annealing
    # We will maintain a population of M spin configurations.
    # Spins are stored as a matrix X of shape (M, n) with values -1 and 1.
    # We update all M configurations in parallel using vectorized numpy operations.
    
    M = 1000  # Population size. 1000 * 100 floats = 800KB, very small.
    
    # Initialize population.
    # Mix of spectral, random, and perturbed best.
    X = np.zeros((M, n), dtype=np.float64)
    
    # 10% Spectral
    for i in range(int(0.1 * M)):
        X[i] = (spectral_init() * 2 - 1).astype(np.float64)
        
    # 40% Random
    for i in range(int(0.1 * M), int(0.5 * M)):
        X[i] = (np.random.randint(0, 2, size=n) * 2 - 1).astype(np.float64)
        
    # 50% Perturbed Best (Local Search results from random/spectral)
    # Actually, let's just use random for the rest to ensure diversity, 
    # or perturb the current best_labels.
    for i in range(int(0.5 * M), M):
        # Start with best, flip some random vertices
        x_i = best_labels.copy()
        num_flips = np.random.randint(1, 10)
        flip_idx = np.random.choice(n, num_flips, replace=False)
        x_i[flip_idx] = 1 - x_i[flip_idx]
        X[i] = (x_i * 2 - 1).astype(np.float64)

    # Track best cut for each individual in the population
    # Cut = 0.25 * (sum(A) - sum(X * (A @ X.T)))? 
    # Let's derive the vectorized cut calculation.
    # For a single spin vector x (shape n), Cut = 0.25 * (sum(A) - x^T A x).
    # For a batch X (shape M, n):
    # AX = X @ A.T (shape M, n). Since A is symmetric, A.T = A.
    # x^T A x for each row is sum(X * (X @ A), axis=1).
    
    sum_A = np.sum(A)
    
    # Pre-compute AX for the initial state
    AX = X @ A.T
    xT_A_x = np.sum(X * AX, axis=1)
    cuts = 0.25 * (sum_A - xT_A_x)
    
    # Track the best individual in the population
    best_idx = np.argmax(cuts)
    if cuts[best_idx] > best_cut:
        best_cut = cuts[best_idx]
        best_labels = ((X[best_idx] + 1) / 2).astype(np.int8)
    
    # SA Parameters
    T_start = 5.0
    T_end = 0.01
    num_steps = 500000  # Total steps to attempt. We will run in batches.
    
    # We will process the population in chunks? No, we process ALL M individuals 
    # in parallel for each "step" of the annealing schedule.
    # So "num_steps" here refers to the number of annealing iterations.
    # In each iteration, every individual in the population makes one move.
    # This is extremely fast because it's vectorized.
    
    # Let's estimate speed.
    # Per iteration:
    # 1. Random indices for M individuals: M ops
    # 2. Flip those vertices in X: M ops
    # 3. Calculate delta energy:
    #    We need Ax for the specific vertex v for each individual i.
    #    This is tricky to vectorize perfectly without gathering.
    #    
    #    Alternative: Update the whole X, then recalculate cuts for all?
    #    Recalculating cuts for all M individuals:
    #    AX = X @ A.T (M x n @ n x n = M x n) -> O(M n^2)
    #    xT_A_x = sum(X * AX) -> O(M n)
    #    For M=1000, n=100: 1000 * 10000 = 10^7 ops per iteration.
    #    50000 iterations -> 5 * 10^11 ops. This is too slow in Python/NumPy overhead?
    #    NumPy is fast. 10^7 float ops is ~1ms? 
    #    50000 * 1ms = 50s. This fits in the budget!
    #    
    #    However, we can be smarter. We only flip ONE vertex per individual per step.
    #    Delta Energy for flipping v in x:
    #    dE = -2 x_v (Ax)_v
    #    We can precompute Ax for all individuals?
    #    AX = X @ A.T.
    #    For individual i, flipping v_i changes x_i[v_i].
    #    The change in Ax for individual i is:
    #    d(Ax)_i = -2 x_i[v_i] A[:, v_i]  (vector of length n)
    #    This is hard to vectorize for arbitrary v_i per row.
    #    
    #    Let's stick to the "Recalculate All" method if it's fast enough, 
    #    or use a simpler vectorized SA step.
    #    
    #    Actually, there is a trick. 
    #    If we flip v_i for row i:
    #    New X[i, v_i] = -X[i, v_i]
    #    New AX[i, :] = AX[i, :] - 2 * X[i, v_i] * A[:, v_i]
    #    
    #    We can do this:
    #    1. v_idx = np.random.randint(0, n, M)
    #    2. old_vals = X[np.arange(M), v_idx]
    #    3. X[np.arange(M), v_idx] = -old_vals
    #    4. Update AX:
    #       # A[:, v_idx] is tricky because v_idx varies per row.
    #       # We can use np.take_along_axis or advanced indexing.
    #       # Let's just recalculate AX = X @ A.T. It's simpler and likely fast enough for n=100.
    
    for step in range(num_steps):
        if time.time() > deadline - 5:
            break
            
        # Cooling
        T = T_start * (T_end / T_start) ** (step / num_steps)
        
        # 1. Choose random vertex to flip for each individual
        v_idx = np.random.randint(0, n, M)
        
        # 2. Flip spins
        rows = np.arange(M)
        old_vals = X[rows, v_idx]
        X[rows, v_idx] = -old_vals
        
        # 3. Calculate Delta Energy for each individual
        # We need Ax[v_idx] for each row.
        # Let's compute the full AX = X @ A.T
        AX = X @ A.T
        # Get the specific component
        Ax_v = AX[rows, v_idx]
        
        # Delta E = -2 * x_v * Ax_v
        # Note: x_v is the NEW value? 
        # Let's check.
        # E = x^T A x.
        # Flip v: x_v -> -x_v.
        # E_new = E_old - 2 x_v (Ax)_v_old?
        # (Ax)_v_new = (Ax)_v_old - 2 x_v A[v,v]? No, A[v,v]=0.
        # (Ax)_v_new = (Ax)_v_old - 2 x_v * sum_{j!=v} A[v,j] x_j? No.
        # Let's use the formula: dE = -2 x_v (Ax)_v where x_v is the OLD value and (Ax)_v is the OLD Ax.
        # Wait, if x_v changes to -x_v, the term x_v (Ax)_v becomes -x_v (Ax)_v.
        # Change = -2 x_v (Ax)_v.
        # So we need OLD x_v and OLD Ax_v.
        # We have old_vals (which is OLD x_v).
        # But we just updated X and AX.
        # So we need to calculate Ax_v BEFORE updating X?
        # Or calculate it using old_vals and the PREVIOUS AX?
        
        # Let's swap the order:
        # 1. Get v_idx.
        # 2. Get old_vals = X[rows, v_idx].
        # 3. Get old_Ax_v = AX[rows, v_idx] (using current AX before update).
        # 4. Calculate dE = -2 * old_vals * old_Ax_v.
        # 5. Acceptance decision.
        # 6. Update X and AX.
        
        # Let's re-implement the loop body with this order.
        
        # Undo the flip for now to restart logic cleanly in the loop below.
        X[rows, v_idx] = old_vals 
        
        # --- Corrected Loop Body ---
        
        # 1. Choose random vertex
        v_idx = np.random.randint(0, n, M)
        rows = np.arange(M)
        
        # 2. Get old values
        old_vals = X[rows, v_idx]
        old_Ax_v = AX[rows, v_idx]
        
        # 3. Calculate Delta Energy
        dE = -2.0 * old_vals * old_Ax_v
        
        # 4. Acceptance
        # Accept if dE < 0 OR random < exp(-dE/T)
        # For dE >= 0, prob = exp(-dE/T)
        # For dE < 0, prob = 1
        # We can use: prob = np.where(dE < 0, 1.0, np.exp(-dE / T))
        # But exp(-dE/T) for large dE is 0.
        
        # Optimization: Only calculate exp for dE >= 0
        rand = np.random.random(M)
        
        # Mask for dE < 0
        accept_mask = dE < 0
        # For dE >= 0
        if T > 1e-9:
            exp_vals = np.exp(-dE[~accept_mask] / T)
            accept_mask[~accept_mask] = (rand[~accept_mask] < exp_vals)
        else:
            accept_mask[~accept_mask] = False
        
        # 5. Apply accepted flips
        if np.any(accept_mask):
            # Flip spins in X
            X[rows[accept_mask], v_idx[accept_mask]] = -X[rows[accept_mask], v_idx[accept_mask]]
            
            # Update AX
            # AX[i, :] = AX[i, :] - 2 * old_vals[i] * A[:, v_idx[i]]
            # We need to add the change to AX.
            # Change in AX for row i is: -2 * old_vals[i] * A[:, v_idx[i]]
            # Let's construct this vectorized.
            # A_cols = A[:, v_idx[accept_mask]] -> shape (n, K) where K is count of accepted
            # old_vals_acc = old_vals[accept_mask] -> shape (K,)
            # Delta_AX = -2 * old_vals_acc[:, None] * A_cols
            # AX[rows[accept_mask], :] += Delta_AX
            
            acc_rows = rows[accept_mask]
            acc_v_idx = v_idx[accept_mask]
            acc_old_vals = old_vals[accept_mask]
            
            A_cols = A[:, acc_v_idx] # (n, K)
            Delta_AX = -2.0 * acc_old_vals[:, None] * A_cols # (K, n)
            
            AX[acc_rows, :] += Delta_AX
            
            # Update cuts for accepted individuals
            # Cut = 0.25 * (sum_A - x^T A x)
            # We can update x^T A x incrementally?
            # Or just recalculate cuts for the accepted ones?
            # x^T A x = sum(X * AX, axis=1)
            # It's easier to just recalculate the cut for the accepted rows.
            # new_xT_A_x = np.sum(X[acc_rows] * AX[acc_rows], axis=1)
            # new_cuts = 0.25 * (sum_A - new_xT_A_x)
            
            # Actually, we can update the global cuts array.
            # Let's just recalculate cuts for ALL individuals every N steps?
            # No, that's expensive.
            # Let's update cuts for accepted individuals.
            
            new_xT_A_x = np.sum(X[acc_rows] * AX[acc_rows], axis=1)
            new_cuts = 0.25 * (sum_A - new_xT_A_x)
            cuts[acc_rows] = new_cuts
            
            # Check for global best
            best_in_batch = np.argmax(new_cuts)
            if new_cuts[best_in_batch] > best_cut:
                best_cut = new_cuts[best_in_batch]
                best_labels = ((X[acc_rows[best_in_batch]] + 1) / 2).astype(np.int8)
        
        # Periodic check for best in the whole population?
        # We track cuts for all. 
        # We can check if the max of the whole cuts array is better than best_cut.
        # But we only update cuts for accepted individuals. 
        # The unaccepted ones remain valid.
        # So we can just check:
        if step % 100 == 0:
            global_best_idx = np.argmax(cuts)
            if cuts[global_best_idx] > best_cut:
                best_cut = cuts[global_best_idx]
                best_labels = ((X[global_best_idx] + 1) / 2).astype(np.int8)
                
                # Optional: Reset some population members to the new best?
                # This is "elitism". Let's copy the best to the first 10% of the population.
                # This helps convergence.
                elite_count = int(0.1 * M)
                X[:elite_count] = X[global_best_idx]
                AX[:elite_count] = AX[global_best_idx]
                cuts[:elite_count] = cuts[global_best_idx]

    # Final Polish
    # Run local search on the best found
    final_labels = local_search(best_labels)
    final_cut = calc_cut_size(final_labels)
    
    if final_cut > best_cut:
        best_cut = final_cut
        best_labels = final_labels

    return best_labels
# EVOLVE-BLOCK-END
