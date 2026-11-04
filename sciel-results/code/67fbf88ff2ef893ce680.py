import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # 20s margin for safety

    # Fallback: Rank 17 solution (Strassen + Direct)
    # This is a known valid solution.
    def get_rank17():
        rank = 17
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        def set_U(i, j, r, val): U[i * m + j, r] = val
        def set_V(j, k, r, val): V[j * p + k, r] = val
        def set_W(k, i, r, val): W[k * n + i, r] = val

        # Block 1: Strassen 2x2x2 (j in {0,1}, k in {0,1})
        # M1: (A00+A11)(B00+B11) -> C00, C11
        set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
        set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
        set_W(0, 0, 0, 1); set_W(1, 1, 0, 1)
        
        # M2: (A10+A11)B00 -> C10, -C11
        set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
        set_V(0, 0, 1, 1)
        set_W(0, 1, 1, 1); set_W(1, 1, 1, -1)
        
        # M3: A00(B01-B11) -> C01, C11
        set_U(0, 0, 2, 1)
        set_V(0, 1, 2, 1); set_V(1, 1, 2, -1)
        set_W(1, 0, 2, 1); set_W(1, 1, 2, 1)
        
        # M4: A11(B10-B00) -> C00, C10
        set_U(1, 1, 3, 1)
        set_V(1, 0, 3, 1); set_V(0, 0, 3, -1)
        set_W(0, 0, 3, 1); set_W(0, 1, 3, 1)
        
        # M5: (A00+A01)B11 -> -C00, C01
        set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
        set_V(1, 1, 4, 1)
        set_W(0, 0, 4, -1); set_W(1, 0, 4, 1)
        
        # M6: (A10-A00)(B00+B01) -> C11
        set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
        set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
        set_W(1, 1, 5, 1)
        
        # M7: (A01-A11)(B10+B11) -> C00
        set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
        set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
        set_W(0, 0, 6, 1)

        # Block 2: Direct 2x2x1 (j in {0,1}, k=2)
        # P1: A00 B02 -> C02
        set_U(0, 0, 7, 1)
        set_V(0, 2, 7, 1)
        set_W(2, 0, 7, 1)
        
        # P2: A01 B12 -> C02
        set_U(0, 1, 8, 1)
        set_V(1, 2, 8, 1)
        set_W(2, 0, 8, 1)
        
        # P3: A10 B02 -> C12
        set_U(1, 0, 9, 1)
        set_V(0, 2, 9, 1)
        set_W(2, 1, 9, 1)
        
        # P4: A11 B12 -> C12
        set_U(1, 1, 10, 1)
        set_V(1, 2, 10, 1)
        set_W(2, 1, 10, 1)

        # Block 3: Direct 2x1x3 (j=2, k in {0,1,2})
        # P5: A02 B20 -> C00
        set_U(0, 2, 11, 1)
        set_V(2, 0, 11, 1)
        set_W(0, 0, 11, 1)
        
        # P6: A02 B21 -> C01
        set_U(0, 2, 12, 1)
        set_V(2, 1, 12, 1)
        set_W(1, 0, 12, 1)
        
        # P7: A02 B22 -> C02
        set_U(0, 2, 13, 1)
        set_V(2, 2, 13, 1)
        set_W(2, 0, 13, 1)
        
        # P8: A12 B20 -> C10
        set_U(1, 2, 14, 1)
        set_V(2, 0, 14, 1)
        set_W(0, 1, 14, 1)
        
        # P9: A12 B21 -> C11
        set_U(1, 2, 15, 1)
        set_V(2, 1, 15, 1)
        set_W(1, 1, 15, 1)
        
        # P10: A12 B22 -> C12
        set_U(1, 2, 16, 1)
        set_V(2, 2, 16, 1)
        set_W(2, 1, 16, 1)

        return U, V, W

    U17, V17, W17 = get_rank17()
    best_rank = 17
    best_U, best_V, best_W = U17, V17, W17

    def compute_error(U, V, W, rank):
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        
        # Efficient vectorized error computation
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # We want T == 1
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over r
                    # U[i,j,:] * V[j,k,:] * W[k,i,:]
                    # Note: W is indexed [k, i] in the tensor W_t
                    s = np.dot(U_t[i, j, :], V_t[j, k, :] * W_t[k, i, :])
                    error += (s - 1.0) ** 2
        return error

    def is_exact(U, V, W, rank):
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_t[i, j, :], V_t[j, k, :] * W_t[k, i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_integers(U, V, W):
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        return U_s, V_s, W_s

    def snap_to_integers(U, V, W):
        U_s = np.round(U)
        V_s = np.round(V)
        W_s = np.round(W)
        return U_s, V_s, W_s

    # Structural Change: Local Search with Integer/Sparse constraints
    # Instead of pure gradient descent which wanders in continuous space,
    # we use a "Combinatorial" local search:
    # 1. Start from a valid high-rank solution or random sparse init.
    # 2. Try to remove a rank term (zero out a column) if it keeps validity or reduces error significantly.
    # 3. Try to modify specific entries to {-1, 0, 1} to reduce error.
    
    # For rank 16, we need to remove one term from rank 17 or find a new structure.
    # Since rank 17 is a "Strassen + Direct" hybrid, maybe we can merge two direct terms into one Strassen-like term?
    # Or perhaps use a different Strassen partitioning.
    
    # Let's try a randomized local search on the rank-17 solution to see if we can drop a term.
    # If we zero out column r, we introduce error. Can we adjust other columns to fix it?
    # This is hard.
    
    # Alternative: Randomized restart with "Integer" initialization.
    # Many known low-rank decompositions have integer entries.
    # Let's try random integer initializations in {-1, 0, 1} and gradient descent with frequent snapping.
    
    # We will focus on Rank 16.
    
    found_16 = False
    
    # Strategy A: Perturb Rank 17 and try to collapse.
    # This is risky.
    
    # Strategy B: Random Search with Snapping.
    # The search space of {-1,0,1}^R for R=16 is huge, but random restarts with GD might find a basin.
    
    # Let's try a few random seeds for rank 16.
    # We'll use a faster gradient computation.
    
    def fast_grad_descent(U, V, W, rank, steps, lr, snap_interval, snap_mode):
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        
        # Precompute indices for faster loops? 
        # n=2, m=3, p=3 is small.
        
        for step in range(steps):
            if time.time() > deadline:
                return False, U, V, W
            
            # Compute Residuals R[i,j,k]
            R = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # U[i,j,:] dot (V[j,k,:] * W[k,i,:])
                        R[i, j, k] = np.dot(U_t[i, j, :], V_t[j, k, :] * W_t[k, i, :]) - 1.0
            
            # Check convergence
            if np.max(np.abs(R)) < 1e-9:
                return True, U, V, W

            # Compute Gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res = R[i, j, k]
                        # dU[i,j,r] += 2 * res * V[j,k,r] * W[k,i,r]
                        # dV[j,k,r] += 2 * res * U[i,j,r] * W[k,i,r]
                        # dW[k,i,r] += 2 * res * U[i,j,r] * V[j,k,r]
                        
                        prod_VW = V_t[j, k, :] * W_t[k, i, :]
                        dU[i * m + j, :] += 2 * res * prod_VW
                        
                        prod_UW = U_t[i, j, :] * W_t[k, i, :]
                        dV[j * p + k, :] += 2 * res * prod_UW
                        
                        prod_UV = U_t[i, j, :] * V_t[j, k, :]
                        dW[k * n + i, :] += 2 * res * prod_UV
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Snap
            if step % snap_interval == 0:
                if snap_mode == 'half':
                    U, V, W = snap_to_half_integers(U, V, W)
                elif snap_mode == 'int':
                    U, V, W = snap_to_integers(U, V, W)
                else:
                    # Mix: snap 50% to int, 50% to half?
                    if random.random() > 0.5:
                        U, V, W = snap_to_integers(U, V, W)
                    else:
                        U, V, W = snap_to_half_integers(U, V, W)
            
            # Reshape back for next iteration
            U_t = U.reshape(n, m, rank)
            V_t = V.reshape(m, p, rank)
            W_t = W.reshape(p, n, rank)
            
        return False, U, V, W

    # Try Rank 16
    target_rank = 16
    seeds = range(200) # Increase seeds if time permits
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        np.random.seed(seed)
        
        # Initialize with sparse random integers
        # This mimics the structure of known decompositions
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        # For each rank r, pick random non-zero patterns
        # This is a heuristic: real decompositions are sparse.
        for r in range(target_rank):
            # Randomly select a few indices to be non-zero
            # U indices: i*m+j (0..5)
            # V indices: j*p+k (0..8)
            # W indices: k*n+i (0..5)
            
            # Ensure at least one non-zero in each matrix for this rank
            idx_u = random.sample(range(n*m), random.randint(1, 3))
            idx_v = random.sample(range(m*p), random.randint(1, 3))
            idx_w = random.sample(range(p*n), random.randint(1, 3))
            
            for idx in idx_u:
                U[idx, r] = random.choice([-1, 1])
            for idx in idx_v:
                V[idx, r] = random.choice([-1, 1])
            for idx in idx_w:
                W[idx, r] = random.choice([-1, 1])
        
        # Run GD
        success, U, V, W = fast_grad_descent(U, V, W, target_rank, steps=500, lr=0.05, snap_interval=10, snap_mode='mix')
        
        if success:
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            found_16 = True
            break
        
        # Also try starting from Rank 17 and removing a column?
        # This is complex. Let's stick to random restarts for now.
        # But maybe we should try starting from the Rank 17 solution, zeroing a column, and fixing?
        
        # Let's try a different initialization: Rank 17 with one column zeroed out.
        if seed % 10 == 0: # Every 10th seed, try this
            U, V, W = get_rank17()
            # Zero out a random column
            col_to_remove = random.randint(0, 16)
            U[:, col_to_remove] = 0
            V[:, col_to_remove] = 0
            W[:, col_to_remove] = 0
            
            # Now we have a rank-16 structure (effectively) but it's incorrect.
            # We need to adjust the remaining 16 columns to fix the error.
            # This is a constrained optimization.
            # Let's just run GD on the full 17x? No, we want rank 16.
            # So we take the first 16 columns of the modified U,V,W?
            # Actually, if we zero out col 16, we have 16 active columns.
            # Let's slice to 16 columns.
            U16 = U[:, :16]
            V16 = V[:, :16]
            W16 = W[:, :16]
            
            # But wait, if we zeroed col 5, we should keep the others.
            # Let's just shuffle columns so the zeroed one is last, then slice.
            # Simpler: just run GD on the 17-column matrix but force one column to stay 0?
            # No, the interface requires rank R.
            # So we must provide R x (n*m) matrices.
            # If we use R=16, we can't have 17 columns.
            
            # So, let's take the 17 solution, remove column `col_to_remove`, and run GD on the resulting 16-column matrices.
            cols = [c for c in range(17) if c != col_to_remove]
            U16 = U[:, cols]
            V16 = V[:, cols]
            W16 = W[:, cols]
            
            success, U16, V16, W16 = fast_grad_descent(U16, V16, W16, target_rank, steps=500, lr=0.05, snap_interval=10, snap_mode='mix')
            
            if success:
                best_rank = target_rank
                best_U, best_V, best_W = U16, V16, W16
                found_16 = True
                break

    # If 16 not found, try 15? Unlikely to be found quickly.
    # Return best found
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
