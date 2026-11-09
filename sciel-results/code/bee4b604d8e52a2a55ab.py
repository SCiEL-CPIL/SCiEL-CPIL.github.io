"""
Program for matrix multiplication tensor decomposition 4x4 * 4x6.
Uses a model-and-optimize approach: start from Strassen-based block decomposition,
then try to reduce rank via sparse random initialization + iterative refinement.
"""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 120.0  # safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # First, build a valid decomposition using schoolbook (guaranteed correct)
    rank_school = n * m * p  # 96
    best_rank = rank_school
    best_U = None
    best_V = None
    best_W = None
    
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U, V, W, best_rank = make_schoolbook()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def verify_full(U, V, W):
        """Full verification of all (i,j,k) triples."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True
    
    def snap_to_halfint(U, V, W):
        """Snap values to nearest half-integer and verify."""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        if verify_full(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
        return None, None, None
    
    def try_rank(rank, max_iter=3000, seed=42):
        """Try to find a rank-r decomposition using gradient descent."""
        if rank >= best_rank:
            return None, None, None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(n*m, rank) * 0.5
        V = rng.randn(m*p, rank) * 0.5
        W = rng.randn(p*n, rank) * 0.5
        
        # Precompute index arrays for fast access
        idx_U = np.array([i*m+j for i in range(n) for j in range(m)])
        idx_V = np.array([j*p+k for j in range(m) for k in range(p)])
        idx_W = np.array([k*n+i for k in range(p) for i in range(n)])
        
        # Build the full tensor of (i,j,k) triples
        triples = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    triples.append((i, j, k))
        
        # For each triple, we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        lr = 0.05
        best_loss = float('inf')
        
        for it in range(max_iter):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Compute loss and gradients
            loss = 0.0
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for (i, j, k) in triples:
                ui = i*m + j
                vj = j*p + k
                wk = k*n + i
                
                # Current value
                val = np.dot(U[ui, :], V[vj, :] * W[wk, :])
                err = val - 1.0
                loss += err * err
                
                # Gradients
                # d/dU[ui,r] = 2*err * V[vj,r] * W[wk,r]
                # d/dV[vj,r] = 2*err * U[ui,r] * W[wk,r]
                # d/dW[wk,r] = 2*err * U[ui,r] * V[vj,r]
                dU[ui, :] += 2 * err * V[vj, :] * W[wk, :]
                dV[vj, :] += 2 * err * U[ui, :] * W[wk, :]
                dW[wk, :] += 2 * err * U[ui, :] * V[vj, :]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check convergence
            if loss < 1e-12:
                break
            
            # Adaptive learning rate
            if loss < best_loss * 0.99:
                best_loss = loss
            else:
                lr *= 0.95
        
        # Final check
        if verify_full(U, V, W):
            U_s, V_s, W_s = snap_to_halfint(U, V, W)
            if U_s is not None:
                return U_s, V_s, W_s
            return U, V, W
        
        U_s, V_s, W_s = snap_to_halfint(U, V, W)
        if U_s is not None:
            return U_s, V_s, W_s
            
        return None, None, None
    
    # Try ranks from low to high
    trial_ranks = [32, 36, 40, 44, 48, 50, 52, 56, 60, 64, 70, 76, 80, 84, 88, 92]
    
    for target_rank in trial_ranks:
        if target_rank >= best_rank:
            continue
        if time.time() - start_time > DEADLINE - 10:
            break
        
        found = False
        for seed in range(8):
            if time.time() - start_time > DEADLINE - 5:
                break
            U_r, V_r, W_r = try_rank(target_rank, max_iter=2000, seed=seed + 100)
            if U_r is not None:
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                found = True
                break
        if found:
            break
    
    # Convert to list format
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
