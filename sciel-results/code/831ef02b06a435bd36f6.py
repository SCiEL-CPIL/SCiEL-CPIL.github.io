"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes: U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    nm = n * m  # 16
    mp = m * p  # 32
    pn = p * n  # 32
    
    # Build the list of (i,j,k) triples and their corresponding indices
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                triples.append((u_idx, v_idx, w_idx))
    
    num_triples = len(triples)  # 4*4*8 = 128
    schoolbook_rank = n * m * p  # 128
    
    def verify_numpy(U, V, W, rank):
        """Vectorized verification."""
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        # For each triple, compute sum_r U[u,r]*V[v,r]*W[w,r]
        for (u_idx, v_idx, w_idx) in triples:
            val = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
            if abs(val - 1.0) > 1e-6:
                return False
        return True

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def gradient_search(rank, num_restarts=10, iters_per_restart=500, lr=0.01):
        """Try to find rank-r decomposition via gradient descent."""
        np_rng = np.random.RandomState(rank * 7 + 13)
        
        best_loss = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(num_restarts):
            if time.time() - start_time > deadline:
                break
                
            U = np_rng.randn(nm, rank) * 0.5
            V = np_rng.randn(mp, rank) * 0.5
            W = np_rng.randn(pn, rank) * 0.5
            
            for it in range(iters_per_restart):
                if time.time() - start_time > deadline:
                    break
                    
                # Compute loss and gradients
                loss = 0.0
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for (u_idx, v_idx, w_idx) in triples:
                    # val = sum_r U[u,r] * V[v,r] * W[w,r]
                    val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                    err = val - 1.0
                    loss += err * err
                    # dU[u,r] += 2*err * V[v,r] * W[w,r]
                    # dV[v,r] += 2*err * U[u,r] * W[w,r]
                    # dW[w,r] += 2*err * U[u,r] * V[v,r]
                    dU[u_idx] += 2 * err * (V[v_idx] * W[w_idx])
                    dV[v_idx] += 2 * err * (U[u_idx] * W[w_idx])
                    dW[w_idx] += 2 * err * (U[u_idx] * V[v_idx])
                
                if loss < best_loss:
                    best_loss = loss
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if loss < 1e-8:
                    break
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            if best_loss < 1e-6:
                break
        
        if best_loss < 1e-6:
            # Convert to lists
            U_list = [[float(x) for x in row] for row in best_U]
            V_list = [[float(x) for x in row] for row in best_V]
            W_list = [[float(x) for x in row] for row in best_W]
            return U_list, V_list, W_list, best_loss
        return None

    # Try a few ranks
    for r in [128, 120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32]:
        if time.time() - start_time > deadline:
            break
        
        result = gradient_search(r, num_restarts=5, iters_per_restart=200, lr=0.005)
        if result is not None:
            U_list, V_list, W_list, loss = result
            if verify_numpy(U_list, V_list, W_list, r):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
