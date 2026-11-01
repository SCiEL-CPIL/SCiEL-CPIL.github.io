"""Matrix multiplication 2x5x5 - attempt at lower rank than schoolbook (50)."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]
N, M, P = 2, 5, 5

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the trivial rank-50 decomposition."""
    rank = N * M * P
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, trials=200, tol=1e-10):
    """Check that sum_c U[a,c]*V[b,c]*W[c] == T[a,b] for random inputs.
    U has shape (N*M, rank), V has shape (M*P, rank), W has shape (P*N, rank).
    The tensor T[i,j,k] = delta[i,j,k] for matmul 2x5 x 5x5.
    We check: for each (i,j,k), sum_c U[i*M+j, c] * V[j*P+k, c] * W[k*N+i, c] == 1 iff (i,j,k) is a valid multiplication term.
    
    Actually, the decomposition says:
    (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    = sum_c (sum_j U[i*M+j, c] * A[i,j]) * (sum_k W[k*N+i, c] * B[j,k]) ... 
    
    Wait, let me re-derive. The standard decomposition of matrix multiplication tensor T_{n,m,p}:
    T[i,j,k] = 1 (the tensor has 1s at all (i,j,k) positions for the multiplication tensor)
    
    No wait. The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k when we define it as the bilinear map.
    Actually T[i,j,k] = delta_{i,j,k} doesn't make sense. Let me think again.
    
    The multiplication of A (n x m) by B (m x p) gives C (n x p) where C[i,k] = sum_j A[i,j] B[j,k].
    
    The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p), and T[i,j,k] = 1.
    The decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    where U has shape (n*m, rank) indexed as U[i*m+j, c],
    V has shape (m*p, rank) indexed as V[j*p+k, c],  
    W has shape (p*n, rank) indexed as W[k*n+i, c].
    
    So we need: for all (i,j,k), sum_c U[i*M+j, c] * V[j*P+k, c] * W[k*N+i, c] = 1.
    """
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check all (i,j,k) triples
    for i in range(N):
        for j in range(M):
            for k in range(P):
                u_idx = i * M + j
                v_idx = j * P + k
                w_idx = k * N + i
                val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_search(deadline):
    """Try to find a lower-rank decomposition using random initialization + gradient-like search.
    This is a heuristic attempt; may not succeed within time budget."""
    
    # Try ranks from 34 upward (lower bound is 34)
    # For each rank, try random initializations and check if we can solve the system.
    
    # The constraint is: for all (i,j,k) in range(N)*range(M)*range(P):
    #   sum_c U[i*M+j, c] * V[j*P+k, c] * W[k*N+i, c] = 1
    # This is a system of N*M*P = 50 equations in (N*M + M*P + P*N) * rank variables.
    # But it's not linear - it's trilinear.
    
    # Alternative approach: fix W to identity-like structure and solve for U, V.
    # Or use the fact that for n=2, we can write:
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[1,k] = sum_j A[1,j] B[j,k]
    
    # Let's try a structured approach:
    # For n=2, we can use the "bipartite" structure.
    # Write the multiplication as:
    # [C_0; C_1] = [A_0 * B; A_1 * B]
    # where A_0, A_1 are row vectors of length 5, B is 5x5.
    
    # This is equivalent to computing two independent 1x5 x 5x5 multiplications,
    # but they share the same B. So we can potentially share computation.
    
    # The rank of a single 1x5 x 5x5 multiplication is at most 25 (schoolbook),
    # and the lower bound is... well, it's just a bilinear map from (R^5 x R^{25}) to R^5.
    # The rank is at most 25 and at least 5 (since the output is 5-dimensional and input has 5+25=30 dims).
    
    # For two such multiplications sharing B:
    # C_0 = A_0 * B, C_1 = A_1 * B
    # Total output: 10 values from inputs: 10 A-values + 25 B-values.
    
    # One approach: compute B^T or use column operations.
    # Actually, C_0 = A_0 * B means C_0[k] = sum_j A_0[j] B[j,k]
    # C_1 = A_1 * B means C_1[k] = sum_j A_1[j] B[j,k]
    
    # Let D[j,k] = B[j,k]. Then:
    # C_0[k] = sum_j A_0[j] D[j,k]
    # C_1[k] = sum_j A_1[j] D[j,k]
    
    # This is: [C_0; C_1] = D^T * [A_0; A_1]  ... no wait.
    # C_0[k] = sum_j A_0[j] * D[j,k] = (A_0 * D)[k]
    # C_1[k] = sum_j A_1[j] * D[j,k] = (A_1 * D)[k]
    
    # So [C_0; C_1]^T = D^T * [A_0^T; A_1^T]
    # This is a 5x2 matrix = 5x5 matrix * 5x2 matrix.
    # The rank of this multiplication (5x5 * 5x2 -> 5x2) is what we need.
    
    # The schoolbook rank for 5x5 * 5x2 is 5*5*2 = 50.
    # But we can do better!
    
    # For multiplying a 5x5 matrix by a 5x2 matrix, we can use the structure.
    # Actually, this is the same as computing D^T * [A_0; A_1] where D^T is 5x5 and [A_0;A_1] is 5x2.
    # Result is 5x2.
    
    # The tensor rank of M_{5,5,2} (multiplying 5x5 by 5x2 to get 5x2) ... 
    # Wait, I'm getting confused with the indexing. Let me restart.
    
    # The original problem: A is 2x5, B is 5x5, C = A*B is 2x5.
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    
    # The decomposition uses:
    # U[i*M+j, c] = U[i*5+j, c] for i in {0,1}, j in {0..4}
    # V[j*P+k, c] = V[j*5+k, c] for j in {0..4}, k in {0..4}
    # W[k*N+i, c] = W[k*2+i, c] for k in {0..4}, i in {0,1}
    
    # Constraint: for all i in {0,1}, j in {0..4}, k in {0..4}:
    #   sum_c U[5i+j, c] * V[5j+k, c] * W[2k+i, c] = 1
    
    # That's 2*5*5 = 50 equations.
    # Number of unknowns: (2*5 + 5*5 + 5*2) * rank = (10 + 25 + 10) * rank = 45 * rank.
    
    # For rank r, we have 45r unknowns and 50 equations (but they're nonlinear - trilinear).
    
    # Strategy: Fix W arbitrarily (or in a structured way) and solve for U, V.
    # If we fix W, then for each c, we need:
    #   sum_c U[5i+j, c] * V[5j+k, c] * W[2k+i, c] = 1
    # This is still nonlinear in U and V.
    
    # Alternative: For fixed U and W, the equation is linear in V:
    #   For each (i,j,k): sum_c U[5i+j,c] * W[2k+i,c] * V[5j+k,c] = 1
    # Let's define for each j, k: coefficient_{j,k,c} = sum over i of ... no wait.
    # For fixed (j,k), the equation is:
    #   sum_c (sum over i? No, each (i,j,k) is a separate equation)
    
    # Let me index differently. For each triple (i,j,k):
    #   sum_c U_{5i+j, c} * V_{5j+k, c} * W_{2k+i, c} = 1
    
    # Group by (j,k): for fixed j,k, we have 2 equations (i=0,1):
    #   i=0: sum_c U_{j,c} * V_{5j+k,c} * W_{2k,c} = 1
    #   i=1: sum_c U_{5+j,c} * V_{5j+k,c} * W_{2k+1,c} = 1
    
    # This is getting complex. Let me try a different approach: random search with restarts.
    
    best_rank = N * M * P  # 50
    best_result = _schoolbook()
    
    for rank in range(34, best_rank):
        if time.time() > deadline:
            break
        
        max_trials = 20
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(N * M, rank)
            V = np.random.randn(M * P, rank)
            W = np.random.randn(P * N, rank)
            
            # Check if this happens to satisfy the constraints (very unlikely)
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Try a structured approach: set some entries to 1 and others to 0
            # based on the schoolbook pattern but with shared columns
            
            if trial == 0:
                # Structured: use all 50 schoolbook terms but try to merge some
                # This is heuristic and probably won't work directly
                pass
            
            # Gradient descent on the residual
            # Residual: for each (i,j,k), r_{i,j,k} = 1 - sum_c U_{5i+j,c} V_{5j+k,c} W_{2k+i,c}
            # Minimize sum r_{i,j,k}^2
            
            # Compute gradients (this is expensive but rank is small)
            num_iters = 50
            lr = 0.01
            for it in range(num_iters):
                if time.time() > deadline:
                    break
                
                # Compute residual
                U_arr = U
                V_arr = V
                W_arr = W
                
                # For each (i,j,k): residual = 1 - sum_c U[5i+j,c]*V[5j+k,c]*W[2k+i,c]
                # Gradient w.r.t U[5i+j,c]: -W[2k+i,c]*V[5j+k,c] * residual
                # But this sum is over k (and the equation is per (i,j,k))
                
                # Let's compute it properly
                grad_U = np.zeros_like(U_arr)
                grad_V = np.zeros_like(V_arr)
                grad_W = np.zeros_like(W_arr)
                
                for i in range(N):
                    for j in range(M):
                        for k in range(P):
                            ui = 5 * i + j
                            vi = 5 * j + k
                            wi = 2 * k + i
                            prod = U_arr[ui] * V_arr[vi] * W_arr[wi]
                            res = 1.0 - np.sum(prod)
                            # grad_U[ui, c] += -res * V_arr[vi, c] * W_arr[wi, c]
                            # grad_V[vi, c] += -res * U_arr[ui, c] * W_arr[wi, c]
                            # grad_W[wi, c] += -res * U_arr[ui, c] * V_arr[vi, c]
                            grad_U[ui] += -res * (V_arr[vi] * W_arr[wi])
                            grad_V[vi] += -res * (U_arr[ui] * W_arr[wi])
                            grad_W[wi] += -res * (U_arr[ui] * V_arr[vi])
                
                U_arr -= lr * grad_U
                V_arr -= lr * grad_V
                W_arr -= lr * grad_W
                
                # Check convergence
                max_res = 0
                for i in range(N):
                    for j in range(M):
                        for k in range(P):
                            ui = 5 * i + j
                            vi = 5 * j + k
                            wi = 2 * k + i
                            val = np.sum(U_arr[ui] * V_arr[vi] * W_arr[wi])
                            max_res = max(max_res, abs(val - 1.0))
                
                if max_res < 1e-8:
                    # Found a valid decomposition!
                    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
        
        if time.time() > deadline:
            break
    
    return best_result


def construct():
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s safety margin
    
    # Try to find a lower-rank decomposition
    try:
        result = _try_search(deadline)
    except Exception:
        result = _schoolbook()
    
    # Verify the result
    if not _verify(result["U"], result["V"], result["W"], trials=100, tol=1e-8):
        return _schoolbook()
    
    return result
# EVOLVE-BLOCK-END
