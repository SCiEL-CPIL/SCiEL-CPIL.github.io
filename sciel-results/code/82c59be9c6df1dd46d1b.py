"""Matrix multiplication 3x4x6: rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

# Theoretical lower bounds
# For 3x4x6: rank >= max(3*4, 4*6, 3*6) = max(12, 24, 18) = 24
# Strassen-like bounds suggest rank might be around 24-30

def _check_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the multiplication tensor T[i,j,k] = delta_{i,k} * A[i,j] * B[j,k]
    # The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this is a valid mult
    # Actually the tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p
    # T_{i,j,k} = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C
    # So T[i,j,k] = 1 for all valid i,j,k (it's the structure tensor)
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # With indexing: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n+i, r] is the (i,k) entry indexed as k*n+i
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx], V[v_idx]) * 0  # placeholder
                # Actually: sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                # This is not a simple dot product, it's a trilinear form
                s = 0.0
                for r in range(rank):
                    s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _check_decomposition_fast(U, V, W, n, m, p, tol=1e-10):
    """Faster check using numpy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                # sum_r U[u_idx,r] * V[v_idx,r] * W[w_idx,r]
                s = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2_base():
    """Return Strassen's 7-term decomposition for 2x2x2."""
    # Strassen's algorithm: 7 multiplications
    # U[i,j,r], V[j,k,r], W[i,k,r] for 2x2x2
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # Strassen:
    # M1 = (A11-A22)(B22-B11)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # r=0: M1 = (A11-A22)(B22-B11)
    # U[0,0]=1, U[3,0]=-1 (A11, A22)
    # V[2,0]=-1, V[3,0]=1 (B22, B11) -- wait, V[j*2+k, r]
    # B11 is j=0,k=0 -> V[0,0]; B22 is j=1,k=1 -> V[3,0]
    # M1 = (A11 - A22)*(B22 - B11)
    # U: A11 -> U[0,0]=1, A22 -> U[3,0]=-1
    # V: B22 -> V[3,0]=1, B11 -> V[0,0]=-1
    # W: C11 -> W[0,0]=1, C22 -> W[3,0]=1
    U[0, 0] = 1; U[3, 0] = -1
    V[0, 0] = -1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C21
    W[3, 1] = -1  # C22
    
    # r=2: M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1  # B12
    V[3, 2] = -1  # B22
    W[1, 2] = 1  # C12
    W[3, 2] = 1  # C22
    
    # r=3: M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1  # B21
    V[0, 3] = -1  # B11
    W[0, 3] = 1  # C11
    W[2, 3] = 1  # C21
    
    # r=4: M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1  # C11
    W[1, 4] = 1  # C12
    
    # r=5: M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C22
    
    # r=6: M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C11
    W[3, 6] = -1  # C22
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Start with schoolbook
    rank_schoolbook = n * m * p  # 72
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Strategy: Use random initialization + optimization
    # The constraint is: for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # We'll try ranks from the lower bound up
    # Lower bound: max(n*m, m*p, n*p) = max(12, 24, 18) = 24
    lower_bound = max(n * m, m * p, n * p)
    
    # Try a few candidate ranks
    candidate_ranks = list(range(lower_bound, min(lower_bound + 20, rank_schoolbook)))
    
    np.random.seed(42)
    
    for rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        max_attempts = 50
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small values
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Use gradient-based optimization
            # Loss: sum over (i,j,k) of (sum_r U[i,j,r]*V[j,k,r]*W[i,k,r] - 1)^2
            lr = 0.01
            num_iters = 200
            
            for it in range(num_iters):
                if time.time() > deadline - 3:
                    break
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                loss = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            
                            # f = sum_r U[u_idx,r] * V[v_idx,r] * W[w_idx,r]
                            f = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                            err = f - 1.0
                            loss += err * err
                            
                            # d/dU[u_idx,r] = 2*err * V[v_idx,r] * W[w_idx,r]
                            dU[u_idx] += 2 * err * V[v_idx] * W[w_idx]
                            dV[v_idx] += 2 * err * U[u_idx] * W[w_idx]
                            dW[w_idx] += 2 * err * U[u_idx] * V[v_idx]
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if loss < 1e-12:
                    break
            
            # Check if we found a valid decomposition
            if loss < 1e-10:
                # Verify
                if _check_decomposition_fast(U, V, W, n, m, p, tol=1e-10):
                    # Convert to list of lists
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    return best_result
