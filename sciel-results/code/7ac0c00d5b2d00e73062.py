import time
import sys
import random
import numpy as np
import scipy.optimize
from fractions import Fraction

# Dimensions for the matrix multiplication: (n, m, p)
# A is n x m, B is m x p, C is n x p
# Here n=2, m=3, p=3
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    We first try to find a decomposition with rank 16 using numerical search.
    If not found within time budget, return the schoolbook rank-18 or known rank-17 solution.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        rank = len(U[0])
        # Convert to numpy for speed
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Compute the tensor
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We check all (i, j, k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = np.sum(U_np[idx_u] * V_np[idx_v] * W_np[idx_w])
                    if abs(s - 1.0) > 1e-9:
                        return False
        
        # Check zeros
        # Total triples 6*9*6 = 324
        # Valid ones are 18.
        # We need to check that all other 306 are 0.
        # To be efficient, we can compute the full tensor and check against the target.
        # But the loop above only checks the 18 ones.
        # We must check the zeros.
        
        # Re-implement verify to check all entries
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    T_target[idx_u, idx_v, idx_w] = 1.0
        
        T_calc = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U_np[:, r], V_np[:, r], W_np[:, r])
            
        if not np.allclose(T_calc, T_target, atol=1e-9):
            return False
            
        return True

    def try_rank16():
        """
        Attempt to find a rank-16 decomposition using numerical optimization.
        """
        rank = 16
        n, m, p = DIMENSIONS
        dim_u = n * m
        dim_v = m * p
        dim_w = p * n
        
        # Target tensor
        T_target = np.zeros((dim_u, dim_v, dim_w))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    T_target[idx_u, idx_v, idx_w] = 1.0
        
        # Flatten target
        T_flat = T_target.flatten()
        
        def compute_flat(U, V, W):
            # U: (dim_u, rank), V: (dim_v, rank), W: (dim_w, rank)
            T = np.zeros((dim_u, dim_v, dim_w))
            for r in range(rank):
                T += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            return T.flatten()
        
        def objective(x):
            U = x[:dim_u*rank].reshape(dim_u, rank)
            V = x[dim_u*rank:dim_u*rank + dim_v*rank].reshape(dim_v, rank)
            W = x[dim_u*rank + dim_v*rank:].reshape(dim_w, rank)
            T_calc = compute_flat(U, V, W)
            return T_calc - T_flat
        
        # Random initialization
        # We try a few seeds
        for seed in range(5):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(seed)
            # Initialize with small values
            x0 = np.random.normal(0, 1, dim_u*rank + dim_v*rank + dim_w*rank)
            
            try:
                # Use least squares
                result = scipy.optimize.least_squares(objective, x0, method='lm', max_nfev=1000)
                x_sol = result.x
                
                U = x_sol[:dim_u*rank].reshape(dim_u, rank)
                V = x_sol[dim_u*rank:dim_u*rank + dim_v*rank].reshape(dim_v, rank)
                W = x_sol[dim_u*rank + dim_v*rank:].reshape(dim_w, rank)
                
                # Round to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Convert to lists of floats for the interface
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
            except Exception as e:
                pass
                
        return None

    def schoolbook():
        """Return the schoolbook rank-18 decomposition."""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    U[idx_u][col_idx] = 1.0
                    V[idx_v][col_idx] = 1.0
                    W[idx_w][col_idx] = 1.0
                    col_idx += 1
        
        return {"U": U, "V": V, "W": W}
    
    def rank17_decomposition():
        """
        Construct a rank-17 decomposition for 2x3x3 matrix multiplication.
        """
        rank = 17
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        U[0][0] = 1; U[4][0] = 1
        V[0][0] = 1; V[4][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        U[3][1] = 1; U[4][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        U[0][2] = 1
        V[1][2] = 1; V[4][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        U[4][3] = 1
        V[3][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        U[0][4] = 1; U[1][4] = 1
        V[4][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        U[3][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        U[1][6] = 1; U[4][6] = -1
        V[3][6] = 1; V[4][6] = 1
        W[0][6] = 1
        
        # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
        U[2][7] = 1
        V[6][7] = 1
        W[0][7] = 1
        
        U[2][8] = 1
        V[7][8] = 1
        W[2][8] = 1
        
        U[5][9] = 1
        V[6][9] = 1
        W[1][9] = 1
        
        U[5][10] = 1
        V[7][10] = 1
        W[3][10] = 1
        
        # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
        U[0][11] = 1
        V[2][11] = 1
        W[4][11] = 1
        
        U[1][12] = 1
        V[5][12] = 1
        W[4][12] = 1
        
        U[2][13] = 1
        V[8][13] = 1
        W[4][13] = 1
        
        U[3][14] = 1
        V[2][14] = 1
        W[5][14] = 1
        
        U[4][15] = 1
        V[5][15] = 1
        W[5][15] = 1
        
        U[5][16] = 1
        V[8][16] = 1
        W[5][16] = 1
        
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # First, try the known rank-17 decomposition
    if time.time() < deadline:
        r17 = rank17_decomposition()
        if verify(r17["U"], r17["V"], r17["W"]):
            best_result = r17
            best_rank = 17
    
    # Attempt rank 16 search if time permits
    if best_rank > 16 and time.time() < deadline - 20:
        try:
            r16 = try_rank16()
            if r16:
                return r16
        except Exception:
            pass

    if best_rank < n * m * p:
        return best_result
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
