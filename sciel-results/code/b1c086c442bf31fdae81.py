"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    deadline = time.time() + 140  # 10s safety margin
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    def validate(U, V, W):
        """Check if U,V,W exactly decompose the matrix multiplication tensor."""
        # The tensor identity: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # And for all (i,j,k) where the product should be 0 (which is never in this case for valid (i,j,k)), it should be 0
        # Actually, the matrix multiplication tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all valid i,j,k in the sum, and the decomposition must satisfy:
        # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        r = U_arr.shape[1]
        if V_arr.shape[1] != r or W_arr.shape[1] != r:
            return False
        
        # Check all (i,j,k) combinations
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i*m + j, :]
                    v_val = V_arr[j*p + k, :]
                    w_val = W_arr[k*n + i, :]
                    s = np.dot(np.dot(u_val, v_val), w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def validate_fast(U, V, W):
        """Faster validation using matrix operations."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        r = U_arr.shape[1]
        if V_arr.shape[1] != r or W_arr.shape[1] != r:
            return False
        
        # Build the tensor and check
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We need T[i,j,k] = 1 for all i,j,k
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j,:], V_arr[j*p+k,:] * W_arr[k*n+i,:])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find lower-rank decompositions using random projections
    # Strategy: combine schoolbook terms randomly and check if rank can be reduced
    
    np.random.seed(42)
    
    # Try various random combinations
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Create a random linear combination of schoolbook factors
        # This may create cancellations that reduce effective rank
        
        # Method 1: Random mixing of columns
        if attempt % 2 == 0:
            # Try to find a lower-rank equivalent by mixing
            r = rank_sb
            # Generate random orthogonal matrix
            Q = np.random.randn(r, r)
            Q, _ = np.linalg.qr(Q)
            
            U_new = U_sb @ Q
            V_new = V_sb @ Q
            W_new = W_sb @ Q
            
            # Check if this still validates (it should, since it's an orthogonal transformation)
            if validate_fast(U_new, V_new, W_new):
                # Try to compress by removing zero or near-zero columns
                U_norms = np.linalg.norm(U_new, axis=0)
                V_norms = np.linalg.norm(V_new, axis=0)
                W_norms = np.linalg.norm(W_new, axis=0)
                
                # Keep only columns that are not negligible
                mask = (U_norms > 1e-10) & (V_norms > 1e-10) & (W_norms > 1e-10)
                if mask.sum() < best_rank:
                    U_c = U_new[:, mask]
                    V_c = V_new[:, mask]
                    W_c = W_new[:, mask]
                    if validate_fast(U_c, V_c, W_c):
                        best_rank = mask.sum()
                        best = {"U": U_c.tolist(), "V": V_c.tolist(), "W": W_c.tolist()}
        
        # Method 2: Try structured reductions (Strassen-style)
        else:
            # For 4x4x9, try to use block structure
            # Split into 2x2 blocks for A and B, compute 4x9 result
            # A is 4x4, B is 4x9, C = A*B is 4x9
            
            # Strassen for 2x2 uses 7 multiplications instead of 8
            # For 4x4, we can use 2x2 blocks and apply Strassen recursively
            
            # Let's try a direct Strassen-based approach for 4x4 x 4x9
            # A = [[A11, A12], [A21, A22]] where each is 2x2
            # B = [[B11, B12, B13, B14, B15, B16, B17, B18, B19]] -- wait, B is 4x9
            # B = [[B1, B2, B3, B4, B5, B6, B7, B8, B9]] where each Bj is 2x9? No...
            # B is 4x9, so split rows: B = [[B11, B12], [B21, B22]] where B11,B12 are 2x9
            # Actually, for block matrix multiplication with different block sizes:
            # A is 4x4 = [[A11, A12], [A21, A22]] with 2x2 blocks
            # B is 4x9 = [[B1], [B2]] where B1, B2 are 2x9
            # C = A*B = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]] which is 4x9
            
            # Strassen for the 2x2 block multiplication (A is 2x2 blocks, B is 2x1 blocks of 2x9):
            # This is a bit tricky since B blocks are 2x9 not square.
            
            # Alternative: just try random search with gradient-free methods
            pass
    
    # Final: try to verify and return best found
    return best
# EVOLVE-BLOCK-END
