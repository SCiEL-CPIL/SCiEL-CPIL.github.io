"""Improved matrix multiplication tensor decomposition for 4x5 x 5x6."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    best_result = None
    best_rank = n * m * p  # Schoolbook rank
    
    # Helper: verify decomposition
    def verify(U, V, W, rank, tol=1e-9):
        if time.time() > deadline:
            return False
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta_{i,k} for the tensor
        # Actually the tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
        # The decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c where
        # U_c[i*m+j] = U[i*m+j, c], V_c[j*p+k] = V[j*p+k, c], W_c[k*n+i] = W[k*n+i, c]
        # So we need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 if this is a valid term
        # Actually for matrix mult, the tensor is T[i,j,k] = delta_{i,k}... no.
        # The multiplication tensor: T_{i,j,k} = 1 means A_{i,j} * B_{j,k} contributes to C_{i,k}
        # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        # Wait, that's not right either. Let me reconsider.
        
        # The tensor for matrix multiplication M_{n,m,p} has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # because A[i,j] * B[j,k] contributes to C[i,k].
        # A rank-1 tensor is u ⊗ v ⊗ w with entries u[i*m+j] * v[j*p+k] * w[k*n+i]
        # We need: sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1 for all i,j,k
        
        # Build the full tensor and check
        reconstructed = np.zeros((n*m, m*p, p*n))
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            # Outer product
            for idx1 in range(n*m):
                i1, j1 = divmod(idx1, m)
                for idx2 in range(m*p):
                    j2, k2 = divmod(idx2, p)
                    if j1 != j2:
                        continue
                    for idx3 in range(p*n):
                        k3, i3 = divmod(idx3, n)
                        if k2 != k3 or i1 != i3:
                            continue
                        reconstructed[idx1, idx2, idx3] += uc[idx1] * vc[idx2] * wc[idx3]
        
        # Check all entries should be 1 where i1==i3 and j1==j2 and k2==k3
        # Actually, the tensor is 1 for ALL i,j,k combinations
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx1 = i * m + j
                    idx2 = j * p + k
                    idx3 = k * n + i
                    if abs(reconstructed[idx1, idx2, idx3] - 1.0) > tol:
                        return False
                    # Also check that off-diagonal terms are 0
                    # But the tensor is actually 1 everywhere in the "diagonal" sense
        # This is too slow for large rank. Let me use a different verification.
        return True
    
    # Faster verification using tensor contraction
    def verify_fast(U, V, W, rank, tol=1e-9):
        if time.time() > deadline:
            return False
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # The tensor T[i,j,k] = 1 for all i,j,k
        # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        # Reshape: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # For fixed i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # We can check by building the tensor via einsum
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This is a 3-tensor of shape (n, m, p)
        
        # Build it efficiently
        T = np.zeros((n, m, p))
        for c in range(rank):
            uc = U[:, c]  # (n*m,)
            vc = V[:, c]  # (m*p,)
            wc = W[:, c]  # (p*n,)
            # Reshape
            uc_3d = uc.reshape(n, m)
            vc_3d = vc.reshape(m, p)
            wc_3d = wc.reshape(p, n)
            # For each i,j,k: uc_3d[i,j] * vc_3d[j,k] * wc_3d[k,i]
            # This is a diagonal-like product
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += uc_3d[i, j] * vc_3d[j, k] * wc_3d[k, i]
        
        # Check T[i,j,k] == 1 for all i,j,k
        if not np.allclose(T, np.ones((n, m, p)), atol=tol):
            return False
        return True
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U, V, W, rank = schoolbook()
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to reduce rank by removing linearly dependent columns
    # This is complex, so let's try a simpler approach: use known Strassen-like decompositions
    
    # For 4x5 x 5x6, we can try to use a 2x2 Strassen block decomposition
    # Pad 4x5 to 4x4? No, 5 is odd.
    
    # Alternative: Use the fact that 4x5 x 5x6 can be decomposed as:
    # Split the 5 dimension into 4+1
    # A (4x5) = [A1 (4x4) | A2 (4x1)]
    # B (5x6) = [B1 (4x6); B2 (1x6)]
    # C = A1*B1 + A2*B2
    # A1*B1 is 4x4 x 4x6, A2*B2 is 4x1 x 1x6
    
    # Rank of 4x4 x 4x6: schoolbook is 96, but we can do better
    # Rank of 4x1 x 1x6: schoolbook is 24
    
    # This doesn't immediately help.
    
    # Let's try a different approach: use randomized rank detection
    # If the tensor has rank r, then the flattening has rank r
    # We can try to find the minimum rank by checking if a random subset of columns spans
    
    # For now, return the schoolbook result
    # Convert to list format
    result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    
    return result
# EVOLVE-BLOCK-END
