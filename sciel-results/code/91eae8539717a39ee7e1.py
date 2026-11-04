import time
import random
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank = 42
    
    # Build the rank-42 Strassen block decomposition
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(i_start, j_start, k_start, offset):
        col = offset
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 1
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] -= 1
        
        col = offset + 2
        U[(i_start+0)*m + (j_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 3
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] -= 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        
        col = offset + 4
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        
        col = offset + 5
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+0)][col] -= 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 6
        U[(i_start+0)*m + (j_start+1)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] -= 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1

    add_strassen_block(0, 0, 0, 0)
    add_strassen_block(0, 0, 2, 7)
    add_strassen_block(0, 0, 4, 14)
    add_strassen_block(0, 2, 0, 21)
    add_strassen_block(0, 2, 2, 28)
    add_strassen_block(0, 2, 4, 35)
    
    # Verify the decomposition is correct
    def verify(U, V, W, n, m, p, rank):
        """Check that the tensor decomposition is exact."""
        # The matrix multiplication tensor T[i,j,k] = 1 if the (i,k) element of 
        # A*B uses A[i,j]*B[j,k]
        # For decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # (with W indexed as W[k*n+i, r] per the trace convention)
        
        # Build expected tensor
        # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # because (AB)[i,k] = sum_j A[i,j] B[j,k]
        
        # Check using random matrices
        np.random.seed(42)
        A = np.random.randint(-5, 6, size=(n, m))
        B = np.random.randint(-5, 6, size=(m, p))
        AB = A @ B
        
        # Compute using decomposition
        # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
        # Actually: C[i,k] = sum_r U[i,j,r]*A[i,j] * V[j,k,r]*B[j,k] * W[k*n+i,r]
        # More precisely: C[i,k] = sum_j sum_k' sum_r U[i*m+j, r] * A[i,j] * V[j*p+k', r] * B[j,k'] * W[k'*n+i, r]
        # But this isn't quite right. Let me think again.
        
        # The tensor for matrix mult: T_{ijk} where i indexes rows of A (and C), 
        # j indexes cols of A / rows of B, k indexes cols of B (and C)
        # T_{ijk} = 1 (the multiplication tensor has all 1s)
        # Decomposition: T_{ijk} = sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}
        # where W is stored as W[k*n + i, r]
        
        # So C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k]) ... 
        # No, that's not right either. Let me be more careful.
        
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k] / (U*V*W terms)
        
        # Actually the decomposition says:
        # sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j,r] A[i,j]) ... no
        
        # The correct identity is:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # In tensor form with T[i,j,k] = 1:
        # C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k] = sum_j A[i,j] B[j,k]
        # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # So: C[i,k] = sum_j sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] A[i,j] B[j,k]
        
        # This is a bilinear form. Let me just verify numerically with the tensor.
        
        # Build the decomposition tensor
        decomp = np.zeros((n, m, p), dtype=float)
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        decomp[i, j, k] += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        
        # Should equal all 1s
        if not np.allclose(decomp, 1.0, atol=1e-10):
            return False
        return True
    
    if not verify(U, V, W, n, m, p, rank):
        # Fallback: schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
    
    return {"U": U, "V": V, "W": W}
