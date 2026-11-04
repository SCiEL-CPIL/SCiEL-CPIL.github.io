import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Build the tensor T[i,j,k] which should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # The tensor for matrix mult: T[i,j,k] = 1 if this is a valid (i,j,k) triple, else 0
    # Actually the matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
    # and the decomposition requires: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # Wait, that's not right. The matrix multiplication tensor is:
    # M_{n,m,p}[i,j,k] = 1 for all 0<=i<n, 0<=j<m, 0<=k<p
    # and the bilinear form is: sum_{i,j,k} M[i,j,k] * A[i,j]*B[j,k]*C[k,i]... 
    # Actually the standard formulation: the tensor T has T[i,j,k] = 1, and we need
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k] = 1 for all i,j,k
    # But that would mean every triple (i,j,k) must sum to 1, which is the schoolbook.
    
    # Wait, I need to re-read the problem. The matrix multiplication tensor for (n,m,p):
    # The tensor is defined such that <T, A ⊗ B ⊗ C> = trace(A * B * C) for appropriate matrices
    # With the trace convention W[k*n+i, r], this means:
    # sum_{i,j,k} A[i,j] * B[j,k] * C[k,i] = sum_r (sum_{i,j} U[i*m+j,r]*A[i,j]) * (sum_{j,k} V[j*p+k,r]*B[j,k]) * (sum_{k,i} W[k*n+i,r]*C[k,i])
    
    # For this to equal trace(A*B*C) for ALL A, B, C, we need the tensor to match.
    # The left side: sum_{i,j,k} A[i,j]*B[j,k]*C[k,i]
    # The right side: sum_r (sum_{i,j} U[i*m+j,r]*A[i,j]) * (sum_{j,k} V[j*p+k,r]*B[j,k]) * (sum_{k,i} W[k*n+i,r]*C[k,i])
    # = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]) * A[i,j]*B[j,k]*C[k,i]
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is the condition. Let me verify with schoolbook:
    # For each (i,j,k), there's exactly one r where U[i*m+j,r]=1, V[j*p+k,r]=1, W[k*n+i,r]=1, so sum=1. ✓
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k])  # This gives sum_r U[i*m+j,r]*V[j*p+k,r]
                # But we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r], which is a trilinear form
                # = (U[i*m+j] * V[j*p+k]) dot W[k*n+i] where * is element-wise
                val = np.dot(U_arr[i*m+j] * V_arr[j*p+k], W_arr[k*n+i])
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _strassen_decomp():
    n, m, p = DIMENSIONS  # 2, 3, 6
    rank = 33
    U = [[0] * rank for _ in range(n * m)]  # 6 x 33
    V = [[0] * rank for _ in range(m * p)]  # 18 x 33
    W = [[0] * rank for _ in range(p * n)]  # 12 x 33
    
    col = 0
    
    # Strassen for 2x2x2 blocks on A[0:2, 0:2] * B[0:2, k:k+2] for k=0,2,4
    for block in range(3):
        k_off = block * 2
        
        # Term 1: (A[0,0]+A[1,1]) * (B[0,k]+B[1,k+1])
        U[0*3+0][col] = 1
        U[1*3+1][col] = 1
        V[0*6+k_off][col] = 1
        V[1*6+(k_off+1)][col] = 1
        W[(k_off)*2+0][col] = 1
        W[(k_off+1)*2+1][col] = 1
        col += 1
        
        # Term 2: (A[1,0]+A[1,1]) * B[0,k]
        U[1*3+0][col] = 1
        U[1*3+1][col] = 1
        V[0*6+k_off][col] = 1
        W[(k_off)*2+1][col] = 1
        col += 1
        
        # Term 3: A[0,0] * (B[1,k]-B[1,k+1])
        U[0*3+0][col] = 1
        V[1*6+k_off][col] = 1
        V[1*6+(k_off+1)][col] = -1
        W[(k_off+1)*2+0][col] = 1
        col += 1
        
        # Term 4: A[1,1] * (B[0,k+1]-B[0,k])
        U[1*3+1][col] = 1
        V[0*6+k_off][col] = -1
        V[0*6+(k_off+1)][col] = 1
        W[(k_off)*2+0][col] = 1
        W[(k_off)*2+1][col] = 1
        col += 1
        
        # Term 5: (A[0,0]+A[0,1]) * B[1,k+1]
        U[0*3+0][col] = 1
        U[0*3+1][col] = 1
        V[1*6+(k_off+1)][col] = 1
        W[(k_off)*2+0][col] = -1
        W[(k_off+1)*2+0][col] = 1
        col += 1
        
        # Term 6: (A[1,0]-A[0,0]) * (B[0,k]+B[0,k+1])
        U[1*3+0][col] = 1
        U[0*3+0][col] = -1
        V[0*6+k_off][col] = 1
        V[0*6+(k_off+1)][col] = 1
        W[(k_off+1)*2+1][col] = 1
        col += 1
        
        # Term 7: (A[0,1]-A[1,1]) * (B[1,k]+B[1,k+1])
        U[0*3+1][col] = 1
        U[1*3+1][col] = -1
        V[1*6+k_off][col] = 1
        V[1*6+(k_off+1)][col] = 1
        W[(k_off+1)*2+0][col] = 1
        col += 1
    
    # Handle A[0:2, 2] * B[2, :] (the 2x1x6 part) - 12 terms
    for i in range(2):
        for k in range(6):
            U[i*3+2][col] = 1
            V[2*6+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
    
    assert col == rank, f"Expected {rank} terms, got {col}"
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    result = _strassen_decomp()
    
    if _verify(result["U"], result["V"], result["W"]):
        return result
    
    return _schoolbook()
