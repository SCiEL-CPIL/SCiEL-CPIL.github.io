import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 5]

def _strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    T[i,j,k] = 1 iff j matches between input indices.
    a=(i,j) in [0,4), b=(j,k) in [0,4), c=(k,i) in [0,4)
    """
    n, m, p = 2, 2, 2
    nm, mp, np_ = 4, 4, 4
    # Strassen's 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U: maps A[i,j] to products
    # V: maps B[j,k] to products  
    # W: maps C[k,i] to products
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1 # B11
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # B00
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1 # C00
    W[1, 4] = 1  # C01
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[3, 5] = 1  # C11
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00
    
    return U, V, W

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the tensor decomposition is exact."""
    nm = n * m
    mp = m * p
    np_ = n * p
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build target tensor
    T_target = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Reconstruct
    T_recon = np.zeros((nm, mp, np_))
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    if np.linalg.norm(T_recon - T_target) > tol:
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    np_ = n * p
    
    # Strategy: decompose 2x4x5 using Strassen 2x2x2 blocks
    # 4 = 2+2, 5 = 2+2+1
    # (2x4)(4x5) = (2x2)(2x5) + (2x2)(2x5) + (2x2)(1x5)
    # Each 2x2x2 block uses rank 7 (Strassen)
    # 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1
    # 2x2x1: (2x2)(2x1)->(2x1), rank = 4 (schoolbook) or less?
    # Actually 2x2x1: T[i,j,k,l] where inner dim is 2, output is 2x1
    # T[i,j,0] = 1 iff j matches. For (2x2)(2x1): a=(i,j), b=(j,0), c=(0,i)
    # T[i*2+j, j, i] = 1 for i,j in {0,1}. That's 4 entries.
    # Rank: each (i,j) pair gives one term, so rank 4? Or can we do better?
    # T[i*2+j, j, i] = 1. For fixed i, we have T[i*2, 0, i]=1 and T[i*2+1, 1, i]=1.
    # This is like two separate 1x1x1 tensors per i, so rank 4 total.
    
    # Let's use a simpler approach: direct Strassen embedding
    # Split m=4 into two 2s, p=5 into two 2s and one 1
    # Block (0,0): 2x2x2 -> rank 7
    # Block (0,1): 2x2x2 -> rank 7  
    # Block (0,2): 2x2x1 -> rank 4
    # Block (1,0): 2x2x2 -> rank 7
    # Block (1,1): 2x2x2 -> rank 7
    # Block (1,2): 2x2x1 -> rank 4
    # Total: 7+7+4+7+7+4 = 36
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    for block_j in range(2):  # which 2x2 block of the 4-dim inner
        for block_k in range(3):  # which block of the 5-dim output (2,2,1)
            if block_k < 2:
                # 2x2x2 block
                Ub, Vb, Wb = _strassen_2x2x2()
                # Ub: 4x7, maps A[i, block_j*2 + j'] 
                # Vb: 4x7, maps B[block_j*2 + j', block_k*2 + k']
                # Wb: 4x7, maps C[block_k*2 + k', i]
                
                # Create full-size matrices
                U_full = np.zeros((nm, 7))
                V_full = np.zeros((mp, 7))
                W_full = np.zeros((np_, 7))
                
                for i in range(2):
                    for j in range(2):
                        U_full[i*m + block_j*2 + j, :] = Ub[i*2 + j, :]
                
                for j in range(2):
                    for k in range(2):
                        V_full[(block_j*2 + j)*p + block_k*2 + k, :] = Vb[j*2 + k, :]
                
                for k in range(2):
                    for i in range(2):
                        W_full[(block_k*2 + k)*n + i, :] = Wb[k*2 + i, :]
                
                U_blocks.append(U_full)
                V_blocks.append(V_full)
                W_blocks.append(W_full)
            else:
                # 2x2x1 block: (2x2)(2x1)->(2x1)
                # Inner dim 2, output 2x1
                # T[i,j,k] where a=(i,j) in 2x2, b=(j,0) in 2x1, c=(0,i) in 2x1
                # T[i*2+j, j, i] = 1 for i,j in {0,1}
                # Rank 4: one term per (i,j)
                U_full = np.zeros((nm, 4))
                V_full = np.zeros((mp, 4))
                W_full = np.zeros((np_, 4))
                
                for idx, (i, j) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
                    U_full[i*m + block_j*2 + j, idx] = 1
                    V_full[(block_j*2 + j)*p + block_k*2, idx] = 1  # block_k*2+0
                    W_full[(block_k*2)*n + i, idx] = 1  # (block_k*2+0)*n + i
                    
                U_blocks.append(U_full)
                V_blocks.append(V_full)
                W_blocks.append(W_full)
    
    U = np.hstack(U_blocks)
    V = np.hstack(V_blocks)
    W = np.hstack(W_blocks)
    
    rank = U.shape[1]
    
    # Verify
    if _check_decomposition(U, V, W, n, m, p):
        # Snap to half-integers
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        if _check_decomposition(U_s, V_s, W_s, n, m, p):
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        else:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((nm, rank_sb))
        V_sb = np.zeros((mp, rank_sb))
        W_sb = np.zeros((np_, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
