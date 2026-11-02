"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W form a valid rank-r decomposition of the matrix multiplication tensor.
    Condition: for all i, k, j1, j2:
    sum_c U[i*m+j1, c] * V[j2*p+k, c] * W[k*n+i, c] = delta_{j1, j2}
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    val = np.dot(U[i*m+j1, :] * V[j2*p+k, :], W[k*n+i, :])
                    expected = 1.0 if j1 == j2 else 0.0
                    if abs(val - expected) > tol:
                        return False
    return True

def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """
    More thorough verification: check that the trilinear form matches the multiplication tensor.
    The multiplication tensor T satisfies:
    sum_{i,j,k} C[i,k] A[i,j] B[j,k] = sum_c (sum_{i,k} C[i,k] W[k*n+i, c])
                                      (sum_{i,j} A[i,j] U[i*m+j, c])
                                      (sum_{j,k} B[j,k] V[j*p+k, c])
    
    This is equivalent to: for all i, j, k:
    sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    AND for any "inconsistent" triple (i', j', k') where the indices don't form a valid
    multiplication path, the sum should be 0. But actually the tensor is defined such that
    T[i,j,k] = 1 for ALL valid (i,j,k) in the range, because the multiplication tensor
    represents the full trilinear form of matrix multiplication.
    
    Wait - let me re-examine. The standard matrix multiplication tensor M_{n,m,p} has
    M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p). This is because:
    C[i,k] = sum_j A[i,j] B[j,k]
    The trilinear form is sum_{i,j,k} C[i,k] A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k] C[i,k]
    So the tensor entry T[i,j,k] = 1 for all valid indices.
    
    BUT the verification in _verify_decomposition checks a different condition involving
    delta_{j1,j2}. Let me re-read the task contract.
    
    The task says: "exactly decompose the matrix multiplication tensor"
    With U of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank).
    
    The standard decomposition is:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For the matrix multiplication tensor, T[i,j,k] = 1 for all valid (i,j,k).
    
    But wait, that would mean the schoolbook decomposition with U[i*m+j, c]=1, V[j*p+k, c]=1,
    W[k*n+i, c]=1 for c corresponding to (i,j,k) gives:
    sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 (only the c matching (i,j,k) contributes)
    
    For other (i',j',k'), the sum is 0 because no single c has all three non-zero.
    
    So actually T[i,j,k] = 1 for all valid (i,j,k) and the schoolbook works.
    
    But the _verify_decomposition function checks a DIFFERENT condition. Let me look again...
    
    It checks: sum_c U[i*m+j1, c] * V[j2*p+k, c] * W[k*n+i, c] = delta_{j1,j2}
    
    This is a different tensor! This is the tensor for computing C[i,k] = sum_j A[i,j] B[j,k]
    where we fix i and k and vary j1 (from A) and j2 (from B). The condition delta_{j1,j2}
    means that only when j1=j2 do we get a contribution.
    
    Hmm, this doesn't match the standard matrix multiplication tensor. Let me think more carefully.
    
    Actually, I think the issue is in how the tensor is indexed. Let me re-derive.
    
    The matrix multiplication C = AB where A is n x m and B is m x p.
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The trilinear form is: sum_{i,j,k} C[i,k] A[i,j] B[j,k]
    
    We want to write this as: sum_c (sum_{i,k} C[i,k] W[k*n+i, c]) (sum_{i,j} A[i,j] U[i*m+j, c]) (sum_{j,k} B[j,k] V[j*p+k, c])
    
    Expanding: sum_c sum_{i,k} sum_{i',j} sum_{j',k'} W[k*n+i, c] U[i'*m+j, c] V[j'*p+k', c] C[i,k] A[i',j] B[j',k']
    
    For this to equal sum_{i,j,k} C[i,k] A[i,j] B[j,k], we need:
    sum_c W[k*n+i, c] U[i'*m+j, c] V[j'*p+k', c] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    So the condition is: for all i, i', j, j', k, k':
    sum_c U[i'*m+j, c] V[j'*p+k', c] W[k*n+i, c] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    This is the correct condition! The _verify_decomposition function is checking a WRONG condition.
    It's checking sum_c U[i*m+j1, c] V[j2*p+k, c] W[k*n+i, c] = delta_{j1,j2}, which is not the
    right condition for matrix multiplication.
    
    Let me fix the verification to check the correct condition.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for i2 in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for k2 in range(p):
                            val = np.dot(U[i2*m+j, :] * V[j2*p+k2, :], W[k*n+i, :])
                            expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                            if abs(val - expected) > tol:
                                return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125.0  # Safety margin before 150s kill
    
    # Schoolbook decomposition (rank = n*m*p = 54)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, c] = 1.0
                V_sb[j * p + k, c] = 1.0
                W_sb[k * n + i, c] = 1.0
                c += 1
    
    # Verify schoolbook with correct condition
    assert _verify_full(U_sb, V_sb, W_sb, n, m, p), "Schoolbook verification failed"
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # Try to find decompositions with rank < 54
    
    # Strategy: Exploit the structure p=9=3*3
    # We can view the 3x9 multiplication as related to 3x3 multiplication
    # using a block structure
    
    # First, let's try to find a rank-9 decomposition for 2x3x3
    # If we find one, we can potentially use it to build a lower-rank 2x3x9
    
    found_233 = False
    U_233 = None
    V_233 = None
    W_233 = None
    r_233 = None
    
    np.random.seed(42)
    
    for target_r in range(9, 19):
        if time.time() > deadline - 30:
            break
        for trial in range(100):
            if time.time() > deadline - 25:
                break
            
            # Try structured initialization
            U = np.zeros((6, target_r))
            V = np.zeros((9, target_r))
            W = np.zeros((6, target_r))
            
            # Use a pattern based on indices
            for c in range(target_r):
                # Try different patterns
                if trial % 3 == 0:
                    i = (c * 7) % 2
                    j = (c * 13) % 3
                    k = (c * 19) % 3
                elif trial % 3 == 1:
                    i = c % 2
                    j = (c // 2) % 3
                    k = (c // 6) % 3
                else:
                    i = np.random.randint(0, 2)
                    j = np.random.randint(0, 3)
                    k = np.random.randint(0, 3)
                
                U[i*3+j, c] = 1.0
                V[j*3+k, c] = 1.0
                W[k*2+i, c] = 1.0
            
            if _verify_full(U, V, W, 2, 3, 3):
                U_233, V_233, W_233 = U, V, W
                r_233 = target_r
                found_233 = True
                break
        if found_233:
            break
    
    # If we found a 2x3x3 decomposition, try to use it for 2x3x9
    if found_233 and r_233 is not None:
        # Try block diagonal: 3 copies of the 2x3x3 decomposition
        total_rank = 3 * r_233
        
        if total_rank < best_rank:
            U_full = np.zeros((6, total_rank))
            V_full = np.zeros((27, total_rank))
            W_full = np.zeros((18, total_rank))
            
            # For k = 3*k1 + k2, assign to block k2
            for k2 in range(3):
                offset = k2 * r_233
                U_full[:, offset:offset+r_233] = U_233
                
                for j in range(3):
                    for k1 in range(3):
                        k = 3*k1 + k2
                        V_full[j*9+k, offset:offset+r_233] = V_233[j*3+k1, :]
                
                for k1 in range(3):
                    for i in range(2):
                        k = 3*k1 + k2
                        W_full[k*2+i, offset:offset+r_233] = W_233[k1*2+i, :]
            
            if _verify_full(U_full, V_full, W_full, n, m, p):
                best_U, best_V, best_W = U_full, V_full, W_full
                best_rank = total_rank
    
    # Try direct search for lower ranks using structured patterns
    for target_rank in range(max(27, best_rank - 15), min(best_rank, 54)):
        if time.time() > deadline - 20:
            break
        
        found = False
        for trial in range(50):
            if time.time() > deadline - 15:
                break
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Use a hash-like pattern to assign positions
            for c in range(target_rank):
                h = (c * 7 + trial * 13) % (n * m * p)
                i = h // (m * p)
                j = (h // p) % m
                k = h % p
                
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
            
            if _verify_full(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
        
        if found:
            continue
    
    # Convert to lists for the return interface
    U_out = best_U.tolist() if isinstance(best_U, np.ndarray) else best_U
    V_out = best_V.tolist() if isinstance(best_V, np.ndarray) else best_V
    W_out = best_W.tolist() if isinstance(best_W, np.ndarray) else best_W
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
