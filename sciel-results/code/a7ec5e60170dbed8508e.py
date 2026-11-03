"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
from fractions import Fraction

def _validate(U, V, W, n=4, m=4, p=5, tol=1e-9):
    """Check that U,V,W form a valid tensor decomposition of matmul(n,m,p)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
    if not np.allclose(M_decomp, 1.0, atol=tol):
        return False
    return True

def _construct_schoolbook(n=4, m=4, p=5):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_3x3x3_terms():
    """
    Return the 7 rank-1 tensors for 3x3x3 matrix multiplication.
    Each term is (u, v, w) where:
      u: length 9, indexed [i*3+j] for i,j in 0..2
      v: length 9, indexed [j*3+k] for j,k in 0..2
      w: length 9, indexed [k*3+i] for k,i in 0..2
    
    The decomposition satisfies: for all i,k,j in 0..2,
      sum_t u_t[i*3+j] * v_t[j*3+k] * w_t[k*3+i] = 1
    """
    terms = []
    
    # Term 1: (A00+A11)(B00+B11)
    u1 = [0]*9; u1[0]=1; u1[4]=1   # A[0,0], A[1,1]
    v1 = [0]*9; v1[0]=1; v1[4]=1   # B[0,0], B[1,1]
    w1 = [0]*9; w1[0]=1; w1[4]=1   # C[0,0], C[1,1]
    terms.append((u1, v1, w1))
    
    # Term 2: (A10+A11)(B00)
    u2 = [0]*9; u2[3]=1; u2[4]=1   # A[1,0], A[1,1]
    v2 = [0]*9; v2[0]=1            # B[0,0]
    w2 = [0]*9; w2[0]=1            # C[0,0] (k=0,i=0)
    terms.append((u2, v2, w2))
    
    # Term 3: A00*(B10-B11)
    u3 = [0]*9; u3[0]=1            # A[0,0]
    v3 = [0]*9; v3[3]=1; v3[4]=-1  # B[1,0], -B[1,1]
    w3 = [0]*9; w3[3]=1            # C[1,0] (k=1,i=0)
    terms.append((u3, v3, w3))
    
    # Term 4: A11*(B00-B01)
    u4 = [0]*9; u4[4]=1            # A[1,1]
    v4 = [0]*9; v4[0]=1; v4[1]=-1  # B[0,0], -B[0,1]
    w4 = [0]*9; w4[0]=1; w4[4]=1   # C[0,0], C[1,1]
    terms.append((u4, v4, w4))
    
    # Term 5: (A00+A01)*B11
    u5 = [0]*9; u5[0]=1; u5[1]=1   # A[0,0], A[0,1]
    v5 = [0]*9; v5[4]=1            # B[1,1]
    w5 = [0]*9; w5[3]=1; w5[4]=1   # C[1,0], C[1,1]
    terms.append((u5, v5, w5))
    
    # Term 6: (A10-A00)*(B00+B01)
    u6 = [0]*9; u6[3]=1; u6[0]=-1  # A[1,0], -A[0,0]
    v6 = [0]*9; v6[0]=1; v6[1]=1   # B[0,0], B[0,1]
    w6 = [0]*9; w6[0]=1; w6[4]=1   # C[0,0], C[1,1]
    terms.append((u6, v6, w6))
    
    # Term 7: (A01-A11)*(B10+B11)
    u7 = [0]*9; u7[1]=1; u7[4]=-1  # A[0,1], -A[1,1]
    v7 = [0]*9; v7[3]=1; v7[4]=1   # B[1,0], B[1,1]
    w7 = [0]*9; w7[0]=1; w7[4]=1   # C[0,0], C[1,1]
    terms.append((u7, v7, w7))
    
    return terms

def _construct_strassen_embedded(n=4, m=4, p=5):
    """
    Construct a decomposition using Strassen for the 3x3x3 sub-tensor
    and schoolbook for the boundary.
    
    The 3x3x3 sub-tensor covers i in 0..2, k in 0..2, j in 0..2.
    Boundary entries are handled with schoolbook terms.
    """
    # Get the 7 Strassen terms for the 3x3x3 sub-tensor
    strassen_terms = _strassen_3x3x3_terms()
    
    # Build U, V, W matrices
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Start with 7 columns for Strassen
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the Strassen terms
    for t, (u, v, w) in enumerate(strassen_terms):
        for i in range(3):
            for j in range(3):
                U[i * m + j][t] = u[i * 3 + j]
        for j in range(3):
            for k in range(3):
                V[j * p + k][t] = v[j * 3 + k]
        for k in range(3):
            for i in range(3):
                W[k * n + i][t] = w[k * 3 + i]
    
    # Now add schoolbook terms for the boundary
    # Boundary: any (i,k,j) where i>=3 or k>=3 or j>=3
    # For each such triple, add a schoolbook term
    boundary_triples = []
    for i in range(n):
        for k in range(p):
            for j in range(m):
                if i >= 3 or k >= 3 or j >= 3:
                    boundary_triples.append((i, k, j))
    
    for (i, k, j) in boundary_triples:
        U[i * m + j][rank] = 1
        V[j * p + k][rank] = 1
        W[k * n + i][rank] = 1
        rank += 1
    
    # Trim unused columns
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    return {"U": U, "V": V, "W": W}

def _greedy_build(n, m, p, max_rank, time_limit=30):
    """
    Greedy construction: start with schoolbook, then try to merge terms
    to reduce rank. Returns best decomposition found.
    """
    start_time = time.time()
    
    # Start with schoolbook
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
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to remove terms by checking if they're redundant
    for attempt in range(200):
        if time.time() - start_time > time_limit:
            break
        if rank <= max_rank:
            break
            
        # Try removing each term
        found_merge = False
        for t in range(rank - 1, -1, -1):
            if time.time() - start_time > time_limit:
                break
                
            # Temporarily remove term t
            U_test = U.copy()
            V_test = V.copy()
            W_test = W.copy()
            
            # Zero out column t
            U_test[:, t] = 0
            V_test[:, t] = 0
            W_test[:, t] = 0
            
            # Check if still valid
            U3 = U_test[:, :rank].reshape(n, m, -1)
            V3 = V_test[:, :rank].reshape(m, p, -1)
            W3 = W_test[:, :rank].reshape(p, n, -1)
            
            # Only check non-zero parts for efficiency
            valid = True
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        val = np.sum(U3[i, j, :] * V3[j, k, :] * W3[k, i, :])
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                # Remove this column
                U = np.delete(U, t, axis=1)
                V = np.delete(V, t, axis=1)
                W = np.delete(W, t, axis=1)
                rank -= 1
                found_merge = True
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
        
        if not found_merge:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}, best_rank

def construct():
    """Construct a rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (150s limit, 10s buffer)
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best_result = _construct_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try the Strassen-embedded approach
    try:
        strassen_result = _construct_strassen_embedded(n, m, p)
        rank_s = len(strassen_result["U"][0]) if strassen_result["U"] else 0
        if rank_s < best_rank:
            if _validate(strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p):
                best_result = strassen_result
                best_rank = rank_s
    except Exception:
        pass
    
    # Try greedy improvement
    try:
        greedy_result, greedy_rank = _greedy_build(n, m, p, best_rank, time_limit=60)
        if greedy_rank < best_rank:
            if _validate(greedy_result["U"], greedy_result["V"], greedy_result["W"], n, m, p):
                best_result = greedy_result
                best_rank = greedy_rank
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_result
    
    return best_result

# EVOLVE-BLOCK-END
