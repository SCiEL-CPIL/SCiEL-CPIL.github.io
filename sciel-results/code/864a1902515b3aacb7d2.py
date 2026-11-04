"""Matrix multiplication 4x4x5 rank-minimal decomposition using Strassen + schoolbook."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (U, V, W) where U, V, W are vectors of length 4 (for 2x2 matrices).
    
    For C = A * B where A, B are 2x2:
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    Strassen:
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C11 = M1 - M4 + M5 + M6
    C12 = M3 + M4
    C21 = M2 + M5
    C22 = M1 - M3 + M2 - M7
    """
    # Index convention: U[a*2+b], V[c*2+d], W[e*2+f] for 2x2 matrices
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    terms = []
    
    # M1 = (a+d)(e+h)
    # a=A[0,0]=idx0, d=A[1,1]=idx3
    # e=B[0,0]=idx0, h=B[1,1]=idx3
    U1 = [1, 0, 0, 1]
    V1 = [1, 0, 0, 1]
    # C11 += M1, C22 += M1
    W1 = [1, 0, 0, 1]
    terms.append((U1, V1, W1))
    
    # M2 = (c+d)e
    # c=A[1,0]=idx2, d=A[1,1]=idx3
    # e=B[0,0]=idx0
    U2 = [0, 0, 1, 1]
    V2 = [1, 0, 0, 0]
    # C21 += M2, C22 += M2
    W2 = [0, 0, 1, 1]
    terms.append((U2, V2, W2))
    
    # M3 = a(f-h)
    # a=A[0,0]=idx0
    # f=B[0,1]=idx1, h=B[1,1]=idx3
    U3 = [1, 0, 0, 0]
    V3 = [0, 1, 0, -1]
    # C12 += M3, C22 -= M3
    W3 = [0, 1, 0, -1]
    terms.append((U3, V3, W3))
    
    # M4 = d(g-e)
    # d=A[1,1]=idx3
    # g=B[1,0]=idx2, e=B[0,0]=idx0
    U4 = [0, 0, 0, 1]
    V4 = [0, 0, 1, -1]
    # C11 -= M4, C12 += M4
    W4 = [-1, 1, 0, 0]
    terms.append((U4, V4, W4))
    
    # M5 = (a+b)h
    # a=A[0,0]=idx0, b=A[0,1]=idx1
    # h=B[1,1]=idx3
    U5 = [1, 1, 0, 0]
    V5 = [0, 0, 0, 1]
    # C11 += M5, C21 += M5
    W5 = [1, 0, 1, 0]
    terms.append((U5, V5, W5))
    
    # M6 = (c-a)(e+f)
    # c=A[1,0]=idx2, a=A[0,0]=idx0
    # e=B[0,0]=idx0, f=B[0,1]=idx1
    U6 = [0, 0, 1, -1]  # c - a: idx2=1, idx0=-1
    V6 = [1, 1, 0, 0]   # e + f: idx0=1, idx1=1
    # C11 += M6
    W6 = [1, 0, 0, 0]
    terms.append((U6, V6, W6))
    
    # M7 = (b-d)(g+h)
    # b=A[0,1]=idx1, d=A[1,1]=idx3
    # g=B[1,0]=idx2, h=B[1,1]=idx3
    U7 = [0, 1, 0, -1]  # b - d: idx1=1, idx3=-1
    V7 = [0, 0, 1, 1]   # g + h: idx2=1, idx3=1
    # C22 -= M7
    W7 = [0, 0, 0, -1]
    terms.append((U7, V7, W7))
    
    return terms

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid decomposition."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                if not np.isscalar(val):
                    val = float(np.sum(val))
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Strategy: Split 4x4 * 4x5 into 4x4 * 4x4 (Strassen, rank 49) + 4x4 * 4x1 (schoolbook, rank 4).
    Total rank: 53.
    """
    n, m, p = 4, 4, 5
    deadline = time.time() + 140
    
    # Build Strassen terms for 2x2x2
    strassen_terms = _strassen_2x2_terms()
    
    # For 4x4x4: split into 2x2 blocks
    # A is 4x4, split into [[A11, A12], [A21, A22]] where each is 2x2
    # B is 4x4, split into [[B11, B12], [B21, B22]] where each is 2x2
    # C is 4x4, split into [[C11, C12], [C21, C22]] where each is 2x2
    #
    # Each Strassen term operates on 2x2 blocks. For the 4x4 case, we apply
    # Strassen to each pair of 2x2 block positions.
    #
    # The 4x4x4 tensor can be decomposed using the Kronecker structure:
    # For each of the 7 Strassen terms, and for each of the 4 block positions
    # in the output, we get a term. But actually, we need to be more careful.
    #
    # Actually, the standard approach: the 4x4x4 multiplication tensor has
    # rank at most 7^3 = 343 by naive Kronecker, but we can do better.
    # 
    # A better approach: use Strassen directly on the 4x4 by viewing it as
    # 2x2 blocks of 2x2 matrices. The Strassen algorithm for 2x2 matrices
    # of 2x2 blocks gives 7 multiplications of 2x2 matrices, each of which
    # can be done with schoolbook (4 multiplications) or recursively.
    #
    # For rank minimization, let's use schoolbook for the inner 2x2 multiplications.
    # So 7 Strassen multiplications * 4 schoolbook multiplications each = 28?
    # No, that's not right either.
    #
    # Let me think differently. The 4x4x4 tensor T4[i,j,k] = 1 for all i,j,k in {0,1,2,3}.
    # We want to decompose it.
    #
    # Using the block structure:
    # i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0
    # where i1,j1,k1 in {0,1} and i0,j0,k0 in {0,1}
    #
    # T4[i,j,k] = T2[i1,j1,k1] * T2[i0,j0,k0]... no, that's not how it works.
    #
    # Actually, for matrix multiplication, the tensor is:
    # T[i,j,k] = 1 (all ones) for the standard multiplication tensor.
    # The decomposition is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    #
    # For 4x4x4, we can use the fact that it's a Kronecker product of 2x2x2 tensors.
    # T4 = T2 ⊗ T2 (in the appropriate sense)
    #
    # If T2 has rank 7 with decomposition (U2, V2, W2), then T4 has rank 7*7 = 49
    # with decomposition given by the Kronecker product of the factors.
    #
    # Specifically, if T2[a,b,c] = sum_r U2[a,b,r] * V2[b,c,r] * W2[c,a,r]
    # then T4[i,j,k] = sum_{r1,r2} U2[i1,j1,r1]*U2[i0,j0,r2] * V2[j1,k1,r1]*V2[j0,k0,r2] * W2[k1,i1,r1]*W2[k0,i0,r2]
    #
    # So for 4x4x4, rank = 49.
    
    # Build the 4x4x4 decomposition using Kronecker product of Strassen terms
    U_444 = []
    V_444 = []
    W_444 = []
    
    for r1 in range(7):
        for r2 in range(7):
            U1, V1, W1 = strassen_terms[r1]
            U2, V2, W2 = strassen_terms[r2]
            
            # U for 4x4: index is i*4+j, where i=2*i1+i0, j=2*j1+j0
            # U_444[i*4+j] = U1[i1,j1] * U2[i0,j0]
            U_term = [0.0] * 16
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            i = 2*i1 + i0
                            j = 2*j1 + j0
                            U_term[i*4+j] = U1[i1*2+j1] * U2[i0*2+j0]
            
            # V for 4x4: index is j*4+k, where j=2*j1+j0, k=2*k1+k0
            # V_444[j*4+k] = V1[j1,k1] * V2[j0,k0]
            V_term = [0.0] * 16
            for j1 in range(2):
                for k1 in range(2):
                    for j0 in range(2):
                        for k0 in range(2):
                            j = 2*j1 + j0
                            k = 2*k1 + k0
                            V_term[j*4+k] = V1[j1*2+k1] * V2[j0*2+k0]
            
            # W for 4x4: index is k*4+i, where k=2*k1+k0, i=2*i1+i0
            # W_444[k*4+i] = W1[k1,i1] * W2[k0,i0]
            W_term = [0.0] * 16
            for k1 in range(2):
                for i1 in range(2):
                    for k0 in range(2):
                        for i0 in range(2):
                            k = 2*k1 + k0
                            i = 2*i1 + i0
                            W_term[k*4+i] = W1[k1*2+i1] * W2[k0*2+i0]
            
            U_444.append(U_term)
            V_444.append(V_term)
            W_444.append(W_term)
    
    rank_444 = len(U_444)  # Should be 49
    
    # Build schoolbook decomposition for 4x4 * 4x1
    # A is 4x4, B is 4x1, C is 4x1
    # C[i,0] = sum_j A[i,j] * B[j,0]
    # Rank = 4 (one term per j)
    U_441 = []
    V_441 = []
    W_441 = []
    
    for j in range(4):
        U_term = [0.0] * 16  # 4x4 matrix, index i*4+j
        U_term[j] = 1.0  # A[i,j] for all i, only j-th column matters
        # Wait, U[i*4+j] should be 1 for all i when we're extracting column j
        # Actually, for schoolbook: C[i] = sum_j A[i,j] * B[j]
        # Term j: U[i*4+j] = 1 for all i, V[j*1+0] = 1, W[0*4+i] = 1 for all i
        for i in range(4):
            U_term[i*4+j] = 1.0
        
        V_term = [0.0] * 4  # 4x1 matrix, index j*1+0 = j
        V_term[j] = 1.0
        
        W_term = [0.0] * 4  # 1x4 matrix... wait, W is (p*n, rank) = (1*4, rank) = (4, rank)
        # W[k*n+i] where k=0 (only column), i in 0..3
        for i in range(4):
            W_term[0*4+i] = 1.0
        
        U_441.append(U_term)
        V_441.append(V_term)
        W_441.append(W_term)
    
    rank_441 = len(U_441)  # Should be 4
    
    # Now combine for 4x4 * 4x5
    # The 4x5 matrix B is split into B_left (4x4) and B_right (4x1)
    # C = A * B = A * [B_left | B_right] = [A*B_left | A*B_right]
    #
    # For the full 4x4 * 4x5 problem:
    # U has shape (4*4, rank) = (16, rank)
    # V has shape (4*5, rank) = (20, rank)  
    # W has shape (5*4, rank) = (20, rank)
    #
    # For the 4x4x4 part:
    # U_444[i*4+j] for i,j in 0..3
    # V_444[j*4+k] for j,k in 0..3 (but in the full problem, k goes 0..4 for the 4x5)
    # W_444[k*4+i] for k,i in 0..3
    #
    # In the full 4x4*4x5:
    # U[i*4+j, r] = U_444[i*4+j, r] for the 4x4x4 part
    # V[j*5+k, r] = V_444[j*4+k, r] for k in 0..3 (left 4 columns)
    # W[k*4+i, r] = W_444[k*4+i, r] for k in 0..3 (first 4 rows of output)
    #
    # For the 4x4x1 part:
    # U[i*4+j, r] = U_441[i*4+j, r]
    # V[j*5+4, r] = V_441[j, r] (the 5th column, index 4)
    # W[4*4+i, r] = W_441[i, r] (5th row of output, index 4*4+i = 16+i)
    
    total_rank = rank_444 + rank_441
    
    U_full = [[0.0] * total_rank for _ in range(16)]
    V_full = [[0.0] * total_rank for _ in range(20)]
    W_full = [[0.0] * total_rank for _ in range(20)]
    
    # Add 4x4x4 part
    for r in range(rank_444):
        for idx in range(16):
            U_full[idx][r] = U_444[r][idx]
        for idx in range(16):
            V_full[idx][r] = V_444[r][idx]  # j*4+k for j,k in 0..3 -> indices 0..15 in V (which is 4*5=20)
        for idx in range(16):
            W_full[idx][r] = W_444[r][idx]  # k*4+i for k,i in 0..3 -> indices 0..15 in W (which is 5*4=20)
    
    # Add 4x4x1 part
    for r in range(rank_441):
        r_offset = rank_444 + r
        for idx in range(16):
            U_full[idx][r_offset] = U_441[r][idx]
        # V: j*5+4 for j in 0..3 -> indices 4, 9, 14, 19
        for j in range(4):
            V_full[j*5+4][r_offset] = V_441[r][j]
        # W: 4*4+i for i in 0..3 -> indices 16, 17, 18, 19
        for i in range(4):
            W_full[16+i][r_offset] = W_441[r][i]
    
    # Verify
    if not _verify(U_full, V_full, W_full, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U_sb = [[0.0] * rank for _ in range(n*m)]
        V_sb = [[0.0] * rank for _ in range(m*p)]
        W_sb = [[0.0] * rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][col] = 1.0
                    V_sb[j*p+k][col] = 1.0
                    W_sb[k*n+i][col] = 1.0
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
