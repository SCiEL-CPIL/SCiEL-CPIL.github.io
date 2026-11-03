import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _get_strassen_222():
    """
    Returns (U, V, W) for 2x2x2 Strassen multiplication.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen Products M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # V indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # W indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) representing C[i,j] -> k=i, j? 
    # Wait, the task convention is W[k*n+i, r].
    # For 2x2x2 block, n=2, m=2, p=2.
    # W index local (lk*2 + li) corresponds to global W[k*n+i].
    # In the sub-matrix context, the output C is 2x2.
    # C[li, lj] = sum_{lk} A[li, lk] B[lk, lj].
    # The tensor identity is sum_r W[local_k*n + local_i] U[local_i*m + local_j] V[local_j*p + local_k]
    # Here n=2, m=2, p=2.
    # W[local_k*2 + local_i]
    
    # Let's define the linear combos for U (A), V (B), W (C) for each product r
    
    # Product 0 (M1):
    # U: A00 + A11 -> U[0][0]=1, U[3][0]=1
    # V: B00 + B11 -> V[0][0]=1, V[3][0]=1
    # W: C00 + C11 -> W[0][0]=1 (k=0,i=0), W[3][0]=1 (k=1,i=1)
    # Note: C00 and C11 get this contribution.
    # But we need to reconstruct C from M1..M7.
    # Standard Strassen Recurrence:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So W entries:
    # M1 (r=0): C00 (+1), C11 (+1) -> W[0*2+0][0]=1, W[1*2+1][0]=1
    # M2 (r=1): C10 (+1), C11 (-1) -> W[0*2+1][1]=1, W[1*2+1][1]=-1
    # M3 (r=2): C01 (+1), C11 (+1) -> W[1*2+0][2]=1, W[1*2+1][2]=1
    # M4 (r=3): C00 (-1), C10 (+1) -> W[0*2+0][3]=-1, W[0*2+1][3]=1
    # M5 (r=4): C00 (+1), C01 (+1) -> W[0*2+0][4]=1, W[1*2+0][4]=1
    # M6 (r=5): C11 (+1) -> W[1*2+1][5]=1
    # M7 (r=6): C00 (-1) -> W[0*2+0][6]=-1
    
    # Let's verify this logic carefully.
    # C00 = M1 - M4 + M5 - M7
    # M1: W[0][0]=1
    # M4: W[0][0]=-1
    # M5: W[0][0]=1
    # M7: W[0][0]=-1
    # Sum: 1-1+1-1 = 0? No.
    # C00 should be 1 * A00 * B00 + ...
    # The tensor decomposition must satisfy:
    # sum_r W_kn+i, r U_im+j, r V_jp+k, r = delta_{i,k} delta_{j,l} ? No.
    # The tensor T_{i,j,k} = delta_{i,k} is for identity?
    # Matrix multiplication tensor T_{i,j,k} where C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So T_{i,k,j}? No, the contract is W[k*n+i].
    # So T_{i,j,k} = 1 if k is the column index of B and i is row of A?
    # Yes, the term in the sum is A[i,j] * B[j,k] * W[k,i].
    # So for the product to be 1, we need the sum to pick out A[i,j] B[j,k].
    
    # Let's just use the explicit definition of the 7 Strassen terms on the (U,V,W) tensors.
    
    # Term 0: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # Contribution to C:
    # C00 += M1
    # C11 += M1
    # So W[0][0] += 1 (k=0, i=0)
    # W[3][0] += 1 (k=1, i=1)
    W[0][0] += 1
    W[3][0] += 1
    
    # Term 1: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # Contribution:
    # (A10+A11)B00 = A10B00 + A11B00
    # C10 += A10B00
    # C11 += A11B00
    # So C10 gets +1, C11 gets +1?
    # Wait, standard Strassen:
    # M2 = (A10 + A11)B00
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So:
    # W[2][1] (k=0, i=1 -> C10) += 1
    # W[3][1] (k=1, i=1 -> C11) += -1
    W[2][1] += 1
    W[3][1] += -1
    
    # Term 2: A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # M3 = A00(B01-B11)
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    # So:
    # W[1][2] (k=1, i=0 -> C01) += 1
    # W[3][2] (k=1, i=1 -> C11) += 1
    W[1][2] += 1
    W[3][2] += 1
    
    # Term 3: A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # M4 = A11(B10-B00)
    # C00 = M1 - M4 + M5 - M7
    # C10 = M2 + M4
    # So:
    # W[0][3] (k=0, i=0 -> C00) += -1
    # W[2][3] (k=0, i=1 -> C10) += 1
    W[0][3] += -1
    W[2][3] += 1
    
    # Term 4: (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # M5 = (A00+A01)B11
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # So:
    # W[0][4] (k=0, i=0 -> C00) += 1
    # W[1][4] (k=1, i=0 -> C01) += 1
    W[0][4] += 1
    W[1][4] += 1
    
    # Term 5: (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # M6 = (A10-A00)(B00+B01)
    # C11 = M1 - M2 + M3 + M6
    # So:
    # W[3][5] (k=1, i=1 -> C11) += 1
    W[3][5] += 1
    
    # Term 6: (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M5 - M7
    # So:
    # W[0][6] (k=0, i=0 -> C00) += -1
    W[0][6] += -1
    
    return U, V, W

def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook multiplication."""
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
    return U, V, W

def _build_rank71():
    n, m, p = 3, 5, 5
    
    su, sv, sw = _get_strassen_222()
    
    # Partitions
    i_blocks = [(0,), (1, 2)]
    j_blocks = [(0, 1), (2, 3), (4,)]
    k_blocks = [(0, 1), (2, 3), (4,)]
    
    total_rank = 0
    # Pre-calculate total rank to allocate arrays
    for i_b in i_blocks:
        for j_b in j_blocks:
            for k_b in k_blocks:
                if len(i_b)==2 and len(j_b)==2 and len(k_b)==2:
                    total_rank += 7
                else:
                    total_rank += len(i_b)*len(j_b)*len(k_b)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i_b in i_blocks:
        for j_b in j_blocks:
            for k_b in k_blocks:
                if len(i_b)==2 and len(j_b)==2 and len(k_b)==2:
                    # Strassen
                    for c in range(7):
                        for li in range(2):
                            gi = i_b[li]
                            for lj in range(2):
                                gj = j_b[lj]
                                U[gi*m + gj][col+c] = su[li*2+lj][c]
                        for lj in range(2):
                            gj = j_b[lj]
                            for lk in range(2):
                                gk = k_b[lk]
                                V[gj*p + gk][col+c] = sv[lj*2+lk][c]
                        for lk in range(2):
                            gk = k_b[lk]
                            for li in range(2):
                                gi = i_b[li]
                                W[gk*n + gi][col+c] = sw[lk*2+li][c]
                    col += 7
                else:
                    # Schoolbook
                    for gi in i_b:
                        for gj in j_b:
                            for gk in k_b:
                                U[gi*m + gj][col] = 1
                                V[gj*p + gk][col] = 1
                                W[gk*n + gi][col] = 1
                                col += 1
                                
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, rank):
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Construct tensor T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We want T[i, j, k] == 1 for all i,j,k
    
    # To be faster, we can matrix multiply slices?
    # T[:, j, :] = U[:, j, :]^T V[j, :, :]? No.
    # For fixed j:
    # U[:, j, :] is (n, rank) -> U[i, r]
    # V[j, :, :] is (p, rank) -> V[k, r]
    # Product P = U[:, j, :] * V[j, :, :]^T is (n, p)
    # P[i, k] = sum_r U[i, r] V[k, r]
    # We need sum_r U[i, r] V[k, r] W[k, i, r] ? No.
    # The term is U[i,j,r] V[j,k,r] W[k,i,r].
    # For a fixed r, we have outer products.
    # Let's just loop over i, j, k if rank is small enough, or use einsum.
    # n=3, m=5, p=5. Total 75 terms to check.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(W_np[k*n + i] * U_np[i*m + j] * V_np[j*p + k])
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Attempt rank 71
    try:
        U, V, W = _build_rank71()
        if _verify_decomposition(U, V, W, n, m, p, 71):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
        
    # Fallback to schoolbook
    U, V, W = _schoolbook_factors(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
