"""Matrix multiplication 2x4x8 rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition U, V, W exactly represents the matmul tensor."""
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0.0
                for r in range(rank):
                    lhs += U[i * m + j, r] * V[j * p + k, r]
                expected = W[k * n + i, r] if rank > 0 else 0.0
                # Actually W is the third factor; the tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                # Wait, the standard form is: C[i,k] = sum_j A[i,j]*B[j,k]
                # Tensor T[i,j,k] = A[i,j]*B[j,k], so T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]... no.
                # The decomposition is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # So we need: for each (i,j,k), T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # But T[i,j,k] = delta_ik? No. T[i,j,k] = A[i,j]*B[j,k].
                # The tensor of matmul: T_{ijk} = A_{ij} B_{jk}. 
                # Actually the matmul tensor has indices (i, j, k) and T[i,j,k] = A[i,j]*B[j,k] is not a tensor in the standard sense for this factorization.
                # Let me re-read the problem.
                # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
                # The tensor is the structure tensor of matmul: M = sum_{j} A[:,j] * B[j,:]
                # So M[i,k] = sum_j A[i,j] B[j,k]
                # The structure tensor T has T[i,j,k] such that M[i,k] = sum_j A[i,j] T[i,j,k] B[j,k]? 
                # No. The structure tensor of bilinear form f(A,B) = C is defined by C[i,k] = sum_{j} A[i,j] B[j,k].
                # In tensor decomposition: C[i,k] = sum_r U[i,k,r] * ... 
                # Actually the standard structure tensor for matmul (n x m) * (m x p) -> (n x p):
                # T[i, j, k] = delta_{ik}?? No.
                # Let me look at the seed code:
                # U[i*m+j][column] = 1, V[j*p+k][column] = 1, W[k*n+i][column] = 1
                # For each (i,j,k), there's a column c with U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1.
                # Then for a given (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 (the column for this (i,j,k)).
                # So the tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c].
                # And the schoolbook says T[i,j,k] = 1 for all i,j,k? 
                # Wait, the matmul tensor: M[i,k] = sum_j A[i,j] B[j,k].
                # If we view M as a bilinear map (A,B) -> M, then M[i,k] = sum_{j} A[i,j] B[j,k].
                # The structure tensor T has T[i,j,k] = 1 if we write M[i,k] = sum_{j} A[i,j] T[i,j,k] B[j,k]? That gives T[i,j,k]=1 for all.
                # Hmm, actually in tensor decomposition of matmul, the tensor is T_{ijk} where C_{ik} = sum_{j} A_{ij} T_{ijk} B_{jk}.
                # No wait. Let me think again.
                # C = AB, C_{ik} = sum_j A_{ij} B_{jk}.
                # This is a trilinear form if we write it as sum_{i,j,k} C_{ik}^* A_{ij} B_{jk}... 
                # The standard structure tensor for matmul has entries T_{ijk} = 1 for all valid i,j,k.
                # So the task is to find rank-r decomposition: T_{ijk} = sum_c U_{i*m+j,c} V_{j*p+k,c} W_{k*n+i,c}.
                # Schoolbook: for each (i,j,k), one column with all 1s. Rank = n*m*p.
                # Strassen reduces this by finding linear combinations that are zero.
                
                pass
    # Let me just do the direct verification
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # T[i,j,k] should be 1 for all i,j,k (since schoolbook has T[i,j,k]=1)
    # Wait, is T[i,j,k]=1 for all? 
    # C[i,k] = sum_j A[i,j] B[j,k]
    # If T[i,j,k] = 1 for all, then sum_j A[i,j] * 1 * B[j,k] = sum_j A[i,j] B[j,k] = C[i,k]. Yes!
    # So the tensor to decompose has T[i,j,k] = 1 for all valid i,j,k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2(n_local, m_local, p_local):
    """Return the 7-column Strassen decomposition for a 2x2x2 matmul block.
    
    For a 2x2 matmul, T[i,j,k] = 1 for i,j,k in {0,1}.
    Strassen: 7 multiplications.
    
    The 7 products are:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    We need T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r] where W is indexed by [k*n+i].
    For local 2x2x2, W is indexed by [k*2+i] since n=2.
    
    Let me set up the columns:
    For each product Mr, we have:
    - A-side coefficients: for each (i,j) pair, the coefficient in the A-linear combination
    - B-side coefficients: for each (j,k) pair, the coefficient in the B-linear combination  
    - C-side coefficients: for each (k,i) pair, the coefficient in the C-linear combination
    
    M1: A: A00=1, A11=1; B: B00=1, B11=1; C: C00=1, C11=1
    M2: A: A10=1, A11=1; B: B00=1; C: C10=1
    M3: A: A00=1; B: B01=1, B11=-1; C: C01=1
    M4: A: A11=1; B: B10=1, B00=-1; C: C00=1, C10=1
    M5: A: A00=1, A01=1; B: B11=1; C: C00=-1, C01=1
    M6: A: A10=1, A00=-1; B: B00=1, B01=1; C: C11=1
    M7: A: A01=1, A11=-1; B: B10=1, B11=1; C: C00=1
    """
    # U rows: (i,j) for i,j in {0,1} -> indices 0,1,2,3
    # V rows: (j,k) for j,k in {0,1} -> indices 0,1,2,3
    # W rows: (k,i) for k,i in {0,1} -> indices 0,1,2,3 (k*2+i)
    
    # Column r corresponds to product Mr
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # M1: A: {00:1, 11:1}, B: {00:1, 11:1}, C: {00:1, 11:1}
    # W indices: C00 -> k=0,i=0 -> 0; C11 -> k=1,i=1 -> 3
    U_cols.append([1, 0, 0, 1])   # A00, A01, A10, A11
    V_cols.append([1, 0, 0, 1])   # B00, B01, B10, B11
    W_cols.append([1, 0, 0, 1])   # C00(k0i0), C01(k0i1), C10(k1i0), C11(k1i1)
    
    # M2: A: {10:1, 11:1}, B: {00:1}, C: {10:1}
    U_cols.append([0, 0, 1, 1])
    V_cols.append([1, 0, 0, 0])
    W_cols.append([0, 0, 1, 0])
    
    # M3: A: {00:1}, B: {01:1, 11:-1}, C: {01:1}
    U_cols.append([1, 0, 0, 0])
    V_cols.append([0, 1, 0, -1])
    W_cols.append([0, 1, 0, 0])
    
    # M4: A: {11:1}, B: {10:1, 00:-1}, C: {00:1, 10:1}
    U_cols.append([0, 0, 0, 1])
    V_cols.append([-1, 0, 1, 0])
    W_cols.append([1, 0, 1, 0])
    
    # M5: A: {00:1, 01:1}, B: {11:1}, C: {00:-1, 01:1}
    U_cols.append([1, 1, 0, 0])
    V_cols.append([0, 0, 0, 1])
    W_cols.append([-1, 1, 0, 0])
    
    # M6: A: {10:1, 00:-1}, B: {00:1, 01:1}, C: {11:1}
    U_cols.append([-1, 0, 1, 0])
    V_cols.append([1, 1, 0, 0])
    W_cols.append([0, 0, 0, 1])
    
    # M7: A: {01:1, 11:-1}, B: {10:1, 11:1}, C: {00:1}
    U_cols.append([0, 1, 0, -1])
    V_cols.append([0, 0, 1, 1])
    W_cols.append([1, 0, 0, 0])
    
    return U_cols, V_cols, W_cols


def construct():
    """Construct rank-56 decomposition using Strassen on 8 independent 2x2x2 blocks."""
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Partition j in {0,1,2,3} into 2 groups of 2
    # Partition k in {0,1,2,3,4,5,6,7} into 4 groups of 2
    # This gives 2*4 = 8 blocks, each 2x2x2 (i in {0,1}, 2 j's, 2 k's)
    
    j_groups = [(0, 1), (2, 3)]
    k_groups = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    # Get Strassen columns for a single 2x2x2 block
    U_local, V_local, W_local = _strassen_2x2x2(2, 2, 2)
    
    # Build full matrices
    rank = 8 * 7  # 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    col = 0
    for jg in j_groups:
        for kg in k_groups:
            j0, j1 = jg
            k0, k1 = kg
            for r in range(7):
                # For each of the 7 Strassen products in this block
                for li in range(2):  # local i
                    gi = li  # global i (since i groups are {0,1})
                    u_idx = gi * m + (j0 + li // 2 * 0)  # Need to map local j to global j
                    # Local j=0 -> global j=j0, local j=1 -> global j=j1
                    lj = li  # This is wrong, local j is independent of i
                    pass
                
                # Let me redo this more carefully.
                # For block with j in {j0, j1} and k in {k0, k1}:
                # Local indices: i' in {0,1}, j' in {0,1}, k' in {0,1}
                # Global: i = i', j = j0 + j', k = k0 + k'
                
                for lr in range(7):
                    # U: for each (i', j'), U[i'*m + (j0+j'), col] = U_local[lr][i'*2+j']
                    for li in range(2):
                        for lj in range(2):
                            gi = li
                            gj = j0 + lj
                            U[gi * m + gj][col] = U_local[lr][li * 2 + lj]
                    
                    # V: for each (j', k'), V[(j0+j')*p + (k0+k'), col] = V_local[lr][j'*2+k']
                    for lj in range(2):
                        for lk in range(2):
                            gj = j0 + lj
                            gk = k0 + lk
                            V[gj * p + gk][col] = V_local[lr][lj * 2 + lk]
                    
                    # W: for each (k', i'), W[(k0+k')*n + i', col] = W_local[lr][k'*2+i']
                    for lk in range(2):
                        for li in range(2):
                            gk = k0 + lk
                            gi = li
                            W[gk * n + gi][col] = W_local[lr][lk * 2 + li]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
