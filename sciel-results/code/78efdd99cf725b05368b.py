"""Matrix multiplication 3x4x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n=3, m=4, p=8):
    """Check if the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check all consistent triples sum to 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > 1e-8:
                    return False
    
    # Check inconsistent j
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j == j2: continue
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j2*p+k, :] * W[k*n+i, :])
                    if abs(s) > 1e-8:
                        return False
    
    # Check inconsistent k
    for j in range(m):
        for k in range(p):
            for k2 in range(p):
                if k == k2: continue
                for i in range(n):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k2*n+i, :])
                    if abs(s) > 1e-8:
                        return False
    
    # Check inconsistent i
    for i in range(n):
        for i2 in range(n):
            if i == i2: continue
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i2, :])
                    if abs(s) > 1e-8:
                        return False
    
    return True

def _compute_error(U, V, W, n=3, m=4, p=8):
    """Compute total squared error of the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err += (s - 1.0)**2
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j == j2: continue
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j2*p+k, :] * W[k*n+i, :])
                    err += s**2
    for j in range(m):
        for k in range(p):
            for k2 in range(p):
                if k == k2: continue
                for i in range(n):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k2*n+i, :])
                    err += s**2
    for i in range(n):
        for i2 in range(n):
            if i == i2: continue
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i2, :])
                    err += s**2
    return err

def _build_strassen_2x2():
    """Build Strassen decomposition for 2x2 matrix multiplication.
    Returns U, V, W of shape (4, 7), (4, 7), (4, 7).
    C[i,k] = sum_j A[i,j]*B[j,k]
    U indexed by (i,j), V by (j,k), W by (k,i)
    """
    r = 7
    U = np.zeros((4, r))
    V = np.zeros((4, r))
    W = np.zeros((4, r))
    
    # Strassen: M1=A00*B00, M2=A10*B01, M3=A01*B10, M4=A11*B11,
    # M5=(A00+A11)*(B00+B11), M6=(A10+A11)*(B10-B00), M7=(A00-A10)*(B01+B11)
    # C00=M1+M5-M4-M6, C01=M3+M7, C10=M2+M4, C11=M1-M5+M2+M7
    
    # M1 = A[0,0]*B[0,0]
    U[0*2+0, 0] = 1; V[0*2+0, 0] = 1; W[0*2+0, 0] = 1  # C[0,0] gets +1
    U[0*2+0, 0] += 0  # already set
    # Actually let me be more careful with indexing
    # U[i*2+j], V[j*2+k], W[k*2+i]
    
    # Reset
    U = np.zeros((4, r))
    V = np.zeros((4, r))
    W = np.zeros((4, r))
    
    # M1 = A00 * B00 -> contributes to C00(+), C11(+)
    U[0, 0] = 1; V[0, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0] and C[1,1]
    
    # M2 = A10 * B01 -> contributes to C10(+), C11(+)
    U[2, 1] = 1; V[1, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1  # C[1,0] and C[1,1]
    
    # M3 = A01 * B10 -> contributes to C01(+)
    U[1, 2] = 1; V[2, 2] = 1
    W[1, 2] = 1  # C[0,1]
    
    # M4 = A11 * B11 -> contributes to C00(-), C10(+), C11(+)
    U[3, 3] = 1; V[3, 3] = 1
    W[0, 3] = -1; W[2, 3] = 1; W[3, 3] = 1
    
    # M5 = (A00+A11) * (B00+B11) -> contributes to C00(+), C11(-)
    U[0, 4] = 1; U[3, 4] = 1
    V[0, 4] = 1; V[3, 4] = 1
    W[0, 4] = 1; W[3, 4] = -1
    
    # M6 = (A10+A11) * (B10-B00) -> contributes to C00(-)
    U[2, 5] = 1; U[3, 5] = 1
    V[2, 5] = 1; V[0, 5] = -1
    W[0, 5] = -1
    
    # M7 = (A00-A10) * (B01+B11) -> contributes to C01(+), C11(+)
    U[0, 6] = 1; U[2, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _embed_2x2_in_3x2x8(U22, V22, W22, base_col=0, row_offset_A=0, row_offset_B=0, row_offset_C=0):
    """Embed a 2x2x2 Strassen decomposition into a 3x2x8 problem.
    
    The 2x2 subproblem handles A[row_offset_A:row_offset_A+2, 0:2] * B[0:2, col_offset:col_offset+2]
    But we need to handle the full 3x2x8 structure.
    
    Actually, let me think differently. For 3x2x8:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    We can split k into groups. For each pair of k values (k1,k2), we have a 3x2x2 subproblem.
    A 3x2x2 can be done with rank 6 (schoolbook) or potentially less.
    
    Let me just use schoolbook for 3x2x2: rank = 3*2*2 = 12 per 8 k-values -> 48 total.
    Or use the 2x2 Strassen on the B side.
    """
    pass

def _build_3x2x8_decomp():
    """Build decomposition for 3x2 * 2x8 -> 3x8.
    
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    Split k into 4 pairs: (0,1), (2,3), (4,5), (6,7)
    For each pair (k1,k2), we have a 3x2x2 subproblem.
    
    For 3x2x2: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for k in {k1,k2}
    This is like multiplying a 3x2 by a 2x2.
    
    We can use a 2x2 Strassen-like approach on the B matrix side.
    Actually, for 3x2x2, the schoolbook rank is 12.
    
    Alternative: view it as two 3x1x2 problems (one for each j).
    
    Let me try: for each k-pair, use rank-6 decomposition.
    Total: 4 * 6 = 24 for the 3x2x8.
    
    Actually, let me just construct it directly.
    For 3x2x2 (with k in {0,1}):
    C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1]
    
    Schoolbook: 12 terms.
    
    Can we do better? The tensor is T[i,j,k] = 1 for all i,j,k in the 3x2x2 box.
    This is just a product of three vectors: (1,1,1) ⊗ (1,1) ⊗ (1,1), so rank 1!
    
    Wait, that's for the specific case where all entries are 1. But we need the actual
    multiplication structure. Let me reconsider.
    
    For 3x2x2: the multiplication tensor has T[i,j,k] = 1 for i in {0,1,2}, j in {0,1}, k in {0,1}.
    This is indeed a rank-1 tensor (outer product of ones vectors). So rank 1 suffices!
    
    But wait, that's only if we're computing the specific matrix product where all A and B entries
    are 1. No, the tensor decomposition needs to work for ALL matrices.
    
    The multiplication tensor for m×n * n×p is T[i,j,k] = 1 if the indices are "consistent" 
    (i.e., the j in A matches the j in B). For 3x2x2, ALL (i,j,k) triples are consistent 
    because there's only 2 possible j values and they always match. So the tensor is 
    T[i,j,k] = 1 for all i,j,k, which IS rank 1.
    
    Wait, that can't be right. Let me reconsider.
    
    For matrix multiplication C = A*B where A is 3x2 and B is 2x8:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor is T[i,j,k] = 1 for all valid (i,j,k) where i in {0,1,2}, j in {0,1}, k in {0,...,7}.
    This is the outer product of (1,1,1), (1,1), (1,1,...,1) which has rank 1.
    
    So the 3x2x8 multiplication has tensor rank 1?! That means:
    U[i,j] = 1 for all i,j
    V[j,k] = 1 for all j,k  
    W[k,i] = 1 for all k,i
    
    Let me verify: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = U[i,j,0]*V[j,k,0]*W[k,i,0] = 1*1*1 = 1 ✓
    
    And for inconsistent indices, there are none in this case since j always ranges over {0,1}
    and both A and B use the same j index.
    
    So 3x2x8 has rank 1! That means 3x4x8 = (3x2x8) + (3x2x8) has rank at most 2!
    
    But wait, that's not how it works. The two 3x2x8 subproblems share the same i and k indices
    but have different j indices. Let me reconsider.
    
    For 3x4x8: C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    Split: C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    
    First term: uses A[i,0:2] and B[0:2,k] -> 3x2x8 tensor
    Second term: uses A[i,2:4] and B[2:4,k] -> 3x2x8 tensor
    
    These are separate tensors with different (i,j) and (j,k) index ranges.
    The first has U indexed by (i, j) for j in {0,1}, V by (j,k) for j in {0,1}
    The second has U indexed by (i, j) for j in {2,3}, V by (j,k) for j in {2,3}
    
    They don't interfere because they use different j indices.
    
    So total rank = 1 + 1 = 2?!
    
    Let me verify this more carefully.
    """
    n, m, p = 3, 2, 8
    # For 3x2x8, the tensor is rank 1
    U = np.ones((n*m, 1))  # shape (6, 1)
    V = np.ones((m*p, 1))  # shape (16, 1)
    W = np.ones((p*n, 1))  # shape (24, 1)
    return U, V, W

def _build_hybrid_decomp():
    """Build hybrid decomposition for 3x4x8 using the rank-1 structure of 3x2x8.
    
    3x4x8 = (3x2x8 for j=0,1) + (3x2x8 for j=2,3)
    Each 3x2x8 has rank 1.
    Total rank = 2.
    """
    n, m, p = 3, 4, 8
    rank = 2
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # First subproblem: j in {0,1}
    for i in range(n):
        for j in range(2):
            U[i*m+j, 0] = 1.0
    for j in range(2):
        for k in range(p):
            V[j*p+k, 0] = 1.0
    for k in range(p):
        for i in range(n):
            W[k*n+i, 0] = 1.0
    
    # Second subproblem: j in {2,3}
    for i in range(n):
        for j in range(2, 4):
            U[i*m+j, 1] = 1.0
    for j in range(2, 4):
        for k in range(p):
            V[j*p+k, 1] = 1.0
    for k in range(p):
        for i in range(n):
            W[k*n+i, 1] = 1.0
    
    return U, V, W

def _build_strassen_hybrid():
    """Build a Strassen-based hybrid decomposition for 3x4x8.
    
    Strategy: Split the 8 output columns into 4 groups of 2.
    For each group of 2 columns, we have a 3x4x2 subproblem.
    C[i,k] for k in {2g, 2g+1} = sum_j A[i,j]*B[j,k]
    
    For 3x4x2: split j into {0,1} and {2,3}
    C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    
    Each 3x2x2 subproblem has rank 1 (as shown above).
    So 3x4x2 has rank 2.
    Total: 4 * 2 = 8.
    
    But we can do better by using Strassen on the 2-column groups.
    
    Actually, let me reconsider. The 3x4x2 tensor:
    T[i,j,k] = 1 for all i in {0,1,2}, j in {0,1,2,3}, k in {0,1}
    This is the outer product of (1,1,1), (1,1,1,1), (1,1) -> rank 1!
    
    So 3x4x2 has rank 1, and 3x4x8 = 4 * (3x4x2) has rank 4.
    
    Wait, but the 4 subproblems share the same (i,j) indices for A. Let me check.
    
    For k in {0,1}: uses A[i,j] for all i,j and B[j,0:2]
    For k in {2,3}: uses A[i,j] for all i,j and B[j,2:4]
    
    The U matrix is the same for all groups (since it depends on (i,j) which is the same).
    The V matrix differs (different k ranges).
    The W matrix differs (different k ranges).
    
    So we can share the U part! Total rank = 4 (one per k-group).
    
    Let me construct this.
    """
    n, m, p = 3, 4, 8
    rank = 4  # one per pair of k values
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # U is the same for all groups (all 1s)
    U[:, :] = 1.0
    
    # V: for group g (k in {2g, 2g+1}), V[j, 2g] and V[j, 2g+1] = 1
    for g in range(4):
        for j in range(m):
            V[j*p + 2*g, g] = 1.0
            V[j*p + 2*g + 1, g] = 1.0
    
    # W: for group g (k in {2g, 2g+1}), W[2g, i] and W[2g+1, i] = 1
    for g in range(4):
        for i in range(n):
            W[2*g*n + i, g] = 1.0
            W[(2*g+1)*n + i, g] = 1.0
    
    return U, V, W

def _build_strassen_2x2x8():
    """Build Strassen-based decomposition for 2x2x8.
    
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1}, k in {0,...,7}
    
    Split k into 4 pairs. For each pair (k1,k2), we have a 2x2x2 subproblem.
    2x2x2 has rank 1 (outer product of ones).
    Total: 4.
    
    But we can use Strassen on the 2x2 A matrix side.
    Actually, for 2x2x2, the tensor is T[i,j,k]=1 for all i,j,k, which is rank 1.
    
    So 2x2x8 has rank 4 (one per k-pair).
    """
    n, m, p = 2, 2, 8
    rank = 4
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U[:, :] = 1.0
    
    for g in range(4):
        for j in range(m):
            V[j*p + 2*g, g] = 1.0
            V[j*p + 2*g + 1, g] = 1.0
        for i in range(n):
            W[2*g*n + i, g] = 1.0
            W[(2*g+1)*n + i, g] = 1.0
    
    return U, V, W

def _try_optimize(U, V, W, n=3, m=4, p=8, max_iters=30, tol=1e-8):
    """Optimize a decomposition using alternating least squares."""
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    rank = U.shape[1]
    
    for it in range(max_iters):
        # Update U
        for i in range(n):
            for j in range(m):
                rows = []
                rhs = []
                for k in range(p):
                    rows.append(V[j*p+k, :] * W[k*n+i, :])
                    rhs.append(1.0)
                for j2 in range(m):
                    if j2 == j: continue
                    for k in range(p):
                        rows.append(V[j2*p+k, :] * W[k*n+i, :])
                        rhs.append(0.0)
                for i2 in range(n):
                    if i2 == i: continue
                    for k in range(p):
                        rows.append(V[j*p+k, :] * W[k*n+i2, :])
                        rhs.append(0.0)
                
                A_mat = np.array(rows)
                b_vec = np.array(rhs)
                try:
                    U[i*m+j, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                rows = []
                rhs = []
                for i in range(n):
                    rows.append(U[i*m+j, :] * W[k*n+i, :])
                    rhs.append(1.0)
                for j2 in range(m):
                    if j2 == j: continue
                    for i in range(n):
                        rows.append(U[i*m+j2, :] * W[k*n+i, :])
                        rhs.append(0.0)
                for k2 in range(p):
                    if k2 == k: continue
                    for i in range(n):
                        rows.append(U[i*m+j, :] * W[k2*n+i, :])
                        rhs.append(0.0)
                
                A_mat = np.array(rows)
                b_vec = np.array(rhs)
                try:
                    V[j*p+k, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                rows = []
                rhs = []
                for j in range(m):
                    rows.append(U[i*m+j, :] * V[j*p+k, :])
                    rhs.append(1.0)
                for k2 in range(p):
                    if k2 == k: continue
                    for j in range(m):
                        rows.append(U[i*m+j, :] * V[j*p+k2, :])
                        rhs.append(0.0)
                for i2 in range(n):
                    if i2 == i: continue
                    for j in range(m):
                        rows.append(U[i2*m+j, :] * V[j*p+k, :])
                        rhs.append(0.0)
                
                A_mat = np.array(rows)
                b_vec = np.array(rhs)
                try:
                    W[k*n+i, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    pass
        
        # Check error
        err = _compute_error(U, V, W, n, m, p)
        if err < tol:
            break
    
    return U, V, W, err

def construct():
    n, m, p = 3, 4, 8
    t_start = time.time()
    time_limit = 140  # Safety margin before 150s kill
    
    # Start with the rank-4 structured decomposition
    U0, V0, W0 = _build_strassen_hybrid()
    
    # Verify the rank-4 decomposition
    if _verify(U0, V0, W0, n, m, p):
        best_U = U0.copy()
        best_V = V0.copy()
        best_W = W0.copy()
        best_rank = 4
        print(f"Structured rank-4 decomposition verified!")
    else:
        print("Rank-4 decomposition failed verification, falling back to schoolbook")
        # Fall back to schoolbook
        rank0 = n * m * p
        best_U = np.zeros((n*m, rank0))
        best_V = np.zeros((m*p, rank0))
        best_W = np.zeros((p*n, rank0))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i*m+j, col] = 1.0
                    best_V[j*p+k, col] = 1.0
                    best_W[k*n+i, col] = 1.0
                    col += 1
        best_rank = rank0
    
    # Try to find even lower rank
    if best_rank > 1:
        # Try rank 1
        if time.time() - t_start < time_limit - 10:
            U1 = np.ones((n*m, 1))
            V1 = np.ones((m*p, 1))
            W1 = np.ones((p*n, 1))
            if _verify(U1, V1, W1, n, m, p):
                best_U = U1
                best_V = V1
                best_W = W1
                best_rank = 1
                print("Found rank-1 decomposition!")
        
        # Try rank 2
        if best_rank > 2 and time.time() - t_start < time_limit - 10:
            U2, V2, W2 = _build_hybrid_decomp()
            if _verify(U2, V2, W2, n, m, p):
                best_U = U2
                best_V = V2
                best_W = W2
                best_rank = 2
                print("Found rank-2 decomposition!")
        
        # Try rank 3
        if best_rank > 3 and time.time() - t_start < time_limit - 20:
            rng = np.random.RandomState(42)
            for attempt in range(10):
                if time.time() - t_start > time_limit - 10:
                    break
                U3 = rng.randn(n*m, 3) * 0.5
                V3 = rng.randn(m*p, 3) * 0.5
                W3 = rng.randn(p*n, 3) * 0.5
                U3, V3, W3, err = _try_optimize(U3, V3, W3, n, m, p, max_iters=30)
                if _verify(U3, V3, W3, n, m, p):
                    best_U = U3
                    best_V = V3
                    best_W = W3
                    best_rank = 3
                    print("Found rank-3 decomposition!")
                    break
    
    # Round to half-integers
    Ur = np.round(best_U * 2) / 2
    Vr = np.round(best_V * 2) / 2
    Wr = np.round(best_W * 2) / 2
    if _verify(Ur, Vr, Wr, n, m, p):
        best_U, best_V, best_W = Ur, Vr, Wr
    
    # Convert to lists with int/half-integer values
    def _convert(v):
        if abs(v - round(v)) < 1e-9:
            return int(round(v))
        elif abs(v * 2 - round(v * 2)) < 1e-9:
            return (2 * int(round(v * 2))) / 2
        return v
    
    U_out = [[_convert(v) for v in row] for row in best_U.tolist()]
    V_out = [[_convert(v) for v in row] for row in best_V.tolist()]
    W_out = [[_convert(v) for v in row] for row in best_W.tolist()]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
