import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Build expected tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Strategy:
    # 1. Try to find a rank-12 decomposition via structured search
    # 2. Fall back to rank-14 (Kronecker/Strassen-based)
    # 3. Fall back to schoolbook rank-48
    
    # First, try a direct rank-12 construction using the block structure.
    # 2x4x6 can be viewed as: split 4 into 2+2, 6 into 3+3
    # This gives 4 blocks of 2x2x3, each of rank <= 2*2*3=12 (schoolbook) 
    # but Strassen-like can do 2x2x2 in rank 7.
    # Actually 2x2x3 has rank at most 2*2*3 = 12 schoolbook, but we can do better.
    # 2x2x3: use 2x2x2 (rank 7) + 2x2x1 (rank 2) = 9? Not quite.
    # Let's just try a computational search for rank 12 and 13.
    
    # Try rank 12 with a structured initialization
    for target_rank in [12, 13]:
        if time.time() > deadline:
            break
        result = _search_rank(n, m, p, target_rank, T_expected, start_time, deadline)
        if result is not None:
            return result
    
    # Fall back to Kronecker-based rank 14
    result = _kronecker_solution(n, m, p, T_expected)
    if result is not None:
        return result
    
    # Final fallback: schoolbook
    return _schoolbook_solution(n, m, p)

def _search_rank(n, m, p, target_rank, T_expected, start_time, deadline):
    """
    Search for a rank-r decomposition using a structured approach.
    We use the fact that the tensor has a specific structure to guide the search.
    """
    # Use a parameterized family of decompositions.
    # Key insight: for 2x4x6, we can use the "pencil" structure.
    # The tensor T = sum_{j=0}^{3} A[:,j] ⊗ B[j,:] ⊗ (something)
    # But this is rank 4*6=24.
    
    # Better: use a bilinear form approach.
    # For 2x4x6, the tensor can be written as:
    # T[i,j,k] = A[i,j] * B[j,k]
    # This is a sum of n*m*p = 48 rank-1 terms (schoolbook).
    # We want to reduce this.
    
    # Let's try a direct algebraic construction for rank 12.
    # Idea: group the 24 (j,k) pairs into 12 groups of 2, and use 
    # linear combinations to merge pairs.
    
    # Actually, let's use a more systematic approach:
    # The tensor for 2x4x6 has the following structure:
    # For fixed i, T[i*m+j, j*p+k, k*n+i] = 1
    # This is a "permutation-like" tensor.
    
    # Let's try to construct a rank-12 decomposition explicitly.
    # We'll use the following ansatz:
    # Each rank-1 term corresponds to a "path" through the indices.
    # For 2x4x6, we can think of it as a 2-layer network with 4 middle nodes.
    
    # Direct construction attempt for rank 12:
    # Use the fact that 4 = 2*2 and 6 = 3*2
    # Split j into (j1, j2) with j1 in {0,1}, j2 in {0,1}
    # Split k into (k1, k2) with k1 in {0,1,2}, k2 in {0,1}
    # Then T[i, (j1,j2), (k1,k2)] = A[i, (j1,j2)] * B[(j1,j2), (k1,k2)]
    # This is a Kronecker product structure: T = T_{2,2,3} ⊗ T_{1,2,2} (roughly)
    # Rank of 2x2x3: at most 7 (Strassen for 2x2x2) + 2 (for the extra column) = 9?
    # Actually 2x2x3 has rank at most 2*2*3 = 12, but can be done in rank 7+2=9?
    # Let's just compute it directly.
    
    # For now, use a numerical optimization approach.
    # Initialize with a Kronecker-like structure and try to reduce rank.
    
    # Build a rank-14 base and try to find a rank-12 sub-decomposition
    U14, V14, W14 = _kronecker_matrices(n, m, p)
    if U14 is None:
        return None
    
    # Try to find a rank-reduction by solving for dependencies
    # This is a hard problem; for now, just return the base if target_rank >= 14
    if target_rank >= 14:
        return _format_result(U14, V14, W14, 14, n, m, p)
    
    # For rank 12 or 13, we need a different approach.
    # Let's try a direct construction.
    
    # Rank-12 construction for 2x4x6:
    # We'll use the following idea:
    # C = A * B where A is 2x4, B is 4x6
    # Split A = [A0 | A1] where A0, A1 are 2x2
    # Split B = [B0; B1] where B0, B1 are 2x6
    # C = A0*B0 + A1*B1
    # Each 2x2*2x6 can be done in rank 7*2 = 14 (Strassen x 2)
    # But we need the sum, so total rank is 28. Not helpful.
    
    # Alternative: use the "sum of products" structure more cleverly.
    # For 2x4x6, the minimal rank is conjectured to be 12 or 13.
    # Let's try a specific rank-12 construction.
    
    # Actually, let's just return None and fall back to rank 14.
    # The search for rank 12/13 is too complex to do reliably in this time.
    return None

def _kronecker_matrices(n, m, p):
    """Build the Kronecker product decomposition matrices (rank 14)."""
    # Strassen 2x2x2 (rank 7)
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # Strassen's 7 multiplications for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # M2 = A22*(B21-B11)
    U_s[2, 1] = 1; U_s[3, 1] = 1  # A22
    V_s[0, 1] = -1; V_s[2, 1] = 1  # B21-B11
    W_s[2, 1] = 1; W_s[3, 1] = 1
    
    # M3 = (A11-A22)*(B22+B12)
    U_s[0, 2] = 1; U_s[2, 2] = -1  # A11-A22
    V_s[1, 2] = 1; V_s[3, 2] = 1  # B22+B12
    W_s[1, 2] = 1; W_s[3, 2] = 1
    
    # M4 = A11*(B12-B22)
    U_s[0, 3] = 1  # A11
    V_s[1, 3] = 1; V_s[3, 3] = -1  # B12-B22
    W_s[0, 3] = 1; W_s[1, 3] = 1
    
    # M5 = (A12+A11)*B11
    U_s[0, 4] = 1; U_s[1, 4] = 1  # A12+A11
    V_s[0, 4] = 1  # B11
    W_s[0, 4] = 1; W_s[1, 4] = 1
    
    # M6 = (A21-A11)*(B11+B12)
    U_s[0, 5] = -1; U_s[2, 5] = 1  # A21-A11
    V_s[0, 5] = 1; V_s[1, 5] = 1  # B11+B12
    W_s[2, 5] = 1; W_s[3, 5] = 1
    
    # M7 = (A11+A22)*(B22-B21)
    U_s[0, 6] = 1; U_s[3, 6] = 1  # A11+A22
    V_s[2, 6] = -1; V_s[3, 6] = 1  # B22-B21
    W_s[1, 6] = 1; W_s[2, 6] = 1
    
    # 1x2x3 decomposition (rank 2)
    # For multiplying a 1x2 row by a 2x3 matrix
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # This can be done in rank 2 (schoolbook) or potentially less.
    # Actually 1x2x3 has rank 2 (it's just two scalar multiplications).
    U_123 = np.zeros((2, 2))
    V_123 = np.zeros((6, 2))
    W_123 = np.zeros((3, 2))
    
    U_123[0, 0] = 1  # A[0,0]
    V_123[0, 0] = 1; V_123[1, 0] = 1; V_123[2, 0] = 1  # B[0,:]
    W_123[0, 0] = 1; W_123[1, 0] = 1; W_123[2, 0] = 1  # C[:,0]
    
    U_123[1, 1] = 1  # A[0,1]
    V_123[3, 1] = 1; V_123[4, 1] = 1; V_123[5, 1] = 1  # B[1,:]
    W_123[0, 1] = 1; W_123[1, 1] = 1; W_123[2, 1] = 1  # C[:,1]
    
    # Combine using Kronecker product
    # 2x4x6 = (2x2x2) ⊗ (1x2x3) roughly
    # More precisely: 2x4x6 = (2x2x2) with 4=2*2 and 6=2*3
    # So we have two dimensions: (i1,j1,k1) for 2x2x2 and (i2,j2,k2) for 1x2x3
    # where i = i1*1 + i2, j = j1*2 + j2, k = k1*3 + k2
    
    total_rank = 14  # 7 * 2
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    for r1 in range(7):
        for r2 in range(2):
            r = r1 * 2 + r2
            
            # U: index (i,j) where i = i1 + i2*0 (i2 is always 0 for 1-row)
            # j = j1*2 + j2
            for i1 in range(2):
                for j1 in range(2):
                    us_val = U_s[i1*2 + j1, r1]
                    if us_val == 0:
                        continue
                    for j2 in range(2):
                        u123_val = U_123[j2, r2]  # U_123 has 2 rows (j=0,1)
                        if u123_val == 0:
                            continue
                        i = i1  # i2 is 0
                        j = j1*2 + j2
                        a = i * 4 + j
                        U_full[a, r] += us_val * u123_val
            
            # V: index (j,k) where j = j1*2 + j2, k = k1*3 + k2
            for j1 in range(2):
                for k1 in range(2):
                    vs_val = V_s[j1*2 + k1, r1]
                    if vs_val == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(3):
                            v123_val = V_123[j2*3 + k2, r2]
                            if v123_val == 0:
                                continue
                            j = j1*2 + j2
                            k = k1*3 + k2
                            b = j * 6 + k
                            V_full[b, r] += vs_val * v123_val
            
            # W: index (k,i) where k = k1*3 + k2, i = i1
            for k1 in range(2):
                for i1 in range(2):
                    ws_val = W_s[k1*2 + i1, r1]
                    if ws_val == 0:
                        continue
                    for k2 in range(3):
                        w123_val = W_123[k2, r2]
                        if w123_val == 0:
                            continue
                        k = k1*3 + k2
                        i = i1
                        c = k * 2 + i
                        W_full[c, r] += ws_val * w123_val
    
    return U_full, V_full, W_full

def _format_result(U, V, W, rank, n, m, p):
    """Format the result as a dictionary with list-of-lists."""
    U_out = [[float(U[a, r]) for r in range(rank)] for a in range(n * m)]
    V_out = [[float(V[b, r]) for r in range(rank)] for b in range(m * p)]
    W_out = [[float(W[c, r]) for r in range(rank)] for c in range(p * n)]
    return {"U": U_out, "V": V_out, "W": W_out}

def _kronecker_solution(n, m, p, T_expected):
    """Return the Kronecker product solution if valid."""
    U_full, V_full, W_full = _kronecker_matrices(n, m, p)
    if U_full is None:
        return None
    
    T_comp = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
    if not np.allclose(T_comp, T_expected, atol=1e-10):
        return None
    
    rank = 14
    return _format_result(U_full, V_full, W_full, rank, n, m, p)

def _schoolbook_solution(n, m, p):
    """Fallback schoolbook solution with rank = n*m*p"""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1.0
                V[j*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
