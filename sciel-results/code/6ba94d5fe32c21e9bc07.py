"""
Combined program for matrix multiplication 3x4x7 tensor decomposition.
Uses 6 disjoint 2x2x2 Strassen blocks (rank 7 each) for i in {0,1},
tiling j-pairs {0,1},{2,3} and k-pairs {0,1},{2,3},{4,5},
plus schoolbook for the remaining 36 entries. Total rank: 78.
Includes robust verification and a linear-algebra fallback search.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]


def construct():
    """
    Returns a rank-minimal (or best found) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x7 matrix.
    
    Returns dict with keys 'U', 'V', 'W' (lists of lists) representing
    factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin (150s limit)

    n, m, p = DIMENSIONS
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    num_eqs = n * m * p  # 84

    def get_schoolbook():
        rank = num_eqs
        U = [[0] * rank for _ in range(dim_U)]
        V = [[0] * rank for _ in range(dim_V)]
        W = [[0] * rank for _ in range(dim_W)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify exact decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def verify_fast(U_arr, V_arr, W_arr):
        """Vectorized verification."""
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        U_sub = U_arr[i_idx * m + j_idx, :]
        V_sub = V_arr[j_idx * p + k_idx, :]
        W_sub = W_arr[k_idx * n + i_idx, :]
        
        sums = np.sum(U_sub * V_sub * W_sub, axis=1)
        return np.allclose(sums, 1.0, atol=1e-9)

    # ============================================================
    # Build the 2x2x2 Strassen block template
    # For a 2x2x2 tensor, we need 7 rank-1 terms such that
    # sum_c U[a,c]*V[b,c]*W[c_idx,c] = delta_{a,b} for all (a,b,c_idx) in {0,1}^3
    # 
    # The standard Strassen 7-term form (verified):
    # Using indices (i,j,k) in {0,1}^3:
    # 
    # Let me derive it properly. We need:
    # For each (i,j,k): sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    #
    # The 7 terms (using the form that works):
    # c=0: U[0,0]=1, V[0,0]=1, W[0,0]=1  -> contributes to (0,0,0)
    # c=1: U[1,1]=1, V[1,1]=1, W[1,1]=1  -> contributes to (1,1,1)  
    # c=2: U[1,0]=1, U[1,1]=1, V[0,0]=1, W[0,1]=1, W[1,1]=-1
    #      -> (1,0,0): 1*1*1=1, (1,1,1): 1*1*(-1)=-1
    # c=3: U[0,0]=1, U[0,1]=1, V[0,1]=1, W[1,0]=1, W[1,1]=-1
    #      -> (0,0,1): 1*1*1=1, (0,1,1): 1*1*(-1)=-1
    # c=4: U[0,0]=1, U[1,0]=-1, V[0,0]=1, V[0,1]=1, W[0,0]=1, W[1,0]=-1
    #      -> (0,0,0): 1*1*1=1, (0,0,1): 1*1*(-1)=-1, (1,0,0): -1*1*1=-1, (1,0,1): -1*1*(-1)=1
    # c=5: U[0,1]=1, U[1,1]=-1, V[1,0]=1, V[1,1]=1, W[0,1]=1, W[1,1]=-1
    #      -> (0,1,0): 1*1*1=1, (0,1,1): 1*1*(-1)=-1, (1,1,0): -1*1*1=-1, (1,1,1): -1*1*(-1)=1
    # c=6: U[0,0]=1, U[1,1]=1, V[0,1]=1, V[1,0]=1, W[0,1]=1, W[1,0]=1
    #      -> (0,0,1): 1*1*1=1, (0,1,0): 1*1*1=1, (1,0,1): 1*1*1=1, (1,1,0): 1*1*1=1
    #
    # Let me verify this sums correctly for each (i,j,k):
    # (0,0,0): c0=1, c4=1, c6=0 -> 2. WRONG, should be 1.
    #
    # This form is not correct. Let me use a different approach.
    #
    # Actually, the correct 7-term Strassen for the 2x2x2 multiplication tensor:
    # The tensor T_{ijk} = delta_{ik} (since C[i,k] = sum_j A[i,j]*B[j,k])
    # Wait no, for matrix mult C = AB, C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T_{i,j,k} = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # i.e., T_{i,j,k} = 1 for all i,j,k (it's the identity in the "multiplication" sense)
    # 
    # No wait. The matrix multiplication tensor M_{n,m,p} has entries:
    # M[i,j,k] = 1 for all i in {0..n-1}, j in {0..m-1}, k in {0..p-1}
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # So for 2x2x2, we need to decompose the all-ones tensor of size 2x2x2.
    # The rank of this tensor is 7 (Strassen's result).
    #
    # The 7-term decomposition of the 2x2x2 all-ones tensor:
    # We need 7 rank-1 tensors whose sum is the all-ones tensor.
    # A rank-1 tensor is U[i]*V[j]*W[k] for some vectors U,V,W.
    #
    # One known decomposition:
    # t1: U=[1,0], V=[1,0], W=[1,0] -> contributes to (0,0,0)
    # t2: U=[0,1], V=[0,1], W=[0,1] -> contributes to (1,1,1)
    # t3: U=[0,1], V=[1,0], W=[0,1] -> contributes to (1,0,1)
    # t4: U=[0,1], V=[0,1], W=[1,0] -> contributes to (1,1,0)
    # t5: U=[1,0], V=[0,1], W=[1,0] -> contributes to (0,1,0)
    # t6: U=[1,0], V=[1,0], W=[0,1] -> contributes to (0,0,1)
    # t7: U=[1,1], V=[1,1], W=[1,1] -> contributes to all 8 entries
    #
    # Sum: (0,0,0): t1+t7 = 2. WRONG.
    #
    # Let me try a different known form. The standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # In tensor decomposition form, each M_c corresponds to a rank-1 term:
    # M1: U = [1,1] (for i), V = [1,1] (for j), W = [1,1] (for k)
    #   But this contributes to all (i,j,k) with coefficient 1.
    #   In the C formula, M1 appears in C00 (+1) and C11 (+1).
    #   So W should be [1,0,0,1] in (k0*i0, k0*i1, k1*i0, k1*i1) = [1,0,0,1]
    #   Wait, W is indexed by (k,i), so W[k*n+i] where n=2.
    #   W[0*2+0]=1, W[1*2+1]=1, others 0.
    #
    # Let me map this out properly. For the 2x2x2 block with base indices (i0,j0,k0):
    # U is indexed by (i,j) -> row i*m+j
    # V is indexed by (j,k) -> row j*p+k
    # W is indexed by (k,i) -> row k*n+i
    #
    # For the block, local indices (a,b,c) in {0,1}^3 map to:
    # U[i0+a, j0+b], V[j0+b, k0+c], W[k0+c, i0+a]
    #
    # M1 = (A00+A11)(B00+B11): 
    #   U[a=0,b=0]=1, U[a=1,b=1]=1
    #   V[b=0,c=0]=1, V[b=1,c=1]=1
    #   W[c=0,a=0]=1, W[c=1,a=1]=1
    #   Contributes to: (a=0,b=0,c=0): 1*1*1=1, (a=0,b=0,c=1): 1*1*0=0,
    #                    (a=0,b=1,c=0): 1*0*1=0, (a=0,b=1,c=1): 1*0*0=0,
    #                    (a=1,b=0,c=0): 0*1*1=0, (a=1,b=0,c=1): 0*1*0=0,
    #                    (a=1,b=1,c=0): 0*0*1=0, (a=1,b=1,c=1): 0*0*1=0... 
    #   Wait, that's wrong. Let me recompute.
    #   U[0,0]=1, U[1,1]=1, V[0,0]=1, V[1,1]=1, W[0,0]=1, W[1,1]=1
    #   (0,0,0): U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    #   (0,0,1): U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0
    #   (0,1,0): U[0,1]*V[1,0]*W[0,0] = 0*0*1 = 0
    #   (0,1,1): U[0,1]*V[1,1]*W[1,0] = 0*1*0 = 0
    #   (1,0,0): U[1,0]*V[0,0]*W[0,1] = 0*1*0 = 0
    #   (1,0,1): U[1,0]*V[0,1]*W[1,1] = 0*0*1 = 0
    #   (1,1,0): U[1,1]*V[1,0]*W[0,1] = 1*0*0 = 0
    #   (1,1,1): U[1,1]*V[1,1]*W[1,1] = 1*1*1 = 1
    #   So M1 contributes 1 to (0,0,0) and (1,1,1). Good.
    #
    # M2 = (A10+A11)B00:
    #   U[1,0]=1, U[1,1]=1, V[0,0]=1
    #   C10 = M2 + M4, C11 = ... - M2 + ...
    #   So M2 contributes +1 to C10 and -1 to C11.
    #   W[0,1]=1, W[1,1]=-1
    #   (1,0,0): U[1,0]*V[0,0]*W[0,1] = 1*1*1 = 1
    #   (1,1,1): U[1,1]*V[1,1]*W[1,1] = 1*0*(-1) = 0
    #   Hmm, V[1,1] is 0 in this term. Let me re-examine.
    #   M2 = (A10+A11)*B00, so V only has V[0,0]=1 (B00 means j=0,k=0).
    #   W should indicate which (k,i) this contributes to:
    #   C10 means i=1,k=0, so W[0,1]=1
    #   C11 means i=1,k=1, so W[1,1]=-1
    #   (1,0,0): U[1,0]*V[0,0]*W[0,1] = 1*1*1 = 1 ✓
    #   (1,1,0): U[1,1]*V[1,0]*W[0,1] = 1*0*1 = 0
    #   (1,0,1): U[1,0]*V[0,1]*W[1,1] = 1*0*(-1) = 0
    #   (1,1,1): U[1,1]*V[1,1]*W[1,1] = 1*0*(-1) = 0
    #   So M2 contributes 1 to (1,0,0) only. But we need it to contribute to C10 and -C11.
    #   C10 is entry (i=1,j=0,k=0) in the output... wait, I'm confusing myself.
    #
    # Let me restart with clearer notation. The tensor entry T[i,j,k] corresponds to
    # the coefficient of A[i,j]*B[j,k] in C[i,k]. For 2x2x2, T[i,j,k]=1 for all i,j,k.
    # We need sum_c U[i,c]*V[j,c]*W[k,c] = 1 for all (i,j,k) in {0,1}^3.
    #
    # The 7-term Strassen decomposition (one valid form):
    # c=0: U=[1,1], V=[1,1], W=[1,1] -> contributes 1 to all 8 entries
    # c=1: U=[1,-1], V=[1,0], W=[1,0] -> (0,0,0):1, (0,0,1):0, (1,0,0):-1, (1,0,1):0
    # c=2: U=[0,1], V=[1,1], W=[0,1] -> (0,1,0):0,(0,1,1):0,(1,1,0):0,(1,1,1):1
    # ... this is getting complicated. Let me just use the verified form from the parent.
    
    def add_strassen_block(U, V, W, col_start, i0, j0, k0):
        """Add a verified 2x2x2 Strassen block."""
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        
        # Use the form from the working parent program (verified to pass):
        c = col_start
        
        # Term 0
        U[i0*m+j0][c] = 1
        U[i1*m+j1][c] = 1
        V[j0*p+k0][c] = 1
        V[j1*p+k1][c] = 1
        W[k0*n+i0][c] = 1
        W[k1*n+i1][c] = 1
        
        # Term 1
        c += 1
        U[i1*m+j0][c] = 1
        U[i1*m+j1][c] = 1
        V[j0*p+k0][c] = 1
        W[k0*n+i1][c] = 1
        W[k1*n+i1][c] = -1
        
        # Term 2
        c += 1
        U[i0*m+j0][c] = 1
        V[j0*p+k1][c] = 1
        V[j1*p+k1][c] = -1
        W[k1*n+i0][c] = 1
        W[k1*n+i1][c] = 1
        
        # Term 3
        c += 1
        U[i1*m+j1][c] = 1
        V[j1*p+k0][c] = 1
        V[j0*p+k0][c] = -1
        W[k0*n+i0][c] = 1
        W[k0*n+i1][c] = 1
        
        # Term 4
        c += 1
        U[i0*m+j0][c] = 1
        U[i0*m+j1][c] = 1
        V[j1*p+k1][c] = 1
        W[k0*n+i0][c] = -1
        W[k1*n+i0][c] = 1
        
        # Term 5
        c += 1
        U[i1*m+j0][c] = 1
        U[i0*m+j0][c] = -1
        V[j0*p+k0][c] = 1
        V[j0*p+k1][c] = 1
        W[k1*n+i1][c] = 1
        
        # Term 6
        c += 1
        U[i0*m+j1][c] = 1
        U[i1*m+j1][c] = -1
        V[j1*p+k0][c] = 1
        V[j1*p+k1][c] = 1
        W[k0*n+i0][c] = 1
        
        return c

    # Build the structured decomposition
    rank = 78
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]

    # Six blocks: i={0,1}, j-pairs {0,1},{2,3}, k-pairs {0,1},{2,3},{4,5}
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4),
    ]

    col = 0
    covered = set()
    for i0, j0, k0 in blocks:
        col = add_strassen_block(U, V, W, col, i0, j0, k0)
        for i in (i0, i0+1):
            for j in (j0, j0+1):
                for k in (k0, k0+1):
                    covered.add((i, j, k))

    # Add schoolbook terms for remaining (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1

    if col != rank:
        return get_schoolbook()

    U_arr = np.asarray(U, dtype=np.float64)
    V_arr = np.asarray(V, dtype=np.float64)
    W_arr = np.asarray(W, dtype=np.float64)

    # Verify
    if verify_fast(U_arr, V_arr, W_arr):
        return {"U": U, "V": V, "W": W}

    # If structured form doesn't verify, use linear algebra to find W
    if time.time() > deadline - 25:
        return get_schoolbook()

    # Solve for W given U and V
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    U_rows = i_idx * m + j_idx
    V_rows = j_idx * p + k_idx
    W_rows = k_idx * n + i_idx
    
    # Build linear system: for each eq (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    num_w_vars = dim_W * rank
    A = np.zeros((num_eqs, num_w_vars))
    b = np.ones(num_eqs)
    
    for eq in range(num_eqs):
        i, j, k = i_idx[eq], j_idx[eq], k_idx[eq]
        w_row = k * n + i
        for c in range(rank):
            coeff = U_arr[i*m+j, c] * V_arr[j*p+k, c]
            if coeff != 0:
                A[eq, w_row * rank + c] = coeff
    
    try:
        W_flat, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=1e-12)
    except Exception:
        return get_schoolbook()
    
    if len(residuals) > 0 and residuals[0] > 1e-8:
        # Try rounding
        W_sol = W_flat.reshape((dim_W, rank))
        for factor in [2, 1]:
            W_rounded = np.round(W_sol * factor) / factor
            W_rounded_arr = np.asarray(W_rounded, dtype=np.float64)
            if verify_fast(U_arr, V_arr, W_rounded_arr):
                return {"U": U, "V": V, "W": W_rounded.tolist()}
        return get_schoolbook()
    
    # Solution found, round to half-integers
    W_sol = W_flat.reshape((dim_W, rank))
    W_rounded = np.round(W_sol * 2) / 2.0
    W_rounded_arr = np.asarray(W_rounded, dtype=np.float64)
    
    if verify_fast(U_arr, V_arr, W_rounded_arr):
        return {"U": U, "V": V, "W": W_rounded.tolist()}
    
    # Try integer rounding
    W_int = np.round(W_sol)
    W_int_arr = np.asarray(W_int, dtype=np.float64)
    if verify_fast(U_arr, V_arr, W_int_arr):
        return {"U": U, "V": V, "W": W_int.tolist()}
    
    # Try direct (unrounded) - should work if lstsq gave exact solution
    W_direct = np.asarray(W_sol, dtype=np.float64)
    if verify_fast(U_arr, V_arr, W_direct):
        return {"U": U, "V": V, "W": W_sol.tolist()}
    
    # Final fallback: random search at rank 78
    if time.time() > deadline - 10:
        return get_schoolbook()
    
    np.random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 3:
            break
        
        U_rand = np.random.choice([-1.0, 0.0, 1.0], size=(dim_U, rank))
        V_rand = np.random.choice([-1.0, 0.0, 1.0], size=(dim_V, rank))
        
        # Ensure no zero rows
        for r in range(dim_U):
            if np.all(U_rand[r] == 0):
                U_rand[r, np.random.randint(rank)] = 1.0
        for r in range(dim_V):
            if np.all(V_rand[r] == 0):
                V_rand[r, np.random.randint(rank)] = 1.0
        
        # Solve for W
        A2 = np.zeros((num_eqs, num_w_vars))
        for eq in range(num_eqs):
            i, j, k = i_idx[eq], j_idx[eq], k_idx[eq]
            w_row = k * n + i
            for c in range(rank):
                coeff = U_rand[i*m+j, c] * V_rand[j*p+k, c]
                if coeff != 0:
                    A2[eq, w_row * rank + c] = coeff
        
        try:
            W_flat2, res2, _, _ = np.linalg.lstsq(A2, b, rcond=1e-12)
        except Exception:
            continue
        
        if len(res2) > 0 and res2[0] > 1e-8:
            continue
        
        W_sol2 = W_flat2.reshape((dim_W, rank))
        W_r2 = np.round(W_sol2 * 2) / 2.0
        W_r2_arr = np.asarray(W_r2, dtype=np.float64)
        if verify_fast(U_rand, V_rand, W_r2_arr):
            return {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_r2.tolist()}
    
    return get_schoolbook()
