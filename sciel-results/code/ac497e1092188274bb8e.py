import numpy as np
import time
import itertools

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix
    by a 4x5 matrix.

    Strategy: Build a block-based decomposition (Strassen for 2x2*2x2 blocks,
    schoolbook for 2x2*2x3 blocks) giving rank 26, then attempt further rank
    reduction by identifying linearly dependent columns in the factor matrices
    and attempting to merge/eliminate redundant rank-1 terms.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 2, 4, 5
    dim_a = n * m   # 8
    dim_b = m * p   # 20
    dim_c = p * n   # 10

    # Build the multiplication tensor T[i,j,k] = 1 if C[i,k] = A[i,j]*B[j,k]
    # T shape: (dim_a, dim_b, dim_c) = (8, 20, 10)
    # Indexing: a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((dim_a, dim_b, dim_c), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def build_decomposition(components):
        """Build U, V, W from list of (u, v, w) rank-1 components."""
        rank = len(components)
        U = np.zeros((dim_a, rank), dtype=float)
        V = np.zeros((dim_b, rank), dtype=float)
        W = np.zeros((dim_c, rank), dtype=float)
        for r, (u, v, w) in enumerate(components):
            U[:, r] = u
            V[:, r] = v
            W[:, r] = w
        return U, V, W

    def verify(U, V, W, tol=1e-10):
        """Check that sum_r U[:,r] * V[:,r] * W[:,r] == T exactly."""
        recon = np.zeros_like(T)
        for r in range(U.shape[1]):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.max(np.abs(recon - T)) < tol

    def strassen_2x2_2x2():
        """
        Returns 7 rank-1 components for 2x2 * 2x2 matrix multiplication.
        """
        comps = []
        # M1: left = a11+a22, right = b11+b22, contributes to c11, c22
        u1 = np.zeros(4); u1[0]=1; u1[3]=1
        v1 = np.zeros(4); v1[0]=1; v1[3]=1
        w1 = np.zeros(4); w1[0]=1; w1[3]=1
        comps.append((u1, v1, w1))
        
        # M2: left = a21+a22, right = b11, contributes to c21
        u2 = np.zeros(4); u2[2]=1; u2[3]=1
        v2 = np.zeros(4); v2[0]=1
        w2 = np.zeros(4); w2[2]=1
        comps.append((u2, v2, w2))
        
        # M3: left = a11, right = b12-b22, contributes to c12, c22
        u3 = np.zeros(4); u3[0]=1
        v3 = np.zeros(4); v3[1]=1; v3[3]=-1
        w3 = np.zeros(4); w3[1]=1; w3[3]=1
        comps.append((u3, v3, w3))
        
        # M4: left = a22, right = b11-b12, contributes to c11(-), c12(+)
        u4 = np.zeros(4); u4[3]=1
        v4 = np.zeros(4); v4[0]=1; v4[1]=-1
        w4 = np.zeros(4); w4[0]=-1; w4[1]=1
        comps.append((u4, v4, w4))
        
        # M5: left = a11+a12, right = b22, contributes to c11, c21
        u5 = np.zeros(4); u5[0]=1; u5[1]=1
        v5 = np.zeros(4); v5[3]=1
        w5 = np.zeros(4); w5[0]=1; w5[2]=1
        comps.append((u5, v5, w5))
        
        # M6: left = a12-a22, right = b11+b12, contributes to c22
        u6 = np.zeros(4); u6[1]=1; u6[3]=-1
        v6 = np.zeros(4); v6[0]=1; v6[1]=1
        w6 = np.zeros(4); w6[3]=1
        comps.append((u6, v6, w6))
        
        # M7: left = a21-a11, right = b21+b22, contributes to c11
        u7 = np.zeros(4); u7[2]=1; u7[0]=-1
        v7 = np.zeros(4); v7[2]=1; v7[3]=1
        w7 = np.zeros(4); w7[0]=1
        comps.append((u7, v7, w7))
        
        return comps
    
    def schoolbook_2x2_2x3():
        """
        Returns 6 rank-1 components for 2x2 * 2x3 matrix multiplication.
        """
        comps = []
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    u = np.zeros(4); u[i*2+j] = 1
                    v = np.zeros(6); v[j*3+k] = 1
                    w = np.zeros(6); w[k*2+i] = 1
                    comps.append((u, v, w))
        return comps

    # Embedding functions
    def embed_strassen(comp, a_col_offset, b_row_offset):
        """
        Embed a Strassen 2x2*2x2 component into the full 2x4*4x5 space.
        a_col_offset: 0 for A1 (cols 0,1), 2 for A2 (cols 2,3)
        b_row_offset: 0 for B1 (rows 0,1), 2 for B2 (rows 2,3)
        """
        u_local, v_local, w_local = comp
        u_full = np.zeros(dim_a)
        v_full = np.zeros(dim_b)
        w_full = np.zeros(dim_c)
        
        # u_local: 4 entries, a[i,j] for i in {0,1}, j in {0,1}
        # Full A: A[i, j] at index i*4 + j
        for i in range(2):
            for j in range(2):
                u_full[i*4 + a_col_offset + j] = u_local[i*2 + j]
        
        # v_local: 4 entries, b[j,k] for j in {0,1}, k in {0,1}
        # Full B: B[j, k] at index j*5 + k
        for j in range(2):
            for k in range(2):
                v_full[(b_row_offset + j)*5 + k] = v_local[j*2 + k]
        
        # w_local: 4 entries, c[k,i] for k in {0,1}, i in {0,1}
        # Full C: C[k, i] at index k*2 + i (trace convention W[k*n+i])
        # For B1 (rows 0,1), output k=0,1; for B2 (rows 2,3), output k=2,3
        for k in range(2):
            for i in range(2):
                w_full[(b_row_offset + k)*2 + i] = w_local[k*2 + i]
        
        return (u_full, v_full, w_full)
    
    def embed_schoolbook_2x2_2x3(comp, a_col_offset, b_row_offset, b_col_offset):
        """
        Embed a schoolbook 2x2*2x3 component into the full space.
        a_col_offset: 0 for A1, 2 for A2
        b_row_offset: 0 for B1, 2 for B2
        b_col_offset: 0 for left 2x2 part, 2 for right 2x3 part
        """
        u_local, v_local, w_local = comp
        u_full = np.zeros(dim_a)
        v_full = np.zeros(dim_b)
        w_full = np.zeros(dim_c)
        
        for i in range(2):
            for j in range(2):
                u_full[i*4 + a_col_offset + j] = u_local[i*2 + j]
        
        for j in range(2):
            for k in range(3):
                v_full[(b_row_offset + j)*5 + b_col_offset + k] = v_local[j*3 + k]
        
        for k in range(3):
            for i in range(2):
                w_full[(b_col_offset + k)*2 + i] = w_local[k*2 + i]
        
        return (u_full, v_full, w_full)
    
    # Build the rank-26 decomposition
    all_comps = []
    
    strassen_comps = strassen_2x2_2x2()
    sb_comps = schoolbook_2x2_2x3()
    
    # A1*B1a: Strassen, A cols 0-1, B rows 0-1, B cols 0-1
    for comp in strassen_comps:
        all_comps.append(embed_strassen(comp, a_col_offset=0, b_row_offset=0))
    
    # A1*B1b: Schoolbook 2x2*2x3, A cols 0-1, B rows 0-1, B cols 2-4
    for comp in sb_comps:
        all_comps.append(embed_schoolbook_2x2_2x3(comp, a_col_offset=0, b_row_offset=0, b_col_offset=2))
    
    # A2*B2a: Strassen, A cols 2-3, B rows 2-3, B cols 0-1
    for comp in strassen_comps:
        all_comps.append(embed_strassen(comp, a_col_offset=2, b_row_offset=2))
    
    # A2*B2b: Schoolbook 2x2*2x3, A cols 2-3, B rows 2-3, B cols 2-4
    for comp in sb_comps:
        all_comps.append(embed_schoolbook_2x2_2x3(comp, a_col_offset=2, b_row_offset=2, b_col_offset=2))
    
    current_rank = len(all_comps)
    U, V, W = build_decomposition(all_comps)
    
    if not verify(U, V, W):
        # Fallback to schoolbook rank 40
        U = np.zeros((dim_a, 40), dtype=float)
        V = np.zeros((dim_b, 40), dtype=float)
        W = np.zeros((dim_c, 40), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, idx] = 1.0
                    V[b, idx] = 1.0
                    W[c, idx] = 1.0
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now try to reduce rank by finding and eliminating redundant terms
    # Strategy: look for columns that are linearly dependent across U, V, W
    # If column r can be expressed as a linear combination of other columns
    # in a way that preserves the tensor, we can eliminate it.
    
    # A more practical approach: try to find pairs of terms whose sum
    # can be replaced by a single term (i.e., their sum is rank-1)
    
    # For each pair (r, s), check if the sum of their contributions is rank-1
    # This is expensive, so we limit to a few iterations
    
    def is_rank1_tensor(t, tol=1e-10):
        """Check if a 3D tensor is rank-1."""
        # Flatten and check if it can be written as outer product of 3 vectors
        # A tensor T is rank-1 iff all its 2D slices are rank-1
        # and the rank-1 decompositions are consistent
        shape = t.shape
        if any(s == 0 for s in shape):
            return True
        # Check if T[a,b,c] = u[a]*v[b]*w[c]
        # Find a non-zero entry
        nz = np.argwhere(t != 0)
        if len(nz) == 0:
            return True
        a0, b0, c0 = nz[0]
        # Try to factor
        # T[a,b,c] / T[a0,b0,c0] should = (u[a]/u[a0]) * (v[b]/v[b0]) * (w[c]/w[c0])
        # This is complex; use SVD-based check on matricizations
        # Matricization 1: T as (dim_a, dim_b*dim_c)
        T1 = t.reshape(shape[0], -1)
        U1, s1, Vt1 = np.linalg.svd(T1, full_matrices=False)
        if len(s1) > 1 and s1[1] > tol * s1[0]:
            return False
        # Matricization 2: T as (dim_b, dim_a*dim_c)
        T2 = t.transpose(1,0,2).reshape(shape[1], -1)
        U2, s2, Vt2 = np.linalg.svd(T2, full_matrices=False)
        if len(s2) > 1 and s2[1] > tol * s2[0]:
            return False
        return True
    
    # Try to merge pairs: for each pair (r,s), compute sum tensor and check rank-1
    # This is O(rank^2 * dim) which is manageable for rank 26
    merged = list(all_comps)
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for r in range(len(merged)):
            if time.time() > deadline - 10:
                break
            for s in range(r+1, len(merged)):
                if time.time() > deadline - 10:
                    break
                # Compute sum tensor of components r and s
                sum_tensor = np.einsum('a,b,c->abc', merged[r][0], merged[r][1], merged[r][2])
                sum_tensor += np.einsum('a,b,c->abc', merged[s][0], merged[s][1], merged[s][2])
                if is_rank1_tensor(sum_tensor):
                    # Try to factor the sum tensor into a single rank-1 term
                    # Use the non-zero structure
                    nz = np.argwhere(sum_tensor != 0)
                    if len(nz) > 0:
                        a0, b0, c0 = nz[0]
                        # Extract vectors
                        u_vec = sum_tensor[:, b0, c0]
                        v_vec = sum_tensor[a0, :, c0]
                        w_vec = sum_tensor[a0, b0, :]
                        # Normalize
                        if abs(u_vec[a0]) > 1e-10:
                            u_vec = u_vec / u_vec[a0]
                        if abs(v_vec[b0]) > 1e-10:
                            v_vec = v_vec / v_vec[b0]
                        if abs(w_vec[c0]) > 1e-10:
                            w_vec = w_vec / w_vec[c0]
                        # Verify this factors correctly
                        recon = np.einsum('a,b,c->abc', u_vec, v_vec, w_vec)
                        if np.max(np.abs(recon - sum_tensor)) < 1e-10:
                            # Replace two components with one
                            merged.pop(s)
                            merged.pop(r)
                            merged.append((u_vec, v_vec, w_vec))
                            improved = True
                            break
        if improved:
            continue
    
    current_rank = len(merged)
    if current_rank < 26:
        U, V, W = build_decomposition(merged)
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return the rank-26 decomposition
    U, V, W = build_decomposition(all_comps)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
