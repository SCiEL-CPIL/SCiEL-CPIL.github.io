import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n, m, p, atol=1e-9):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    T = np.einsum('ar,br,cr->abc', U, V, W)
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    return np.allclose(T, target, atol=atol)

def _build_trivial(n, m, p):
    """Trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                r += 1
    return U, V, W

def _strassen_222():
    """
    Return the 7 rank-1 terms for 2x2x2 Strassen.
    Each term is (u, v, w) where u, v, w are length-4 vectors.
    u[i] corresponds to A[i//2, i%2], v[j] to B[j//2, j%2], w[k] to C[k//2, k%2].
    """
    terms = []
    # m1 = (a+d)(e+h), contributes to C[0,0] and C[1,1]
    terms.append((np.array([1,0,0,1.]), np.array([1,0,0,1.]), np.array([1,0,0,1.])))
    # m2 = (c+d)e, contributes to C[1,0]
    terms.append((np.array([0,0,1,1.]), np.array([1,0,0,0.]), np.array([0,0,1,0.])))
    # m3 = a(f-h), contributes to C[0,1]
    terms.append((np.array([1,0,0,0.]), np.array([0,1,0,-1.]), np.array([0,1,0,0.])))
    # m4 = d(g-f), contributes to C[0,0] and C[1,0]
    terms.append((np.array([0,0,0,1.]), np.array([0,-1,1,0.]), np.array([1,0,1,0.])))
    # m5 = (a+b)h, contributes to C[0,1] and -C[1,0]
    terms.append((np.array([1,1,0,0.]), np.array([0,0,0,1.]), np.array([0,1,-1,0.])))
    # m6 = (c-a)(e+f), contributes to C[1,1]
    terms.append((np.array([-1,0,1,0.]), np.array([1,1,0,0.]), np.array([0,0,0,1.])))
    # m7 = (b-d)(g+h), contributes to C[0,0]
    terms.append((np.array([0,1,0,-1.]), np.array([0,0,1,1.]), np.array([1,0,0,0.])))
    return terms

def _build_strassen_block(n, m, p):
    """
    Build decomposition using Strassen for 2x2x2 blocks and schoolbook for 2x2x1 blocks.
    Strategy: A(2x4) = [A1 | A2] (each 2x2), B(4x7) = [B1; B2] (each 2x7).
    C = A1*B1 + A2*B2.
    Each 2x7 = three 2x2 blocks + one 2x1 block.
    2x2x2: rank 7 (Strassen). 2x2x1: rank 4 (schoolbook).
    Total per 2x7: 3*7 + 4 = 25.
    Total: 2 * 25 = 50.
    """
    strassen_terms = _strassen_222()
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_local, v_local, w_local, u_map, v_map, w_map):
        u_g = np.zeros(8)  # n*m = 2*4
        v_g = np.zeros(28) # m*p = 4*7
        w_g = np.zeros(14) # p*n = 7*2
        for idx, val in enumerate(u_local):
            if val != 0:
                u_g[u_map[idx]] = val
        for idx, val in enumerate(v_local):
            if val != 0:
                v_g[v_map[idx]] = val
        for idx, val in enumerate(w_local):
            if val != 0:
                w_g[w_map[idx]] = val
        U_terms.append(u_g)
        V_terms.append(v_g)
        W_terms.append(w_g)
    
    # U mappings for A1 (cols 0-1) and A2 (cols 2-3)
    u_map_A1 = [0, 1, 2, 3]
    u_map_A2 = [4, 5, 6, 7]
    
    # V mappings for B1 (rows 0-1) and B2 (rows 2-3)
    # B1 is 2x7, split into 3x(2x2) + 1x(2x1)
    # B1 block c (2x2): V indices are (0*2+0, 0*2+1, 1*2+0, 1*2+1) + c*4
    # i.e., (2c, 2c+1, 2c+2, 2c+3) for row 0 and row 1 of the 2x2 block
    # But wait: V index = j*p + k where j is row of B (0..3), k is col of B (0..6)
    # For B1 (rows 0-1), row j in B1 corresponds to global row j.
    # Block 0 (cols 0-1): V indices (0*7+0, 0*7+1, 1*7+0, 1*7+1) = (0, 1, 7, 8)
    # Block 1 (cols 2-3): V indices (0, 1, 7, 8) + 2 = (2, 3, 9, 10)
    # Block 2 (cols 4-5): V indices = (4, 5, 11, 12)
    # Block 3 (col 6): V indices = (6, 13)  [only 2x1]
    
    v_map_B1_0 = [0, 1, 7, 8]
    v_map_B1_1 = [2, 3, 9, 10]
    v_map_B1_2 = [4, 5, 11, 12]
    v_map_B1_3 = [6, 13]
    
    v_map_B2_0 = [14, 15, 21, 22]
    v_map_B2_1 = [16, 17, 23, 24]
    v_map_B2_2 = [18, 19, 25, 26]
    v_map_B2_3 = [20, 27]
    
    # W mappings: W index = k*n + i = k*2 + i
    # For a 2x2 block of C covering columns k0, k0+1:
    # C[0,k0] -> k0*2+0, C[0,k0+1] -> (k0+1)*2+0
    # C[1,k0] -> k0*2+1, C[1,k0+1] -> (k0+1)*2+1
    # Local indices in Strassen: 0=C[0,0], 1=C[0,1], 2=C[1,0], 3=C[1,1]
    # So local 0 -> k0*2+0, local 1 -> (k0+1)*2+0, local 2 -> k0*2+1, local 3 -> (k0+1)*2+1
    
    w_map_0 = [0, 2, 1, 3]      # k0=0: [0, 2, 1, 3]
    w_map_1 = [4, 6, 5, 7]      # k0=2: [4, 6, 5, 7]
    w_map_2 = [8, 10, 9, 11]    # k0=4: [8, 10, 9, 11]
    w_map_3 = [12, 13]           # k0=6, 2x1: [12, 13]
    
    # Schoolbook 2x2x1 terms
    def schoolbook_221():
        terms = []
        terms.append((np.array([1,0,0,0.]), np.array([1,0.]), np.array([1,0.])))
        terms.append((np.array([0,1,0,0.]), np.array([0,1.]), np.array([1,0.])))
        terms.append((np.array([0,0,1,0.]), np.array([1,0.]), np.array([0,1.])))
        terms.append((np.array([0,0,0,1.]), np.array([0,1.]), np.array([0,1.])))
        return terms
    
    sb_terms = schoolbook_221()
    
    # A1 * B1
    for r in range(7):
        u, v, w = strassen_terms[r]
        add_term(u, v, w, u_map_A1, v_map_B1_0, w_map_0)
    for r in range(7):
        u, v, w = strassen_terms[r]
        add_term(u, v, w, u_map_A1, v_map_B1_1, w_map_1)
    for r in range(7):
        u, v, w = strassen_terms[r]
        add_term(u, v, w, u_map_A1, v_map_B1_2, w_map_2)
    for r in range(4):
        u, v, w = sb_terms[r]
        add_term(u, v, w, u_map_A1, v_map_B1_3, w_map_3)
    
    # A2 * B2
    for r in range(7):
        u, v, w = strassen_terms[r]
        add_term(u, v, w, u_map_A2, v_map_B2_0, w_map_0)
    for r in range(7):
        u, v, w = strassen_terms[r]
        add_term(u, v, w, u_map_A2, v_map_B2_1, w_map_1)
    for r in range(7):
        u, v, w = strassen_terms[r]
        add_term(u, v, w, u_map_A2, v_map_B2_2, w_map_2)
    for r in range(4):
        u, v, w = sb_terms[r]
        add_term(u, v, w, u_map_A2, v_map_B2_3, w_map_3)
    
    U = np.column_stack(U_terms)
    V = np.column_stack(V_terms)
    W = np.column_stack(W_terms)
    return U, V, W

def _try_als(n, m, p, rank, max_iters=500, tol=1e-10, seed=0):
    """
    Attempt to find a rank decomposition via ALS (Alternating Least Squares).
    The tensor is T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
    We minimize ||sum_r u_r v_r w_r - T||_F.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(n*m, rank) * 0.1
    V = rng.randn(m*p, rank) * 0.1
    W = rng.randn(p*n, rank) * 0.1
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    for it in range(max_iters):
        # Update U
        for r in range(rank):
            # U[:,r] minimizes ||sum_r' U[:,r'] * (V[:,r']^T * W[:,r']) - T||
            # For fixed r: U[:,r] * (V[:,r]^T * W[:,r]) should approximate T
            # Residual excluding r: R = T - sum_{r'!=r} U[:,r'] * (V[:,r']^T * W[:,r'])
            # U[:,r] = argmin ||R - U[:,r] * (V[:,r]^T * W[:,r])||
            # This is: U[:,r] = (V[:,r]^T * W[:,r])^T * R flattened / ||V[:,r]^T * W[:,r]||^2
            # More precisely: reshape R to (n*m, -1), then U[:,r] = R_flat @ (V[:,r] * W[:,r]) / ||V[:,r]*W[:,r]||^2
            
            vw = V[:, r] * W[:, r]  # outer product factors
            # R = T - sum_{r'!=r} outer(U[:,r'], outer(V[:,r'], W[:,r']))
            # For efficiency, compute residual once and subtract each term
            # But for simplicity with small sizes:
            T_comp = np.einsum('ar,br,cr->abc', U, V, W)
            R = T - T_comp + np.einsum('ar,br,cr->abc', U[:, r:r+1], V[:, r:r+1], W[:, r:r+1])
            R_flat = R.reshape(n*m, -1)
            vw_flat = (V[:, r] * W[:, r])  # length m*p*p*n... no
            # Actually: R[a, b, c] = sum_r' U[a,r'] V[b,r'] W[c,r']
            # We want U[:,r] such that sum_r' U[a,r'] V[b,r'] W[c,r'] ≈ T[a,b,c]
            # Fix all except r: U[a,r] * V[b,r] * W[c,r] should give residual
            # U[:,r] = R_reshaped @ (V[:,r] * W[:,r]) / (V[:,r]^T V[:,r]) (W[:,r]^T W[:,r])? No.
            # U[a,r] = sum_{b,c} R[a,b,c] V[b,r] W[c,r] / (sum_b V[b,r]^2 * sum_c W[c,r]^2)? No, that's not right either.
            # The correct normal equation: U[:,r] = (sum_{b,c} R[a,b,c] V[b,r] W[c,r]) / (||V[:,r]||^2 ||W[:,r]||^2) is wrong.
            # Actually: minimize ||R - U[:,r] * V[:,r] * W[:,r]||_F^2 where the product is outer.
            # d/dU[a,r]: -2 * sum_{b,c} R[a,b,c] V[b,r] W[c,r] + 2 * U[a,r] * sum_{b,c} V[b,r]^2 W[c,r]^2
            # So U[a,r] = sum_{b,c} R[a,b,c] V[b,r] W[c,r] / (sum_b V[b,r]^2) (sum_c W[c,r]^2)
            # Wait no: sum_{b,c} V[b,r]^2 W[c,r]^2 = (sum_b V[b,r]^2)(sum_c W[c,r]^2)
            # So U[a,r] = (R.reshape(n*m,-1) @ (V[:,r]*W[:,r])) / ((V[:,r]**2).sum() * (W[:,r]**2).sum())
            
            R_flat = R.reshape(n*m, m*p*p*n)  # This is wrong shape
            # R has shape (n*m, m*p, p*n)
            # We need sum_{b,c} R[a,b,c] V[b,r] W[c,r]
            # = sum_b R[a,b,:] @ W[:,r] * V[b,r]... no
            # = (R @ (V[:,r][:,None] * W[:,r][None,:]))... this is getting complicated
            # Let me just do it directly:
            numerator = np.einsum('abc,b,c->a', R, V[:, r], W[:, r])
            denom = np.sum(V[:, r]**2) * np.sum(W[:, r]**2)
            if denom > 1e-15:
                U[:, r] = numerator / denom
        
        # Update V similarly
        for r in range(rank):
            T_comp = np.einsum('ar,br,cr->abc', U, V, W)
            R = T - T_comp + np.einsum('ar,br,cr->abc', U[:, r:r+1], V[:, r:r+1], W[:, r:r+1])
            numerator = np.einsum('abc,a,c->b', R, U[:, r], W[:, r])
            denom = np.sum(U[:, r]**2) * np.sum(W[:, r]**2)
            if denom > 1e-15:
                V[:, r] = numerator / denom
        
        # Update W similarly
        for r in range(rank):
            T_comp = np.einsum('ar,br,cr->abc', U, V, W)
            R = T - T_comp + np.einsum('ar,br,cr->abc', U[:, r:r+1], V[:, r:r+1], W[:, r:r+1])
            numerator = np.einsum('abc,a,b->c', R, U[:, r], V[:, r])
            denom = np.sum(U[:, r]**2) * np.sum(V[:, r]**2)
            if denom > 1e-15:
                W[:, r] = numerator / denom
        
        # Check convergence
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T - T_comp)
        if err < tol:
            break
    
    # Round to half-integers if close
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    T_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
    if np.allclose(T_rounded, T, atol=1e-9):
        return U_rounded, V_rounded, W_rounded
    
    # Try without rounding
    if np.allclose(T_comp, T, atol=1e-9):
        return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = 135  # Leave 15s margin from 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with trivial
    U_triv, V_triv, W_triv = _build_trivial(n, m, p)
    best_U, best_V, best_W = U_triv, V_triv, W_triv
    best_rank = n * m * p  # 56
    
    # Try Strassen block decomposition (rank 50)
    if time.time() - start_time < deadline - 30:
        U_str, V_str, W_str = _build_strassen_block(n, m, p)
        if _verify(U_str, V_str, W_str, n, m, p):
            best_U, best_V, best_W = U_str, V_str, W_str
            best_rank = U_str.shape[1]
            print(f"Strassen block: rank {best_rank} verified")
    
    # Try ALS for lower ranks
    for R in range(best_rank - 1, max(27, best_rank - 8), -1):
        if time.time() - start_time > deadline - 20:
            break
        found = False
        for seed in range(5):
            if time.time() - start_time > deadline - 15:
                break
            result = _try_als(n, m, p, R, max_iters=300, seed=seed)
            if result is not None:
                U_try, V_try, W_try = result
                if _verify(U_try, V_try, W_try, n, m, p):
                    best_U, best_V, best_W = U_try, V_try, W_try
                    best_rank = R
                    found = True
                    print(f"ALS found rank {R} with seed {seed}")
                    break
        if found:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i*m + j, j*p + k, k*n + i] = 1
    
    if np.allclose(T, Expected, atol=1e-9):
        print(f"Valid decomposition with rank {U.shape[1]}")
    else:
        print("Invalid decomposition")
        diff = np.abs(T - Expected)
        print(f"Max diff: {diff.max()}")
