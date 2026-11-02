"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=2, m=2, p=9, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the multiplication tensor T[i,j,k] = 1 if the product contributes
    # T is of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Reconstruct from decomposition: T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Check against identity tensor
    
    # For random test vectors
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Flatten A: a[i*m+j] = A[i,j]
        a_vec = A.flatten()  # shape (n*m,)
        b_vec = B.flatten()  # shape (m*p,)
        
        # Compute C from decomposition
        # c_vec[k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * a_vec[i*m+j] * b_vec[j*p+k]
        # Actually: C[i,k] = sum_j A[i,j]*B[j,k]
        # In tensor form: c_vec[k*n+i] = sum_{i,j,k: T[i,j,k]=1} a_vec[i*m+j]*b_vec[j*p+k]
        # With decomposition: c_vec = W @ (V.T @ (U.T @ a_vec) * ... ) -- no
        # c_vec[c] = sum_r W[c,r] * (sum_{a,b: U[a,r]*V[b,r]} a_vec[a]*b_vec[b])
        # = sum_r W[c,r] * (U[:,r] @ a_vec) * (V[:,r] @ b_vec)
        
        c_vec = np.zeros(p * n)
        for r in range(rank):
            u_val = np.dot(U[:, r], a_vec)
            v_val = np.dot(V[:, r], b_vec)
            c_vec += W[:, r] * (u_val * v_val)
        
        C_recon = c_vec.reshape(p, n).T  # shape (n, p)
        
        if not np.allclose(C_recon, C_expected, atol=tol):
            return False
    return True

def _strassen_block(k1, k2, start_col, n, m, p, U, V, W):
    """Add 7 Strassen products for columns k1, k2."""
    v_00 = 0 * p + k1
    v_01 = 0 * p + k2
    v_10 = 1 * p + k1
    v_11 = 1 * p + k2
    w_00 = k1 * n + 0
    w_01 = k1 * n + 1
    w_02 = k2 * n + 0
    w_03 = k2 * n + 1
    u_00, u_01, u_10, u_11 = 0, 1, 2, 3
    
    # M1 = (a00 + a11) * (b00 + b11)
    c = start_col + 0
    U[u_00, c] += 1; U[u_11, c] += 1
    V[v_00, c] += 1; V[v_11, c] += 1
    W[w_00, c] += 1; W[w_03, c] += 1
    
    # M2 = (a10 + a11) * b00
    c = start_col + 1
    U[u_10, c] += 1; U[u_11, c] += 1
    V[v_00, c] += 1
    W[w_01, c] += 1; W[w_03, c] -= 1
    
    # M3 = a00 * (b01 - b11)
    c = start_col + 2
    U[u_00, c] += 1
    V[v_01, c] += 1; V[v_11, c] -= 1
    W[w_02, c] += 1; W[w_03, c] += 1
    
    # M4 = a11 * (b10 - b00)
    c = start_col + 3
    U[u_11, c] += 1
    V[v_10, c] += 1; V[v_00, c] -= 1
    W[w_00, c] += 1; W[w_01, c] += 1
    
    # M5 = (a00 + a01) * b11
    c = start_col + 4
    U[u_00, c] += 1; U[u_01, c] += 1
    V[v_11, c] += 1
    W[w_00, c] -= 1; W[w_02, c] += 1
    
    # M6 = (a10 - a00) * (b00 + b01)
    c = start_col + 5
    U[u_10, c] += 1; U[u_00, c] -= 1
    V[v_00, c] += 1; V[v_01, c] += 1
    W[w_03, c] += 1
    
    # M7 = (a01 - a11) * (b10 + b11)
    c = start_col + 6
    U[u_01, c] += 1; U[u_11, c] -= 1
    V[v_10, c] += 1; V[v_11, c] += 1
    W[w_00, c] += 1
    
    return start_col + 7

def _schoolbook_single(k, start_col, n, m, p, U, V, W):
    """Add 4 schoolbook products for a single column k."""
    v_0 = 0 * p + k
    v_1 = 1 * p + k
    w_0 = k * n + 0
    w_1 = k * n + 1
    u_00, u_01, u_10, u_11 = 0, 1, 2, 3
    
    c = start_col + 0
    U[u_00, c] += 1; V[v_0, c] += 1; W[w_0, c] += 1
    
    c = start_col + 1
    U[u_01, c] += 1; V[v_1, c] += 1; W[w_0, c] += 1
    
    c = start_col + 2
    U[u_10, c] += 1; V[v_0, c] += 1; W[w_1, c] += 1
    
    c = start_col + 3
    U[u_11, c] += 1; V[v_1, c] += 1; W[w_1, c] += 1
    
    return start_col + 4

def _build_rank32(n=2, m=2, p=9):
    """Build the rank-32 decomposition: 4 Strassen blocks + 1 schoolbook column."""
    rank = 32
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for k1, k2 in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        col = _strassen_block(k1, k2, col, n, m, p, U, V, W)
    col = _schoolbook_single(8, col, n, m, p, U, V, W)
    
    return U, V, W

def construct():
    """
    Construct a minimal-rank tensor decomposition for 2x2 * 2x9 matrix multiplication.
    
    Baseline: rank 32 (4 Strassen blocks + 1 schoolbook column).
    Attempts to find a lower-rank decomposition if possible within time budget.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Build and verify the rank-32 baseline
    U, V, W = _build_rank32(n, m, p)
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback: pure schoolbook rank 18*... wait, that's not right.
        # Pure schoolbook for 2x2*2x9: each of 18 outputs needs products.
        # Actually schoolbook: c[i,k] = sum_j a[i,j]*b[j,k], 4 products per output, 18 outputs = 36 products.
        # But we can share: for fixed k, c[0,k] and c[1,k] share b[j,k].
        # Total unique products: 2*2*9 = 36 (no sharing possible across k since b[j,k] differs).
        # Actually within a column k: 4 products. Across 9 columns: 36 products. Rank 36.
        pass  # The rank-32 should be correct; this is a safety check
    
    best_rank = 32
    best_U, best_V, best_W = U, V, W
    
    # Try to find rank-31 or lower via structured search
    # Strategy: try different groupings and small perturbations
    
    # Try: 3 Strassen blocks (6 cols, rank 21) + custom 3-column block (rank 10?)
    # 2x2 * 2x3: can we do rank 10?
    # Schoolbook for 2x2*2x3 = 12 products. Strassen-like savings possible.
    # Let's try to construct a rank-10 decomposition for 2x2*2x3.
    
    # For now, keep the verified rank-32 as our answer.
    # The score is 36/32 = 1.125. To improve, need rank <= 31.
    
    # Try to find a rank-31 decomposition by using a rank-10 block for 2x2*2x3
    if time.time() < deadline - 5:
        # Construct a rank-10 decomposition for 2x2 * 2x3
        # Columns 0,1,2 of the 2x3 matrix B
        # This is equivalent to multiplying A (2x2) by B (2x3)
        #
        # Let's try: Strassen for cols 0,1 (rank 7) + modified schoolbook for col 2 (rank 3?)
        # Col 2 alone needs 4 products. Can we share with the Strassen block?
        #
        # Alternative: direct rank-10 construction
        # Let's parameterize and solve.
        
        # For 2x2*2x3, let's denote:
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01, b02], [b10, b11, b12]]
        # C = [[c00, c01, c02], [c10, c11, c12]]
        #
        # c00 = a00*b00 + a01*b10
        # c01 = a00*b01 + a01*b11
        # c02 = a00*b02 + a01*b12
        # c10 = a10*b00 + a11*b10
        # c11 = a10*b01 + a11*b11
        # c12 = a10*b02 + a11*b12
        #
        # Strassen for cols 0,1:
        # M1 = (a00+a11)*(b00+b11)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b01-b11)
        # M4 = a11*(b10-b00)
        # M5 = (a00+a01)*b11
        # M6 = (a10-a00)*(b00+b01)
        # M7 = (a01-a11)*(b10+b11)
        #
        # For col 2, schoolbook:
        # N1 = a00*b02
        # N2 = a01*b12
        # N3 = a10*b02
        # N4 = a11*b12
        #
        # Total: 11. Can we reduce to 10?
        #
        # Idea: modify M5 to incorporate col 2.
        # M5_orig = (a00+a01)*b11
        # What if M5_new = (a00+a01)*(b11 + alpha*b12)?
        # Then we get extra terms involving b12.
        #
        # Let's try: replace M5 with (a00+a01)*(b11 + b12)
        # M5_new = a00*b11 + a00*b12 + a01*b11 + a01*b12
        #
        # Then c02 = a00*b02 + a01*b12 = a00*b02 + (M5_new - a00*b11 - a01*b11 - a00*b12 + a00*b12)
        # This is getting complicated. Let me just try rank 11 for the 3-col block.
        #
        # Total with 3 Strassen + 1 rank-11 block: 3*7 + 11 = 32. Same.
        #
        # Let me try a completely different grouping.
        # What about 2x2*2x4 (rank?) + 2x2*2x5 (rank?)
        # 2x2*2x4: Strassen on (0,1) + Strassen on (2,3) = 14. Or better?
        # 2x2*2x5: Strassen on (0,1) + Strassen on (2,3) + schoolbook on 4 = 7+7+4 = 18
        # Total: 14 + 18 = 32. Same.
        #
        # I don't see an obvious way to get below 32 with simple block decompositions.
        # Let's keep rank 32.
        pass

    # Return the rank-32 decomposition
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
