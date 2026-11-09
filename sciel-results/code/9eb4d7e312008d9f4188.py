"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _build_target_tensor():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-10):
    """Verify that sum_r U[:,r] V[:,r] W[:,r] == T."""
    S = np.zeros_like(T)
    for r in range(U.shape[1]):
        S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.linalg.norm(T - S) < tol

def _construct_strassen_238():
    """
    Construct a rank-44 decomposition for 2x3x8 matrix multiplication
    using a Strassen-type construction.
    
    A (2x3) * B (3x8) = C (2x8)
    
    Strategy: Split the middle dimension (3) into 2+1.
    A = [A1 | a]  where A1 is 2x2, a is 2x1
    B = [B1 ; b]  where B1 is 2x8, b is 1x8
    
    C = A1 * B1 + a * b
    
    For A1 * B1 (2x2 * 2x8): Split B1 into two 2x4 blocks.
    For each 2x2 * 2x4, use Strassen (7 mults) on the 2x2 part, 
    but actually we need 2x2 * 2x4 which is 8 products schoolbook.
    
    Better: Use the full Strassen construction adapted for rectangular.
    
    Actually, let's use a direct construction. The known rank for 2x3x8 is 44.
    We construct it by combining:
    - 7 Strassen terms for the 2x2x2 core (rows 0-1, cols 0-1 of middle)
    - Additional terms for the remaining parts
    
    Let me define it explicitly using the known rank-44 construction.
    """
    n, m, p = 2, 3, 8
    nm, mp, pn = n*m, m*p, p*n  # 6, 24, 16
    
    # We'll build 44 rank-1 terms
    terms_U = []
    terms_V = []
    terms_W = []
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, val=1.0):
        """u_indices: dict {row: val}, v_indices: dict {row: val}, w_indices: dict {row: val}"""
        u = np.zeros(nm)
        v = np.zeros(mp)
        w = np.zeros(pn)
        for idx, val in u_indices.items():
            u[idx] = val
        for idx, val in v_indices.items():
            v[idx] = val
        for idx, val in w_indices.items():
            w[idx] = val
        terms_U.append(u)
        terms_V.append(v)
        terms_W.append(w)
    
    # Index helpers
    # U index: i*m + j = i*3 + j, for i in {0,1}, j in {0,1,2}
    # V index: j*p + k = j*8 + k, for j in {0,1,2}, k in {0..7}
    # W index: k*n + i = k*2 + i, for k in {0..7}, i in {0,1}
    
    # Strassen for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For 2x2 * 2x4 (A1 is 2x2, B1_left is 2x4):
    # We can use Strassen per pair of columns, but that gives 7*2=14 for 2x2*2x4
    # Or we can do it as: split the 4 columns into two groups of 2, use Strassen on each: 7+7=14
    
    # Total: 
    # A1*B1 (2x2 * 2x8): split B1 into 4 groups of 2 columns each: 4*7=28 terms
    # a*b (2x1 * 1x8): 2*1*8 = 16 terms schoolbook, but we can do better
    #   Actually a*b is outer product of 2x1 and 1x8, which is rank 1! 
    #   Wait no: a is 2x1, b is 1x8. a*b is 2x8. 
    #   In tensor terms: for each i in {0,1}, k in {0..7}: contribution to C[i,k] from a[i]*b[k]
    #   This is 2*8 = 16 rank-1 terms in the tensor, BUT they share structure.
    #   Actually in the matrix multiplication tensor, the term "a[i]*b[k]" for fixed i,k 
    #   contributes to T[i*3+2, 2*8+k, k*2+i]. These are 16 distinct tensor entries, 
    #   each requiring its own rank-1 term unless we can combine them.
    #   But wait - can we combine? The tensor entries (i*3+2, 2*8+k, k*2+i) for 
    #   different (i,k) are all distinct, so we need 16 separate rank-1 terms for a*b.
    #   Hmm, that gives 28+16=44. 
    
    # Let me verify: 28 (for A1*B1) + 16 (for a*b) = 44. 
    
    # But wait, I need to double-check the A1*B1 part.
    # A1 is 2x2, B1 is 2x8. Product is 2x8.
    # In tensor: for i in {0,1}, j in {0,1}, k in {0..7}: T[i*3+j, j*8+k, k*2+i]
    # This is 2*2*8 = 32 tensor entries.
    # Using Strassen with 4 groups of 2 columns: 4*7 = 28 rank-1 terms. Good.
    
    # For a*b: i in {0,1}, j=2, k in {0..7}: T[i*3+2, 2*8+k, k*2+i]
    # This is 2*1*8 = 16 tensor entries. Each needs a rank-1 term. 16 terms.
    
    # Total: 28 + 16 = 44. 
    
    # Now let's construct the 28 Strassen terms for A1*B1.
    # A1[i,j] for i in {0,1}, j in {0,1}: U index = i*3 + j
    # B1[j,k] for j in {0,1}, k in {0..7}: V index = j*8 + k
    # C[i,k] for i in {0,1}, k in {0..7}: W index = k*2 + i
    
    # Split k into 4 groups of 2: group g has k in {2g, 2g+1}
    for g in range(4):
        k0 = 2*g
        k1 = 2*g + 1
        # Strassen for 2x2 * 2x2 (columns k0, k1)
        # M1 = (A1[0,0]+A1[1,1]) * (B1[0,k0]+B1[0,k1]... no wait
        # B1 is 2x2 for this sub-block: rows j=0,1; cols k0,k1
        # B1_sub[j, col] where col=0 -> k0, col=1 -> k1
        
        # M1 = (A1[0,0]+A1[1,1]) * (B1[0,k0]+B1[1,k1])  -- NO, Strassen is:
        # M1 = (A00+A11)(B00+B11)
        # where A00=A1[0,0], A11=A1[1,1], B00=B1[0,k0], B11=B1[1,k1]
        
        # M1
        add_term(
            {0*3+0: 1, 1*3+1: 1},  # A1[0,0] + A1[1,1]
            {0*8+k0: 1, 1*8+k1: 1},  # B1[0,k0] + B1[1,k1]
            {k0*2+0: 1, k1*2+0: 1, k0*2+1: 1, k1*2+1: 1},  # C[0,k0]+C[0,k1]+C[1,k0]+C[1,k1]
        )
        
        # M2 = (A10+A11) * B00 = (A1[1,0]+A1[1,1]) * B1[0,k0]
        add_term(
            {1*3+0: 1, 1*3+1: 1},  # A1[1,0] + A1[1,1]
            {0*8+k0: 1},  # B1[0,k0]
            {k0*2+1: 1, k1*2+1: 1},  # C[1,k0] + C[1,k1]
        )
        
        # M3 = A00 * (B01-B11) = A1[0,0] * (B1[0,k1] - B1[1,k1])
        add_term(
            {0*3+0: 1},  # A1[0,0]
            {0*8+k1: 1, 1*8+k1: -1},  # B1[0,k1] - B1[1,k1]
            {k1*2+0: 1},  # C[0,k1]
        )
        
        # M4 = A11 * (B10-B00) = A1[1,1] * (B1[1,k0] - B1[0,k0])
        add_term(
            {1*3+1: 1},  # A1[1,1]
            {1*8+k0: 1, 0*8+k0: -1},  # B1[1,k0] - B1[0,k0]
            {k0*2+0: 1, k0*2+1: 1},  # C[0,k0] + C[1,k0]
        )
        
        # M5 = (A00+A01) * B11 = (A1[0,0]+A1[0,1]) * B1[1,k1]
        add_term(
            {0*3+0: 1, 0*3+1: 1},  # A1[0,0] + A1[0,1]
            {1*8+k1: 1},  # B1[1,k1]
            {k1*2+0: 1, k1*2+1: 1},  # C[0,k1] + C[1,k1]
        )
        
        # M6 = (A10-A00) * (B00+B01) = (A1[1,0]-A1[0,0]) * (B1[0,k0]+B1[0,k1])
        add_term(
            {1*3+0: 1, 0*3+0: -1},  # A1[1,0] - A1[0,0]
            {0*8+k0: 1, 0*8+k1: 1},  # B1[0,k0] + B1[0,k1]
            {k0*2+1: 1, k1*2+1: 1},  # C[1,k0] + C[1,k1]
        )
        
        # M7 = (A01-A11) * (B10+B11) = (A1[0,1]-A1[1,1]) * (B1[1,k0]+B1[1,k1])
        add_term(
            {0*3+1: 1, 1*3+1: -1},  # A1[0,1] - A1[1,1]
            {1*8+k0: 1, 1*8+k1: 1},  # B1[1,k0] + B1[1,k1]
            {k0*2+0: 1, k1*2+0: 1},  # C[0,k0] + C[0,k1]
        )
    
    # Now add 16 terms for a*b where a = A[:,2] (2x1), b = B[2,:] (1x8)
    # For i in {0,1}, k in {0..7}:
    # U: A[i,2] -> index i*3+2
    # V: B[2,k] -> index 2*8+k = 16+k
    # W: C[i,k] -> index k*2+i
    for i in range(2):
        for k in range(8):
            add_term(
                {i*3+2: 1},  # A[i,2]
                {2*8+k: 1},  # B[2,k]
                {k*2+i: 1},  # C[i,k]
            )
    
    # Build matrices
    rank = len(terms_U)
    U = np.array(terms_U).T  # shape (nm, rank)
    V = np.array(terms_V).T  # shape (mp, rank)
    W = np.array(terms_W).T  # shape (pn, rank)
    
    return U, V, W, rank

def _construct_schoolbook():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor()
    start_time = time.time()
    deadline = 110.0  # Safety margin before 150s kill
    
    # Try Strassen construction first (rank 44)
    try:
        U_str, V_str, W_str, rank_str = _construct_strassen_238()
        if _verify_decomposition(U_str, V_str, W_str, T, tol=1e-10):
            # Valid rank-44 solution
            return {
                "U": U_str.tolist(),
                "V": V_str.tolist(),
                "W": W_str.tolist()
            }
    except Exception:
        pass
    
    # Fallback: schoolbook (rank 48)
    U_school, V_school, W_school, rank_school = _construct_schoolbook()
    
    # Try random search for even lower rank (unlikely to find < 44 for 2x3x8)
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    
    if time.time() < deadline - 5:
        min_rank = max(nm, mp, pn)
        # Only search between lower bound and Strassen rank
        for R in range(min_rank, 44):
            if time.time() > deadline:
                break
            
            for init_idx in range(5):
                if time.time() > deadline:
                    break
                
                np.random.seed(init_idx * 1000 + R)
                U = np.random.randn(nm, R) * 0.1
                V = np.random.randn(mp, R) * 0.1
                W = np.random.randn(pn, R) * 0.1
                
                S = np.zeros_like(T)
                for r in range(R):
                    S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                
                prev_error = float('inf')
                converged = False
                for it in range(50):
                    if time.time() > deadline:
                        break
                    
                    for r in range(R):
                        S -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                        vw_mat = np.outer(V[:, r], W[:, r])
                        numerator = np.einsum('ijk,jk->i', T - S, vw_mat)
                        denominator = np.sum(vw_mat ** 2)
                        if denominator > 1e-10:
                            U[:, r] = numerator / denominator
                        S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                    
                    for r in range(R):
                        S -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                        uw_mat = np.outer(U[:, r], W[:, r])
                        numerator = np.einsum('ijk,ik->j', T - S, uw_mat)
                        denominator = np.sum(uw_mat ** 2)
                        if denominator > 1e-10:
                            V[:, r] = numerator / denominator
                        S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                    
                    for r in range(R):
                        S -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                        uv_mat = np.outer(U[:, r], V[:, r])
                        numerator = np.einsum('ijk,ij->k', T - S, uv_mat)
                        denominator = np.sum(uv_mat ** 2)
                        if denominator > 1e-10:
                            W[:, r] = numerator / denominator
                        S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                    
                    error = np.linalg.norm(T - S)
                    if error < 1e-8:
                        converged = True
                        break
                    if abs(prev_error - error) < 1e-12:
                        break
                    prev_error = error
                
                if not converged:
                    continue
                
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if _verify_decomposition(U_snap, V_snap, W_snap, T, tol=1e-10):
                    best_U, best_V, best_W, best_rank = U_snap, V_snap, W_snap, R
                    break
            
            if best_rank == R:
                break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
