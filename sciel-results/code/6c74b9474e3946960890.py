import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def _build_strassen_2x2x2():
    """Return the standard Strassen rank-7 decomposition for 2x2*2x2.
    Each term is (u_vec, v_vec, w_vec) where vectors are over the
    4-dim A-space, 4-dim B-space, 4-dim C-space respectively.
    A indices: 0=a00, 1=a01, 2=a10, 3=a11
    B indices: 0=b00, 1=b01, 2=b10, 3=b11
    C indices: 0=c00, 1=c01, 2=c10, 3=c11
    """
    terms = []
    # m1 = (a00+a11)(b00+b11) -> c00 + c11
    terms.append(([1,0,0,1], [1,0,0,1], [1,1,0,0]))
    # m2 = (a10+a11)b00 -> c10 - c11
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    # m3 = a00(b01-b11) -> c01 + c11
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,1]))
    # m4 = a11(b10-b00) -> c00 + c10
    terms.append(([0,0,0,1], [0,0,1,-1], [1,0,1,0]))
    # m5 = (a00+a01)b11 -> -c00 + c01
    terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    # m6 = (a10-a00)(b00+b01) -> c11
    terms.append([[-1,0,1,0], [1,1,0,0], [0,0,0,1]])
    # m7 = (a01-a11)(b10+b11) -> c00
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    return terms

def _try_lower_rank_search(U, V, W, deadline):
    """Try to reduce rank by detecting linearly dependent columns via SVD.
    Returns (U, V, W) potentially with fewer columns."""
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    
    # Build the 3-way tensor and check which columns are linearly dependent
    # A column r is redundant if U[:,r], V[:,r], W[:,r] are linearly
    # dependent with other columns in the tensor sense.
    # Practical approach: try removing each column and verify.
    
    # First, use SVD on the matricizations to find near-dependent columns
    # Matricize along first mode: shape (n*m, rank * m*p)
    # Actually, simpler: check if removing column r still gives valid decomposition
    
    # For efficiency, try removing columns in order of smallest contribution
    # Compute "importance" of each column as the norm of its tensor contribution
    
    n, m, p = DIMENSIONS
    tensor_size = n * m * m * p * p * n  # too big, use einsum
    
    # Alternative: try removing each column one at a time (greedy)
    # But that's O(rank * verification_cost). With rank=39, that's 39 verifications.
    # Each verification is an einsum of size (4, 22, 22) which is fast.
    
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = rank
    
    # Greedy: try removing each column, keep removal if still valid
    removed = set()
    for _ in range(10):  # at most 10 removals
        if time.time() > deadline - 5:
            break
        best_remove = -1
        for r in range(current_rank):
            if r in removed:
                continue
            # Try removing column r
            mask = [i for i in range(current_rank) if i != r]
            U_test = current_U[:, mask]
            V_test = current_V[:, mask]
            W_test = current_W[:, mask]
            
            # Quick verify
            T_recon = np.einsum('ar,br,cr->abc', U_test, V_test, W_test)
            T_target = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_target[i*m+j, j*p+k, k*n+i] = 1.0
            if np.allclose(T_recon, T_target, atol=1e-10):
                best_remove = r
                break
        
        if best_remove == -1:
            break
        
        mask = [i for i in range(current_rank) if i != best_remove]
        current_U = current_U[:, mask]
        current_V = current_V[:, mask]
        current_W = current_W[:, mask]
        current_rank -= 1
        removed.add(best_remove)
    
    return current_U, current_V, current_W

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=1e-10)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    # Rank 39 Construction
    # 5 blocks of 2x2 (Strassen, rank 7 each) + 1 block of 2x1 (Schoolbook, rank 4)
    rank = 39
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    r = 0

    # Helper to add a Strassen block for columns k0, k1
    def add_strassen_block(k0, k1):
        nonlocal r
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = 11 + k0  # B[1, k0]
        v_h = 11 + k1  # B[1, k1]

        w_c00 = 2 * k0     # C[0, k0]
        w_c10 = 2 * k0 + 1 # C[1, k0]
        w_c01 = 2 * k1     # C[0, k1]
        w_c11 = 2 * k1 + 1 # C[1, k1]

        # m1 = (a+d)(e+h)
        U[0, r] += 1; U[3, r] += 1
        V[v_e, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1; W[w_c11, r] += 1
        r += 1

        # m2 = (c+d)e
        U[2, r] += 1; U[3, r] += 1
        V[v_e, r] += 1
        W[w_c10, r] += 1; W[w_c11, r] += -1
        r += 1

        # m3 = a(f-h)
        U[0, r] += 1
        V[v_f, r] += 1; V[v_h, r] += -1
        W[w_c01, r] += 1; W[w_c11, r] += 1
        r += 1

        # m4 = d(g-e)
        U[3, r] += 1
        V[v_g, r] += 1; V[v_e, r] += -1
        W[w_c00, r] += 1; W[w_c10, r] += 1
        r += 1

        # m5 = (a+b)h
        U[0, r] += 1; U[1, r] += 1
        V[v_h, r] += 1
        W[w_c00, r] += -1; W[w_c01, r] += 1
        r += 1

        # m6 = (c-a)(e+f)
        U[2, r] += 1; U[0, r] += -1
        V[v_e, r] += 1; V[v_f, r] += 1
        W[w_c11, r] += 1
        r += 1

        # m7 = (b-d)(g+h)
        U[1, r] += 1; U[3, r] += -1
        V[v_g, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1
        r += 1

    # Add 5 Strassen blocks for column pairs (0,1), (2,3), (4,5), (6,7), (8,9)
    for block in range(5):
        if time.time() > deadline - 20:
            break
        k0 = block * 2
        k1 = block * 2 + 1
        add_strassen_block(k0, k1)

    # Last column k=10: Schoolbook
    if r < rank:
        k2 = 10
        v_e2 = k2      # B[0, 10]
        v_g2 = 11 + k2 # B[1, 10]
        w_c02 = 2 * k2      # C[0, 10]
        w_c12 = 2 * k2 + 1  # C[1, 10]

        U[0, r] += 1
        V[v_e2, r] += 1
        W[w_c02, r] += 1
        r += 1

        U[1, r] += 1
        V[v_g2, r] += 1
        W[w_c02, r] += 1
        r += 1

        U[2, r] += 1
        V[v_e2, r] += 1
        W[w_c12, r] += 1
        r += 1

        U[3, r] += 1
        V[v_g2, r] += 1
        W[w_c12, r] += 1
        r += 1

    # Verify the base construction
    if time.time() < deadline - 10:
        if _verify(U, V, W):
            # Try to reduce rank via greedy column removal
            U_red, V_red, W_red = _try_lower_rank_search(U, V, W, deadline)
            if U_red.shape[1] < U.shape[1]:
                return {
                    "U": U_red.tolist(),
                    "V": V_red.tolist(),
                    "W": W_red.tolist()
                }
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback: Schoolbook (Rank 44)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
