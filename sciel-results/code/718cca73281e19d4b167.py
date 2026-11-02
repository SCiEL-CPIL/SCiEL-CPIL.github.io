import numpy as np
import time

DIMENSIONS = [2, 3, 6]


def _verify(U, V, W, n, m, p):
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('abl,cdl,kil->abcdki', U3, V3, W3)
    target = np.zeros((n, m, m, p, p, n), dtype=float)
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    if b == d and c < p:
                        target[a, b, c, d, c, a] = 1.0
    return np.allclose(T, target, atol=1e-9)


def _build_target_tensor(n, m, p):
    T = np.zeros((n, m, m, p, p, n), dtype=float)
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    if b == d and c < p:
                        T[a, b, c, d, c, a] = 1.0
    return T


def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _strassen_2x2x2():
    """
    Return the classic Strassen rank-7 decomposition of 2x2x2.
    The 7 rank-1 terms are indexed 0..6.
    Each term has a 2x2 U, 2x2 V, 2x2 W (flattened).
    Strassen formulas for C = A*B where A,B,C are 2x2:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11) B00
      M3 = A00 (B10-B11)
      M4 = A11 (B01-B11)
      M5 = (A00+A01) B11
      M6 = (A10-A00)(B00+B10)
      M7 = (A01-A11)(B01+B11)
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # Each entry: (u_vec, v_vec, w_vec) where vec is length 4 (row-major 2x2)
    # u_vec[a,b] = coefficient of A[a,b]
    # v_vec[c,d] = coefficient of B[c,d]
    # w_vec[k,i] = coefficient of C[k,i]
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append((np.array([1,0,0,1]), np.array([1,0,0,1]), np.array([1,0,0,-1])))
    # M2 = (A10+A11) B00
    terms.append((np.array([0,0,1,1]), np.array([1,0,0,0]), np.array([0,0,-1,0])))
    # M3 = A00 (B10-B11)
    terms.append((np.array([1,0,0,0]), np.array([0,0,1,-1]), np.array([0,1,0,0])))
    # M4 = A11 (B01-B11)
    terms.append((np.array([0,0,0,1]), np.array([0,1,0,-1]), np.array([1,0,0,1])))
    # M5 = (A00+A01) B11
    terms.append((np.array([1,1,0,0]), np.array([0,0,0,1]), np.array([-1,0,0,0])))
    # M6 = (A10-A00)(B00+B10)
    terms.append((np.array([-1,0,1,0]), np.array([1,0,1,0]), np.array([0,0,0,1])))
    # M7 = (A01-A11)(B01+B11)
    terms.append((np.array([0,1,0,-1]), np.array([0,1,0,1]), np.array([0,1,0,0])))
    
    return terms


def _try_block_decomposition(n, m, p):
    """
    Try to exploit the 2x3x6 = 2x(3x2)x3 structure by using Strassen on each 2x2x2 block
    and then combining. The 3x6 matrix can be seen as 3 rows x 6 columns = 3 rows x 3 groups of 2.
    So the multiplication is C[a, 2c+d] = sum_b A[a,b] * B[b, 2c+d].
    
    We can write this as: for each c in 0,1,2: C_c (2x2) = A * B_c where B_c is the 3x2 block.
    But A is 2x3, B_c is 3x2. This is a 2x3x2 multiplication for each c.
    
    Actually, let's think of it differently. We have A (2x3), B (3x6).
    We can partition B into 3 blocks of 3x2: B = [B_0, B_1, B_2].
    Then C = [A*B_0, A*B_1, A*B_2], where each A*B_c is 2x2.
    
    For 2x3x2 multiplication, the rank is known to be 11 (not 12 as schoolbook).
    If we use rank 11 for each of the 3 blocks, total rank = 33.
    But we can do better by sharing computations between blocks.
    
    Let me try a direct approach: use the Strassen-based decomposition for 2x2x2
    and extend to 2x3x2, then to 2x3x6.
    
    For now, let me try rank 33 = 3 * 11 using independent 2x3x2 blocks.
    """
    pass


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Schoolbook fallback: rank 36
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = n * m * p  # 36

    # Try to find a rank-33 decomposition using three 2x3x2 blocks of rank 11.
    # For 2x3x2: A is 2x3, B is 3x2, C is 2x2.
    # The multiplication tensor has 2*3*2 = 12 entries in the target.
    # Known: rank of 2x3x2 multiplication tensor is 11.
    
    # Let me try to construct a rank-11 decomposition of 2x3x2.
    # A = [[a0, a1, a2], [a3, a4, a5]]
    # B = [[b0, b1], [b2, b3], [b4, b5]]
    # C = [[c0, c1], [c2, c3]]
    # c0 = a0*b0 + a1*b2 + a2*b4
    # c1 = a0*b1 + a1*b3 + a2*b5
    # c2 = a3*b0 + a4*b2 + a5*b4
    # c3 = a3*b1 + a4*b3 + a5*b5
    
    # This is like computing two dot products of length 3 for each row of A.
    # For row 0: [a0,a1,a2] * B^T where B^T is 2x3.
    # For row 1: [a3,a4,a5] * B^T.
    
    # We can use Strassen-like ideas. Let me try to find a rank-11 decomposition.
    # One approach: use the fact that 2x3x2 is a sub-tensor of 2x3x3 (rank ?).
    # Actually, let me just use a greedy/spare search for rank 33-35.
    
    # Alternative: directly search for a low-rank decomposition using a structured ansatz.
    # Let me try the approach of using linear combinations of the schoolbook terms.
    
    # For rank 33: we need to eliminate 3 independent linear dependencies from the 36 schoolbook terms.
    # The schoolbook terms are: (i, j, k) for i in 0,1; j in 0,1,2; k in 0,1,2,3,4,5.
    # Term (i,j,k): U has 1 at (i,j), V has 1 at (j,k), W has 1 at (k,i).
    
    # Let me try a different structural approach:
    # View the 2x3x6 multiplication as a 2x2x2 tensor product with some sharing.
    
    # Actually, let me try a direct numerical optimization approach:
    # Start with the schoolbook 36 terms, then try to express 3 of them as linear combinations
    # of the others by finding appropriate U, V, W modifications.
    
    # For each of the 36 schoolbook terms, we have (u_l, v_l, w_l).
    # We want to find coefficients such that a linear combination of 33 terms equals the target.
    # This is equivalent to finding a 3-dimensional subspace of the term space where the sum vanishes.
    
    # Let me try a simpler approach: use the known result that rank(2x3x6) <= 33
    # and try to construct it explicitly.
    
    # Construction idea:
    # Partition the 6 columns into 3 groups of 2: columns (0,1), (2,3), (4,5).
    # For each group g, we compute a 2x2 subproduct using rank-11 2x3x2.
    # If we use independent rank-11 for each group, total = 33.
    
    # Let me try to find a rank-11 decomposition of 2x3x2.
    # I'll use a small search.
    
    def _find_2x3x2_rank11():
        """Try to find a rank-11 decomposition of 2x3x2."""
        nn, mm, pp = 2, 3, 2
        r = 11
        U = np.zeros((nn*mm, r))
        V = np.zeros((mm*pp, r))
        W = np.zeros((pp*nn, r))
        
        # Initialize with schoolbook (rank 12), then try to eliminate one term.
        # Schoolbook for 2x3x2:
        # Terms (i,j,k) for i in 0,1; j in 0,1,2; k in 0,1.
        # That's 12 terms. We want to reduce to 11.
        
        # Start with schoolbook, then try to express one term as a combination of others.
        # Term (i,j,k): U has 1 at (i,j), V has 1 at (j,k), W has 1 at (k,i).
        
        # Let me try to eliminate term (1,2,1) = (a5, b5, c3) [using 0-indexed]
        # Actually, let me just do a small local search.
        
        # Use schoolbook as starting point
        U_sb = np.zeros((6, 12))
        V_sb = np.zeros((6, 12))
        W_sb = np.zeros((4, 12))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    U_sb[i*3+j, col] = 1
                    V_sb[j*2+k, col] = 1
                    W_sb[k*2+i, col] = 1
                    col += 1
        
        # Try to find a linear dependency among the 12 terms.
        # If we can express one term as a linear combination of the other 11,
        # we can absorb it.
        
        # This is getting complex. Let me just use rank 36 (schoolbook) as the safe answer,
        # and try a quick search for improvement.
        
        return None  # Return None if we don't find a rank-11 decomposition
    
    # Quick search: try random rank-34 and rank-33 decompositions
    # by modifying the schoolbook decomposition.
    
    target = _build_target_tensor(n, m, p)
    target_flat = target.flatten()
    
    # Build the schoolbook atom matrix
    r_school = n * m * p  # 36
    atom_matrix = np.zeros((target.size, r_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n*m)
                v = np.zeros(m*p)
                w = np.zeros(p*n)
                u[i*m+j] = 1
                v[j*p+k] = 1
                w[k*n+i] = 1
                atom_tensor = np.einsum('ab,cd,ki->abcdki', 
                                        u.reshape(n,m), v.reshape(m,p), w.reshape(p,n))
                atom_matrix[:, col] = atom_tensor.flatten()
                col += 1
    
    # Try to find a rank-34 decomposition: select 34 atoms and solve for coefficients
    import random
    random.seed(42)
    
    for target_rank in [33, 34, 35]:
        if time.time() > deadline - 20:
            break
        
        found = False
        n_trials = 50
        for trial in range(n_trials):
            if time.time() > deadline - 10:
                break
            
            selected = random.sample(range(r_school), target_rank)
            A = atom_matrix[:, selected]
            
            try:
                x, residuals, rank, sv = np.linalg.lstsq(A, target_flat, rcond=None)
                residual = np.linalg.norm(A @ x - target_flat)
                if residual < 1e-9:
                    # Found a valid decomposition
                    U = np.zeros((n*m, target_rank))
                    V = np.zeros((m*p, target_rank))
                    W = np.zeros((p*n, target_rank))
                    for j, idx in enumerate(selected):
                        # Reconstruct the atom
                        i = idx // (m*p)
                        rem = idx % (m*p)
                        j_idx = rem // p
                        k_idx = rem % p
                        u = np.zeros(n*m)
                        v = np.zeros(m*p)
                        w = np.zeros(p*n)
                        u[i*m+j_idx] = 1
                        v[j_idx*p+k_idx] = 1
                        w[k_idx*n+i] = 1
                        U[:, j] = u
                        V[:, j] = v
                        W[:, j] = w
                    U = U * x
                    V = V * x
                    W = W * x
                    
                    if _verify(U, V, W, n, m, p):
                        best_U = U
                        best_V = V
                        best_W = W
                        best_rank = target_rank
                        found = True
                        break
            except Exception:
                continue
        
        if found:
            break
    
    # Final verification
    if not _verify(best_U, best_V, best_W, n, m, p):
        best_U = U_school
        best_V = V_school
        best_W = W_school
        best_rank = schoolbook_rank
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
