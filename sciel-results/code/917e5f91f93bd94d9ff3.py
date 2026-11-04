"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 145.0

    # Strategy: split-base-case for 4x4 * 4x6
    # Split A (4x4) into four 2x2 blocks, B (4x6) into four 2x3 blocks
    # Each 2x2 * 2x3 sub-product: rank 7 (Strassen 2x2x2) + rank 4 (schoolbook 2x2x1) = 11
    # Total: 4 * 11 = 44 base terms
    # Then apply SVD-based rank reduction to find linear dependencies.

    def verify(U, V, W):
        """Check decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        Wa = np.asarray(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Strassen 2x2x2 coefficients: (a_dict, b_dict, c_dict)
    strassen_2x2x2 = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}),
        ({(0,1):1, (1,1):1}, {(0,0):1}, {(0,0):-1, (1,1):1}),
        ({(0,0):1}, {(1,1):1, (0,0):-1}, {(0,0):1, (1,0):-1}),
        ({(1,0):1}, {(1,0):1, (0,0):-1}, {(1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):-1, (1,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,0):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,1):1, (1,1):-1}),
    ]

    # Schoolbook 2x2x1 coefficients (B is 2x1): rank 4
    schoolbook_2x2x1 = [
        ({(0,0):1}, {(0,0):1}, {(0,0):1}),
        ({(0,1):1}, {(0,0):1}, {(0,1):1}),
        ({(1,0):1}, {(1,0):1}, {(1,0):1}),
        ({(1,1):1}, {(1,0):1}, {(1,1):1}),
    ]

    terms = []  # list of (a_dict, b_dict, c_dict) with global indices

    # For each 2x2 A-block and 2x3 B-block pair:
    # A-block (bi, bj): rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # B-block (ci, cj): rows 2*ci..2*ci+1, cols 3*cj..3*cj+2
    # Only contribute when A-block col range == B-block row range (bj == ci)
    for bi in range(2):       # A row block
        for bj in range(2):   # A col block (= B row block)
            ci = bj           # B row block must match A col block
            for cj in range(2):  # B col block
                # Strassen 2x2x2 part (B cols 0,1 of the 2x3 block)
                for a_d, b_d, c_d in strassen_2x2x2:
                    a_g = {}
                    for (a, c), v in a_d.items():
                        i = 2*bi + a
                        j = 2*bj + c
                        a_g[(i, j)] = a_g.get((i, j), 0) + v
                    b_g = {}
                    for (c, d), v in b_d.items():
                        j = 2*ci + c
                        k = 3*cj + d
                        b_g[(j, k)] = b_g.get((j, k), 0) + v
                    c_g = {}
                    for (a, d), v in c_d.items():
                        i = 2*bi + a
                        k = 3*cj + d
                        c_g[(i, k)] = c_g.get((i, k), 0) + v
                    terms.append((a_g, b_g, c_g))
                # Schoolbook 2x2x1 part (B col 2 of the 2x3 block)
                for a_d, b_d, c_d in schoolbook_2x2x1:
                    a_g = {}
                    for (a, c), v in a_d.items():
                        i = 2*bi + a
                        j = 2*bj + c
                        a_g[(i, j)] = a_g.get((i, j), 0) + v
                    b_g = {}
                    for (c, d), v in b_d.items():
                        j = 2*ci + c
                        k = 3*cj + 2  # the 3rd column of the 2x3 block
                        b_g[(j, k)] = b_g.get((j, k), 0) + v
                    c_g = {}
                    for (a, d), v in c_d.items():
                        i = 2*bi + a
                        k = 3*cj + 2
                        c_g[(i, k)] = c_g.get((i, k), 0) + v
                    terms.append((a_g, b_g, c_g))

    # Build initial matrices
    rank0 = len(terms)
    U = np.zeros((n * m, rank0))
    V = np.zeros((m * p, rank0))
    W = np.zeros((p * n, rank0))

    for col, (a_g, b_g, c_g) in enumerate(terms):
        for (i, j), coeff in a_g.items():
            U[i * m + j, col] = coeff
        for (j, k), coeff in b_g.items():
            V[j * p + k, col] = coeff
        for (i, k), coeff in c_g.items():
            W[k * n + i, col] = coeff

    # Verify base decomposition
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, c] = 1
                    V_fb[j * p + k, c] = 1
                    W_fb[k * n + i, c] = 1
                    c += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    best_U, best_V, best_W = U, V, W
    best_rank = rank0

    # ---- Rank reduction via SVD-based dependency detection ----
    # Key insight: the tensor T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # If we fix U and V, then W must satisfy:
    #   For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is a linear system in W given U, V.
    # If the system has a solution with fewer non-zero columns, we've reduced rank.
    #
    # Better approach: use the "slice" structure.
    # For each fixed k, the equation is:
    #   sum_c (U[:,c] * V[:,c])_element(i*m+j) * W[k*n+i, c] = 1 for all i
    # where V[:,c] is indexed by j*p+k.
    #
    # We'll use iterative column elimination with linear algebra verification.

    def fast_verify(U_mat, V_mat, W_mat):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_mat[i*m+j], V_mat[j*p+k] * W_mat[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def try_eliminate(U_mat, V_mat, W_mat, max_time=120.0):
        """Iteratively eliminate columns that are linearly redundant."""
        start = time.time()
        current_rank = U_mat.shape[1]
        U_c = U_mat.copy()
        V_c = V_mat.copy()
        W_c = W_mat.copy()

        # Precompute all "tensor slices" for fast checking
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        def check_removal(col_idx):
            """Check if removing column col_idx still gives valid decomposition."""
            if time.time() - start > max_time:
                return False
            mask = [i for i in range(current_rank) if i != col_idx]
            U_t = U_c[:, mask]
            V_t = V_c[:, mask]
            W_t = W_c[:, mask]
            return fast_verify(U_t, V_t, W_t)

        # Greedy elimination: try removing each column, repeat until no more
        improved = True
        while improved and time.time() - start < max_time:
            improved = False
            for col in range(current_rank):
                if time.time() - start > max_time:
                    break
                if check_removal(col):
                    mask = [i for i in range(current_rank) if i != col]
                    U_c = U_c[:, mask]
                    V_c = V_c[:, mask]
                    W_c = W_c[:, mask]
                    current_rank = len(mask)
                    improved = True
                    break  # restart scan after removal

        return U_c, V_c, W_c, current_rank

    # Phase 1: greedy column elimination
    U1, V1, W1, r1 = try_eliminate(U, V, W, max_time=100.0)
    if r1 < best_rank:
        best_U, best_V, best_W, best_rank = U1, V1, W1, r1

    # Phase 2: try SVD-based projection
    # The decomposition can be seen as: for each (i,k), the vector
    #   v_{i,k} = sum_c W[k*n+i,c] * (U[:,c] * V[:,c]) restricted to (j) 
    # Actually, let's think differently.
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Fix i,k: sum_j' ... no, j is free.
    # For fixed i,k: the equation must hold for all j:
    #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
    #
    # Alternative: think of it as a bilinear form.
    # Let's try: for each pair of "input indices" (i, k), we need
    #   sum_c W[k*n+i,c] * (U[i*m+*,c] ⊙ V[*,p+k,c]) = 1 for all j
    # where ⊙ is element-wise and the * ranges over j.
    #
    # This means: W[k*n+i,:] must be in the row space of a certain matrix,
    # and specifically must produce the all-ones vector when multiplied.
    #
    # Let's build the constraint matrix for W given U, V:
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # Group by (i,k): for each j, this is a linear equation in W[k*n+i,:].
    #
    # Actually, the simplest structural reduction: use the fact that
    # the 44-term construction has redundancy because the Strassen and
    # schoolbook parts interact. Let's just do the greedy elimination
    # which already captures this.

    # Phase 3: try random column perturbations to find lower-rank solutions
    # If we have a valid decomposition of rank r, we can try to find
    # a decomposition of rank r-1 by checking if the (r-1)-th column
    # is in the "span" of the others in a tensor sense.
    #
    # More concretely: if we fix U and V, then W is determined by a linear
    # system. If that system is underdetermined, we can find a solution
    # with fewer non-zero entries.
    
    if time.time() < deadline - 10:
        # Try to solve for W given U, V using linear algebra
        # Variables: W[k*n+i, c] for all k, i, c
        # Equations: for all i, j, k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        #
        # This is a linear system A * w = b where w is the flattened W.
        # Number of equations: n*m*p = 96
        # Number of unknowns: p*n*rank = 24*rank
        # If rank is large enough, the system is underdetermined.
        
        r_cur = best_rank
        # Build the linear system
        num_eq = n * m * p
        num_unk = p * n * r_cur
        A_sys = np.zeros((num_eq, num_unk))
        b_sys = np.ones(num_eq)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r_cur):
                        # W index: k*n+i, column c
                        w_idx = (k * n + i) * r_cur + c
                        A_sys[eq_idx, w_idx] = best_U[i*m+j, c] * best_V[j*p+k, c]
                    eq_idx += 1
        
        # Solve: find a solution w with minimal support (sparse)
        # Use least squares first, then try to zero out small components
        try:
            w_sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
            w_sol = w_sol.reshape(p * n, r_cur)
            
            # Check if this solution is valid
            if fast_verify(best_U, best_V, w_sol):
                # Try to sparsify: zero out small components
                for threshold in [1e-8, 1e-6, 1e-4, 1e-2, 0.1]:
                    w_sparse = w_sol.copy()
                    w_sparse[np.abs(w_sparse) < threshold] = 0
                    if fast_verify(best_U, best_V, w_sparse):
                        # Count non-zero columns
                        col_sums = np.abs(w_sparse).sum(axis=0)
                        nonzero_cols = np.where(col_sums > 1e-10)[0]
                        if len(nonzero_cols) < r_cur:
                            U_red = best_U[:, nonzero_cols]
                            V_red = best_V[:, nonzero_cols]
                            W_red = w_sparse[:, nonzero_cols]
                            if fast_verify(U_red, V_red, W_red):
                                if len(nonzero_cols) < best_rank:
                                    best_U, best_V, best_W, best_rank = U_red, V_red, W_red, len(nonzero_cols)
                                break
        except np.linalg.LinAlgError:
            pass

    # Final verification and return
    if not verify(best_U, best_V, best_W):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, c] = 1
                    V_fb[j * p + k, c] = 1
                    W_fb[k * n + i, c] = 1
                    c += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
