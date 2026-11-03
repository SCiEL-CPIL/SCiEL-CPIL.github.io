import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 5, 6]

def _build_target_tensor(n, m, p):
    """Build the flat list of (a,b,c) indices for the matrix multiplication tensor."""
    valid = []
    for i in range(n):
        for k in range(m):
            for j in range(p):
                valid.append((i * m + k, k * p + j, j * n + i))
    return valid

def _verify_decomp(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition is exact."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    valid = _build_target_tensor(n, m, p)
    
    all_entries = set()
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                all_entries.add((a, b, c))
    
    valid_set = set(valid)
    invalid = [(a, b, c) for (a, b, c) in all_entries if (a, b, c) not in valid_set]
    
    max_err = 0.0
    for a, b, c in valid:
        val = np.sum(U[a] * V[b] * W[c])
        max_err = max(max_err, abs(val - 1.0))
    for a, b, c in invalid:
        val = np.sum(U[a] * V[b] * W[c])
        max_err = max(max_err, abs(val))
    
    return max_err < tol

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k, col] = 1.0
                V[k * p + j, col] = 1.0
                W[j * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _strassen_222():
    """Return Strassen decomposition for 2x2x2 with rank 7."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1; W[2, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _embed_sub_decomp(U_sub, V_sub, W_sub, n, m, p, i0, k0, j0, n_sub, m_sub, p_sub):
    """Embed a sub-decomposition into the full tensor at offset positions."""
    rank_sub = U_sub.shape[1]
    U_full = np.zeros((n * m, rank_sub))
    V_full = np.zeros((m * p, rank_sub))
    W_full = np.zeros((p * n, rank_sub))
    
    for a_sub in range(n_sub * m_sub):
        i_sub = a_sub // m_sub
        k_sub = a_sub % m_sub
        a_full = (i0 + i_sub) * m + (k0 + k_sub)
        U_full[a_full, :] = U_sub[a_sub, :]
    
    for b_sub in range(m_sub * p_sub):
        k_sub = b_sub // p_sub
        j_sub = b_sub % p_sub
        b_full = (k0 + k_sub) * p + (j0 + j_sub)
        V_full[b_full, :] = V_sub[b_sub, :]
    
    for c_sub in range(p_sub * n_sub):
        j_sub = c_sub // n_sub
        i_sub = c_sub % n_sub
        c_full = (j0 + j_sub) * n + (i0 + i_sub)
        W_full[c_full, :] = W_sub[c_sub, :]
    
    return U_full, V_full, W_full

def _block_decomposition_356():
    """Block decomposition for 3x5 x 5x6.
    
    Split A into [A1 (3x3) | A2 (3x2)]
    Split B into [B1 (3x6); B2 (2x6)]
    Split B1 into [B1a (3x3) | B1b (3x3)]
    
    C = A1*B1a + A1*B1b + A2*B2
    
    Use Strassen (rank 7) for 2x2 blocks where possible.
    A1*B1a: 3x3 x 3x3 -> split into 2x2 blocks
      A1 = [A11(2x2) A12(2x1); A21(1x2) A22(1x1)]
      B1a = [B11(2x2) B12(2x1); B21(1x2) B22(1x1)]
      This gets complicated. Let me use schoolbook for 3x3 (rank 27) or try to find better.
    
    Actually, let me use a simpler approach:
    - A1*B1a: 3x3 x 3x3, use schoolbook rank 27
    - A1*B1b: 3x3 x 3x3, use schoolbook rank 27  
    - A2*B2: 3x2 x 2x6, use schoolbook rank 36
    Total: 90 (same as schoolbook)
    
    Better: Use Strassen for 2x2 sub-blocks.
    Split 3x3 into:
      A1 = [A_top(2x2) A_right(2x1); A_bottom(1x2) A_corner(1x1)]
      B1a = [B_top(2x2) B_right(2x1); B_bottom(1x2) B_corner(1x1)]
      
    C = A1*B1a = A_top*B_top + A_top*B_right + A_bottom*B_bottom + A_bottom*B_right + A_right*B_bottom + A_right*B_corner + A_corner*B_corner
    Wait, this is getting messy. Let me just use a cleaner block split.
    
    Alternative: Split 5 = 2+3, so:
    A = [A1(3x2) | A2(3x3)]
    B = [B1(2x6); B2(3x6)]
    C = A1*B1 + A2*B2
    A1*B1: 3x2 x 2x6, schoolbook rank 36
    A2*B2: 3x3 x 3x6
      Split B2 into B2a(3x3) and B2b(3x3)
      A2*B2a: 3x3 x 3x3, rank 27 (schoolbook)
      A2*B2b: 3x3 x 3x3, rank 27
    Total: 36+27+27 = 90. Still same.
    
    Let me try using Strassen for the 2x2 parts:
    A1*B1: 3x2 x 2x6
    Split A1 into rows: A1 = [A1_top(2x2); A1_bottom(1x2)]
    Split B1 into columns: B1 = [B1_left(2x3) B1_right(2x3)]
    C1 = A1_top*B1_left + A1_top*B1_right + A1_bottom*B1_left + A1_bottom*B1_right
    A1_top*B1_left: 2x2 x 2x3
    A1_top*B1_right: 2x2 x 2x3
    A1_bottom*B1_left: 1x2 x 2x3
    A1_bottom*B1_right: 1x2 x 2x3
    
    This is getting too complex. Let me just use the known result:
    For 3x5x6, a good decomposition might be around rank 67-70.
    
    Let me try:
    Split 5 = 2+3:
    A = [A1(3x2) | A2(3x3)]
    B = [B1(2x6); B2(3x6)]
    C = A1*B1 + A2*B2
    
    A1*B1: 3x2 x 2x6
    Use Strassen-like: split 3 rows into 2+1
    A1 = [A1a(2x2); A1b(1x2)]
    B1 = [B1a(2x3) B1b(2x3)]
    C1 = A1a*B1a + A1a*B1b + A1b*B1a + A1b*B1b
    A1a*B1a: 2x2 x 2x3 -> rank 7 (Strassen for 2x2 part) + 2 (for extra col) = 9? 
    Actually, 2x2 x 2x3 can be done with rank 7+2=9 or maybe less.
    
    This is too much manual work. Let me just use ALS to optimize from schoolbook.
    """
    n, m, p = 3, 5, 6
    U, V, W = _schoolbook(n, m, p)
    return U, V, W

def _als_update_U(U, V, W, valid, tol=1e-8, max_iter=100, deadline=None):
    """ALS update for U."""
    n, m, p = 3, 5, 6
    rank = U.shape[1]
    V = np.array(V)
    W = np.array(W)
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        U_new = np.zeros_like(U)
        for a in range(n * m):
            # U[a] dot (V[b] * W[c]) = 1 for all valid (a,b,c)
            # Collect equations
            rhs = []
            lhs_idx = []
            for (aa, b, c) in valid:
                if aa == a:
                    rhs.append(1.0)
                    lhs_idx.append((b, c))
            if not lhs_idx:
                continue
            # Solve least squares: U[a] @ M = rhs
            M = np.array([V[b] * W[c] for b, c in lhs_idx])
            sol, _, _, _ = np.linalg.lstsq(M.T, np.array(rhs), rcond=None)
            U_new[a] = sol
        U = U_new
    return U

def _als_update_V(U, V, W, valid, tol=1e-8, max_iter=100, deadline=None):
    """ALS update for V."""
    n, m, p = 3, 5, 6
    rank = U.shape[1]
    U = np.array(U)
    W = np.array(W)
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        V_new = np.zeros_like(V)
        for b in range(m * p):
            rhs = []
            lhs_idx = []
            for (a, bb, c) in valid:
                if bb == b:
                    rhs.append(1.0)
                    lhs_idx.append((a, c))
            if not lhs_idx:
                continue
            M = np.array([U[a] * W[c] for a, c in lhs_idx])
            sol, _, _, _ = np.linalg.lstsq(M.T, np.array(rhs), rcond=None)
            V_new[b] = sol
        V = V_new
    return V

def _als_update_W(U, V, W, valid, tol=1e-8, max_iter=100, deadline=None):
    """ALS update for W."""
    n, m, p = 3, 5, 6
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        W_new = np.zeros_like(W)
        for c in range(p * n):
            rhs = []
            lhs_idx = []
            for (a, b, cc) in valid:
                if cc == c:
                    rhs.append(1.0)
                    lhs_idx.append((a, b))
            if not lhs_idx:
                continue
            M = np.array([U[a] * V[b] for a, b in lhs_idx])
            sol, _, _, _ = np.linalg.lstsq(M.T, np.array(rhs), rcond=None)
            W_new[c] = sol
        W = W_new
    return W

def _try_reduce_rank(U, V, W, valid, n, m, p, start_rank, min_rank, deadline):
    """Try to reduce rank by removing zero/ near-zero columns."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    current_rank = start_rank
    for target_rank in range(start_rank, min_rank - 1, -1):
        if time.time() > deadline:
            break
        # Try to set column target_rank to zero and see if error stays small
        U_orig = U[:, target_rank].copy()
        V_orig = V[:, target_rank].copy()
        W_orig = W[:, target_rank].copy()
        
        U[:, target_rank] = 0
        V[:, target_rank] = 0
        W[:, target_rank] = 0
        
        # Run a few ALS iterations to see if we can recover
        U2 = _als_update_U(U, V, W, valid, max_iter=50, deadline=deadline)
        V2 = _als_update_V(U2, V, W, valid, max_iter=50, deadline=deadline)
        W2 = _als_update_W(U2, V2, W, valid, max_iter=50, deadline=deadline)
        
        # Check error
        max_err = 0.0
        for a, b, c in valid:
            val = np.sum(U2[a] * V2[b] * W2[c])
            max_err = max(max_err, abs(val - 1.0))
        all_entries = set()
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    all_entries.add((a, b, c))
        valid_set = set(valid)
        invalid = [(a, b, c) for (a, b, c) in all_entries if (a, b, c) not in valid_set]
        for a, b, c in invalid:
            val = np.sum(U2[a] * V2[b] * W2[c])
            max_err = max(max_err, abs(val))
        
        if max_err < 1e-4:
            # Successfully reduced rank
            U = U2[:, :target_rank]
            V = V2[:, :target_rank]
            W = W2[:, :target_rank]
            current_rank = target_rank
            print(f"Reduced rank to {current_rank}, error={max_err}")
        else:
            # Restore
            U[:, target_rank] = U_orig
            V[:, target_rank] = V_orig
            W[:, target_rank] = W_orig
    
    return U, V, W, current_rank

def construct():
    deadline = time.time() + 130.0
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    valid = _build_target_tensor(n, m, p)
    current_rank = n * m * p  # 90
    
    # Run ALS to optimize
    for it in range(20):
        if time.time() > deadline - 30:
            break
        U = _als_update_U(U, V, W, valid, max_iter=20, deadline=deadline)
        V = _als_update_V(U, V, W, valid, max_iter=20, deadline=deadline)
        W = _als_update_W(U, V, W, valid, max_iter=20, deadline=deadline)
    
    # Try to reduce rank
    if time.time() < deadline - 10:
        U, V, W, current_rank = _try_reduce_rank(U, V, W, valid, n, m, p, current_rank, 60, deadline)
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
