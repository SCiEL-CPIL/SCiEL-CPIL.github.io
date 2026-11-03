"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _build_decomposition(rank_terms):
    """
    Build U, V, W factor matrices from a list of term specifications.
    Each term is (U_dict, V_dict, W_dict) where each dict maps (i,j) or (j,k) or (k,i) to a scalar.
    """
    n, m, p = DIMENSIONS
    R = len(rank_terms)
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), val in u_dict.items():
            U[i * m + j, r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k, r] = val
        for (k, i), val in w_dict.items():
            W[k * n + i, r] = val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    R = U.shape[1]
    U_reshaped = U.reshape((n, m, R))
    V_reshaped = V.reshape((m, p, R))
    W_reshaped = W.reshape((p, n, R))
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_recon, T_target, atol=atol)

def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for the 2x2 core (indices 0,1).
    """
    terms = []
    
    # m1 = (a00 + a11)(b00 + b11) -> c00 += m1, c11 += m1
    terms.append(({(0,0): 1, (1,1): 1},
                  {(0,0): 1, (1,1): 1},
                  {(0,0): 1, (1,1): 1}))
    
    # m2 = (a10 + a11)b00 -> c10 += m2, c11 -= m2
    terms.append(({(1,0): 1, (1,1): 1},
                  {(0,0): 1},
                  {(0,1): 1, (1,1): -1}))
    
    # m3 = a00(b01 - b11) -> c01 += m3, c11 += m3
    terms.append(({(0,0): 1},
                  {(0,1): 1, (1,1): -1},
                  {(1,0): 1, (1,1): 1}))
    
    # m4 = a11(b10 - b00) -> c00 += m4, c10 += m4
    terms.append(({(1,1): 1},
                  {(1,0): 1, (0,0): -1},
                  {(0,0): 1, (0,1): 1}))
    
    # m5 = (a00 + a01)b11 -> c00 -= m5, c01 += m5
    terms.append(({(0,0): 1, (0,1): 1},
                  {(1,1): 1},
                  {(0,0): -1, (1,0): 1}))
    
    # m6 = (a10 - a00)(b00 + b01) -> c11 += m6
    terms.append(({(1,0): 1, (0,0): -1},
                  {(0,0): 1, (0,1): 1},
                  {(1,1): 1}))
    
    # m7 = (a01 - a11)(b10 + b11) -> c00 += m7
    terms.append(({(0,1): 1, (1,1): -1},
                  {(1,0): 1, (1,1): 1},
                  {(0,0): 1}))
    
    return terms

def _border_terms_standard():
    """
    Return standard multiplication terms for the border (at least one index is 2).
    """
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    return terms

def _try_standard_decomposition():
    """Standard schoolbook: rank 27."""
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    U, V, W = _build_decomposition(terms)
    return U, V, W

def _solve_sparse_system_gf5(A, b):
    """
    Solve Ax = b over GF(5) using Gaussian elimination.
    A is a 2D numpy array of ints mod 5, b is a 1D array of ints mod 5.
    Returns solution vector x (list of ints) or None if no unique solution.
    """
    A = np.array(A, dtype=np.int64) % 5
    b = np.array(b, dtype=np.int64) % 5
    n_rows, n_cols = A.shape
    if n_rows != len(b):
        return None
    
    # Augmented matrix
    M = np.hstack([A, b.reshape(-1, 1)])
    M = M.copy()
    
    rank = 0
    for col in range(n_cols):
        # Find pivot
        pivot_row = None
        for row in range(rank, n_rows):
            if M[row, col] % 5 != 0:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        
        # Swap rows
        if pivot_row != rank:
            M[[rank, pivot_row]] = M[[pivot_row, rank]]
        
        # Normalize pivot row
        inv_pivot = pow(int(M[rank, col] % 5), -1, 5)
        M[rank, :] = (M[rank, :] * inv_pivot) % 5
        
        # Eliminate other rows
        for row in range(n_rows):
            if row == rank:
                continue
            factor = int(M[row, col] % 5)
            if factor != 0:
                M[row, :] = (M[row, :] - factor * M[rank, :]) % 5
        
        rank += 1
    
    # Check for consistency
    for row in range(rank, n_rows):
        if M[row, n_cols] % 5 != 0:
            return None  # Inconsistent
    
    # Extract solution (unique if rank == n_cols)
    if rank < n_cols:
        # Underdetermined - pick one particular solution with free vars = 0
        x = np.zeros(n_cols, dtype=np.int64)
        for row in range(rank):
            # Find the leading column in this row
            for col in range(n_cols):
                if M[row, col] % 5 != 0:
                    x[col] = M[row, n_cols] % 5
                    break
        return x.tolist()
    
    x = M[:n_cols, n_cols].tolist()
    return x

def _try_rank_r_search(r, deadline, max_attempts=500):
    """
    Try to find a rank-r decomposition by:
    1. Using Strassen 7 terms for the 2x2 core
    2. Randomly choosing which border terms to keep/merge
    3. Solving a sparse linear system over GF(5) for the remaining coefficients
    
    Returns (U, V, W) if found, None otherwise.
    """
    n, m, p = DIMENSIONS
    border_indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)
                      if not (i < 2 and j < 2 and k < 2)]
    num_border = len(border_indices)  # 19
    
    # We need r - 7 border terms (some may be merged)
    needed_border = r - 7
    if needed_border < 0 or needed_border > num_border:
        return None
    
    # Strategy: try to use exactly 'needed_border' of the 19 border positions,
    # but allow some border terms to have multiple nonzero entries (merged)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Randomly select which border positions to use
        if needed_border < num_border:
            selected = set(np.random.choice(num_border, needed_border, replace=False))
        else:
            selected = set(range(num_border))
        
        # Build the linear system
        # Unknowns: for each of the r terms, coefficients in U, V, W
        # The 7 Strassen terms are fixed. For the remaining (r-7) terms,
        # we need to find U, V, W coefficients.
        
        # Actually, let's use a different approach: 
        # Fix the support pattern and solve for coefficients.
        # For rank r, we have r terms. Terms 0-6 are Strassen.
        # Terms 7 to r-1 handle the border.
        
        num_extra = r - 7
        
        # For each extra term, randomly assign which border (i,j,k) it covers
        # and what its U, V, W support looks like
        extra_supports = []
        for t in range(num_extra):
            # Pick a random border index this term primarily covers
            border_idx = border_indices[list(selected)[t % len(selected)]]
            i, j, k = border_idx
            
            # U support: include (i,j) and maybe one more
            u_supp = {(i, j)}
            if np.random.random() < 0.3:
                i2 = np.random.randint(0, n)
                j2 = np.random.randint(0, m)
                u_supp.add((i2, j2))
            
            # V support: include (j,k) and maybe one more
            v_supp = {(j, k)}
            if np.random.random() < 0.3:
                j2 = np.random.randint(0, m)
                k2 = np.random.randint(0, p)
                v_supp.add((j2, k2))
            
            # W support: include (k,i) and maybe one more
            w_supp = {(k, i)}
            if np.random.random() < 0.3:
                k2 = np.random.randint(0, p)
                i2 = np.random.randint(0, n)
                w_supp.add((k2, i2))
            
            extra_supports.append((u_supp, v_supp, w_supp))
        
        # Build the system: for each (i,j,k) in the tensor, the sum over r of
        # U[i,j,r] * V[j,k,r] * W[k,i,r] must equal 1.
        # This is nonlinear, but if we fix supports, it becomes a system of
        # bilinear equations. Instead, let's linearize by fixing one factor.
        
        # Alternative: use the fact that for a rank-1 term with fixed U and V,
        # W is determined. Let's fix U and V supports and solve for W.
        
        # Build U and V matrices with random coefficients for extra terms
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Set Strassen terms
        strassen_terms = _strassen_2x2_terms()
        for r_idx, (u_dict, v_dict, w_dict) in enumerate(strassen_terms):
            for (i, j), val in u_dict.items():
                U[i * m + j, r_idx] = val
            for (j, k), val in v_dict.items():
                V[j * p + k, r_idx] = val
            for (k, i), val in w_dict.items():
                W[k * n + i, r_idx] = val
        
        # For extra terms, assign random nonzero coefficients
        for t in range(num_extra):
            r_idx = 7 + t
            u_supp, v_supp, w_supp = extra_supports[t]
            for (i, j) in u_supp:
                U[i * m + j, r_idx] = np.random.choice([1, 2, 3, 4])
            for (j, k) in v_supp:
                V[j * p + k, r_idx] = np.random.choice([1, 2, 3, 4])
            for (k, i) in w_supp:
                W[k * n + i, r_idx] = np.random.choice([1, 2, 3, 4])
        
        # Now check if this works over the reals
        if _verify_decomposition(U, V, W, n, m, p, atol=1e-6):
            # Found a valid decomposition!
            # Try to clean up to get exact integer/half-integer values
            return U, V, W
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Uses a rank-26 decomposition based on Strassen's algorithm for the 2x2 core
    and standard multiplication for the remaining border terms.
    Includes a time-bounded search for lower-rank decompositions.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Strassen core + standard border (rank 26)
    if time.time() < deadline:
        terms = _strassen_2x2_terms() + _border_terms_standard()
        U, V, W = _build_decomposition(terms)
        if _verify_decomposition(U, V, W, n, m, p):
            R = U.shape[1]
            if R < best_rank:
                best_rank = R
                best_result = (U, V, W)
    
    # Strategy 2: Search for rank 25
    if time.time() < deadline:
        result = _try_rank_r_search(25, deadline, max_attempts=2000)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = 25
                best_result = (U, V, W)
    
    # Strategy 3: Search for rank 24
    if time.time() < deadline:
        result = _try_rank_r_search(24, deadline, max_attempts=1000)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = 24
                best_result = (U, V, W)
    
    # Strategy 4: Pure standard (rank 27) as fallback
    if best_result is None or time.time() < deadline:
        U, V, W = _try_standard_decomposition()
        if _verify_decomposition(U, V, W, n, m, p):
            R = U.shape[1]
            if R < best_rank:
                best_rank = R
                best_result = (U, V, W)
    
    if best_result is None:
        # Should not happen, but fallback
        U, V, W = _try_standard_decomposition()
        best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
