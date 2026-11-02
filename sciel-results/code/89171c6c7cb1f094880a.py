"""Matrix multiplication 2x3x10 - Rank-minimal decomposition via ILP search."""
import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
import random

DIMENSIONS = [2, 3, 10]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    
    r = len(U[0]) if U else 0
    if r == 0:
        return False
        
    # Convert to numpy for verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reshape to (n, m, r), (m, p, r), (p, n, r)
    U_reshaped = U_np.reshape(n, m, r)
    V_reshaped = V_np.reshape(m, p, r)
    W_reshaped = W_np.reshape(p, n, r)
    
    # Compute tensor T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Note: The output index in the problem is c = k*n + i.
    # The tensor is defined such that T[i, j, k] corresponds to the coefficient 
    # of A[i, j] * B[j, k] in C[i, k].
    # So we check if T[i, j, k] == 1 for all valid i, j, k.
    # And we must also ensure that the "non-target" parts are zero?
    # Actually, the tensor space is U_n x V_m x W_p.
    # The basis vectors are e_i, f_j, g_k.
    # The matmul tensor is T = sum_{i,j,k} e_i \otimes f_j \otimes g_{k*n+i}? 
    # No, the output index is c = k*n + i.
    # The tensor is T = sum_{i,j,k} e_i \otimes f_j \otimes h_{k*n+i}
    # where h_c are basis vectors for the output space of dimension p*n.
    
    # My previous verification logic was:
    # a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] should be 1 if (a,b,c) is a valid triple, 0 otherwise.
    
    # Let's use the einsum to compute the full contraction for valid indices first.
    try:
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        return False
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i,j,k] - 1) > 1e-9:
                    return False
    
    # Now check the zero entries.
    # The full tensor has dimensions (n*m, m*p, p*n).
    # We need T_full[a,b,c] = 0 for all (a,b,c) not of the form (i*m+j, j*p+k, k*n+i).
    
    # We can construct the full tensor T_full
    # T_full[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # This is a large tensor (6 * 30 * 20 = 3600 entries).
    # We can compute it efficiently using broadcasting or loops.
    
    # Let's just loop over the small number of non-zero checks?
    # No, we need to ensure ALL other entries are 0.
    # There are 3600 - 60 = 3540 entries to check.
    # Looping is fast enough.
    
    for a in range(n*m):
        for b in range(m*p):
            # Determine if there is a valid k for this a,b
            # a = i*m + j => i = a//m, j = a%m
            # b = j'*p + k => j' = b//p, k = b%p
            i = a // m
            j = a % m
            j_prime = b // p
            k = b % p
            
            # Valid if j == j_prime
            if j != j_prime:
                # This (a,b) pair never produces a valid output for any c?
                # Wait, c is determined by k and i.
                # If j != j_prime, then the term A[i,j] B[j',k] is not part of the matmul sum for C[i,k].
                # So T_full[a,b,c] must be 0 for ALL c.
                for c in range(p*n):
                    val = 0
                    for col in range(r):
                        val += U_np[a, col] * V_np[b, col] * W_np[c, col]
                    if abs(val) > 1e-9:
                        return False
            else:
                # j == j_prime, so this is a valid intermediate index.
                # The only valid c is c_valid = k*n + i.
                # For this c_valid, we already checked T[i,j,k] == 1.
                # For all other c, T_full[a,b,c] must be 0.
                c_valid = k * n + i
                for c in range(p*n):
                    if c == c_valid:
                        continue
                    val = 0
                    for col in range(r):
                        val += U_np[a, col] * V_np[b, col] * W_np[c, col]
                    if abs(val) > 1e-9:
                        return False
    return True

def _schoolbook_decomposition(n, m, p):
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    return U_sb, V_sb, W_sb

def _strassen_decomposition(n, m, p):
    """
    Returns the hand-constructed rank-55 decomposition.
    """
    # Re-implementing the logic from the current program to ensure we have a strong baseline
    # if ILP fails or takes too long.
    U_new = []
    V_new = []
    W_new = []
    
    cols_U = []
    cols_V = []
    cols_W = []
    
    # --- Term 2: A_right (2x1) @ B_bottom (1x10) ---
    for k in range(10):
        col_U = [0] * (n*m)
        col_V = [0] * (m*p)
        col_W = [0] * (p*n)
        
        col_U[2] = 1
        col_V[20+k] = 1
        col_W[k*2+0] = 1
        
        cols_U.append(col_U)
        cols_V.append(col_V)
        cols_W.append(col_W)
        
        col_U[5] = 1
        # col_V is same
        col_W[k*2+1] = 1
        
        cols_U.append(col_U)
        cols_V.append(col_V)
        cols_W.append(col_W)
        
    # --- Term 1: A_left (2x2) @ B_top (2x10) ---
    for q in range(5):
        k0 = 2*q
        k1 = 2*q + 1
        
        def add_col(u_vals, v_vals, w_vals):
            col_U = [0] * (n*m)
            col_V = [0] * (m*p)
            col_W = [0] * (p*n)
            for idx, val in u_vals.items():
                col_U[idx] += val
            for idx, val in v_vals.items():
                col_V[idx] += val
            for idx, val in w_vals.items():
                col_W[idx] += val
            cols_U.append(col_U)
            cols_V.append(col_V)
            cols_W.append(col_W)
            
        add_col({0:1, 4:1}, {k0:1, 10+k0:1}, {k0*2:1, k1*2+1:1})
        add_col({3:1, 4:1}, {k0:1}, {k0*2:-1, k0*2+1:1})
        add_col({0:1}, {k1:1, 10+k1:-1}, {k1*2:1, k1*2+1:1})
        add_col({4:1}, {10+k0:1, 10+k1:-1}, {k0*2:1, k1*2+1:-1})
        add_col({0:1, 1:1}, {10+k1:1}, {k1*2:1, k1*2+1:-1})
        add_col({3:1, 4:-1}, {k0:1, k1:1}, {k0*2+1:1, k1*2+1:-1})
        add_col({1:1, 4:-1}, {10+k0:1, 10+k1:1}, {k0*2:1, k1*2+1:-1})
        
    rank_found = len(cols_U)
    U_mat = [[cols_U[c][r] for c in range(rank_found)] for r in range(n*m)]
    V_mat = [[cols_V[c][r] for c in range(rank_found)] for r in range(m*p)]
    W_mat = [[cols_W[c][r] for c in range(rank_found)] for r in range(p*n)]
    
    return U_mat, V_mat, W_mat

def _random_search_rank(n, m, p, rank, time_limit=10.0):
    """
    Randomly search for a decomposition of the given rank.
    Uses random integer/half-integer entries.
    """
    start_time = time.time()
    n_rows_u = n * m
    n_rows_v = m * p
    n_rows_w = p * n
    
    # Generate random matrices with entries in {-1, 0, 1} or {-1, -0.5, 0, 0.5, 1}
    # To make it more likely to find exact solutions, we can try to satisfy constraints.
    # But pure random is a baseline.
    
    # A smarter random search: 
    # Pick random U, V. Compute W to satisfy equations?
    # The equations are linear in W for fixed U, V.
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # For each (a,b), we have a system of linear equations for W[*,r].
    # This is complex. Let's stick to random sampling with verification.
    
    # However, the space is huge. 
    # Let's try a "local search" or "simulated annealing" like approach?
    # Or just random restarts with a small number of non-zeros per row.
    
    # Strategy: 
    # 1. Initialize U, V, W with sparse random integers.
    # 2. Compute error.
    # 3. Perturb.
    
    # Given time limits, simple random restarts might be too slow to find rank < 55.
    # But we have 120s.
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Use numpy for speed
    while time.time() - start_time < time_limit:
        U = np.random.randint(-2, 3, size=(n_rows_u, rank))
        V = np.random.randint(-2, 3, size=(n_rows_v, rank))
        W = np.random.randint(-2, 3, size=(n_rows_w, rank))
        
        # Verify
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
            
    return None

def _ilp_search_rank(n, m, p, rank, time_limit=100.0):
    """
    Attempt to find a decomposition of the given rank using ILP.
    Variables: U[a,r], V[b,r], W[c,r] in {-1, 0, 1} (or extended set).
    Constraints:
    1. For each valid (i,j,k): sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    2. For each invalid (a,b,c): sum_r U[a, r] V[b, r] W[c, r] = 0
    
    This is a cubic program (product of 3 variables).
    Standard ILP solvers handle linear constraints.
    We can linearize by introducing auxiliary variables?
    Let X[a,b,c,r] = U[a,r] V[b,r] W[c,r].
    This is too many variables.
    
    Alternative: Fix U and V, solve for W?
    If U and V are fixed, the constraints are linear in W.
    sum_r (U[a,r] V[b,r]) W[c,r] = target(a,b,c)
    Let M_{ab, c, r} = U[a,r] V[b,r].
    Then sum_r M_{ab, c, r} W[c,r] = target(a,b,c).
    This is a system of linear equations for W.
    We can search over sparse U and V, and check if the resulting linear system for W has a solution in the allowed set (e.g., integers or half-integers).
    """
    start_time = time.time()
    n_rows_u = n * m
    n_rows_v = m * p
    n_rows_w = p * n
    
    # Generate target tensor
    targets = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                targets[(a,b,c)] = 1
    
    # We will randomly generate sparse U and V, and solve for W.
    # W must be such that for all (a,b,c), sum_r U[a,r]V[b,r]W[c,r] = T(a,b,c).
    
    # Let's iterate over random sparse U and V.
    # Sparsity: each row has ~3-5 non-zeros.
    
    while time.time() - start_time < time_limit:
        # Generate random sparse U
        U = np.zeros((n_rows_u, rank))
        for i in range(n_rows_u):
            nnz = random.randint(2, 5)
            idxs = random.sample(range(rank), nnz)
            for idx in idxs:
                U[i, idx] = random.choice([-1, 1])
                
        V = np.zeros((n_rows_v, rank))
        for i in range(n_rows_v):
            nnz = random.randint(2, 5)
            idxs = random.sample(range(rank), nnz)
            for idx in idxs:
                V[i, idx] = random.choice([-1, 1])
                
        # Now we have a linear system for W.
        # Variables: W[c, r] for c in 0..p*n-1, r in 0..rank-1.
        # Total variables: p*n*rank.
        
        # Equations:
        # For each (a,b), we have p*n equations for W[*,r].
        # Wait, the equation is for a specific (a,b,c).
        # sum_r U[a,r] V[b,r] W[c,r] = T(a,b,c).
        
        # This couples W[c,r] for different c? No, each equation is for a specific c.
        # So for each c, we have a system?
        # No, the sum is over r.
        # For fixed a,b,c:
        # sum_r (U[a,r] V[b,r]) W[c,r] = T(a,b,c)
        
        # Let M_{ab} be a vector of length rank: M_{ab}[r] = U[a,r] V[b,r].
        # Then M_{ab} . W[c,:] = T(a,b,c).
        
        # This means that for each c, the vector W[c,:] must satisfy:
        # M_{ab} . W[c,:] = T(a,b,c) for all a,b such that (a,b,c) is in the support?
        # No, the equation must hold for ALL a,b.
        # So for a fixed c, W[c,:] is a vector x such that:
        # M_{ab} . x = T(a,b,c) for all a,b.
        
        # This is a system of (n*m * m*p) equations for rank unknowns.
        # Usually overdetermined.
        
        # We can solve this using least squares or check consistency.
        # If consistent, we get a unique W[c,:] (or a family).
        # We then check if W[c,:] has allowed values (integers/half-integers).
        
        # Let's build the matrix for each c.
        # Rows: indexed by (a,b).
        # Cols: indexed by r.
        # A_c[a*b + b, r] = U[a,r] * V[b,r]
        # b_c[a*b + b] = T(a,b,c)
        
        # We need to solve A_c x = b_c.
        
        valid = True
        W_candidate = np.zeros((n_rows_w, rank))
        
        for c in range(n_rows_w):
            # Build A_c
            A_c = np.zeros((n_rows_u * n_rows_v, rank))
            b_c = np.zeros(n_rows_u * n_rows_v)
            
            row_idx = 0
            for a in range(n_rows_u):
                for b in range(n_rows_v):
                    # Determine if (a,b,c) is a target
                    # a = i*m+j, b = j'*p+k
                    i = a // m
                    j = a % m
                    j_prime = b // p
                    k = b % p
                    
                    if j == j_prime:
                        # Check if c == k*n + i
                        if c == k * n + i:
                            target = 1
                        else:
                            target = 0
                    else:
                        target = 0
                        
                    A_c[row_idx, :] = U[a, :] * V[b, :]
                    b_c[row_idx] = target
                    row_idx += 1
                    
            # Solve A_c x = b_c
            # Use least squares
            try:
                x, residuals, rank_c, sv = np.linalg.lstsq(A_c, b_c, rcond=None)
            except np.linalg.LinAlgError:
                valid = False
                break
                
            # Check if residual is small
            if residuals and residuals[0] > 1e-9:
                valid = False
                break
                
            # Check if x is close to integer or half-integer
            # We can round to nearest half-integer and check.
            x_rounded = np.round(x * 2) / 2.0
            if np.linalg.norm(x_rounded - x) > 1e-9:
                # Maybe it's an integer?
                x_int = np.round(x)
                if np.linalg.norm(x_int - x) > 1e-9:
                    valid = False
                    break
                else:
                    W_candidate[c, :] = x_int
            else:
                W_candidate[c, :] = x_rounded
                
        if not valid:
            continue
            
        # Verify the candidate
        if _verify_decomposition(U.tolist(), V.tolist(), W_candidate.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W_candidate.tolist()
            
    return None

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Generate schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p

    # Generate Strassen as fallback (Rank 55)
    U_str, V_str, W_str = _strassen_decomposition(n, m, p)
    if _verify_decomposition(U_str, V_str, W_str, n, m, p):
        best_U, best_V, best_W = U_str, V_str, W_str
        best_rank = len(U_str[0])

    # Try ILP search for rank 54
    if time.time() < deadline - 60:
        res = _ilp_search_rank(n, m, p, 54, time_limit=50.0)
        if res:
            U_new, V_new, W_new = res
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = 54
                return {"U": best_U, "V": best_V, "W": best_W}

    # Try ILP search for rank 53
    if time.time() < deadline - 30:
        res = _ilp_search_rank(n, m, p, 53, time_limit=20.0)
        if res:
            U_new, V_new, W_new = res
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                best_U, best_V, W_new = U_new, V_new, W_new
                best_rank = 53
                return {"U": best_U, "V": best_V, "W": best_W}

    # Try random search for rank 54 (quick)
    if time.time() < deadline - 10:
        res = _random_search_rank(n, m, p, 54, time_limit=5.0)
        if res:
            U_new, V_new, W_new = res
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = 54
                return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
