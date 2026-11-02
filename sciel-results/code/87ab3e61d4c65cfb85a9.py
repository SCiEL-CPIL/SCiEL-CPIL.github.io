"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from collections import defaultdict

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify exact decomposition."""
    r = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    r = len(U[0])
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, r)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, r)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, r)
    
    Ue = U_arr.reshape(n, m, 1, r)
    Ve = V_arr.reshape(1, m, p, r)
    We = W_arr.T.reshape(n, 1, p, r)
    
    T = np.sum(Ue * Ve * We, axis=3)
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using fractions to avoid floating point issues."""
    from fractions import Fraction
    r = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(r):
                    u_val = Fraction(U[i*m+j][c]).limit_denominator(1000)
                    v_val = Fraction(V[j*p+k][c]).limit_denominator(1000)
                    w_val = Fraction(W[k*n+i][c]).limit_denominator(1000)
                    s += u_val * v_val * w_val
                if s != Fraction(1):
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard O(nmp) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank

def _structured_rank1_terms(n, m, p):
    """
    Generate structured rank-1 terms using index arithmetic.
    Each term is defined by (U_row_pattern, V_row_pattern, W_row_pattern)
    where patterns are functions of indices.
    """
    terms = []
    
    # Basic terms: for each (i,j,k), a trivial rank-1 term
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_pat = [1 if (ii == i and jj == j) else 0 for ii in range(n) for jj in range(m)]
                V_pat = [1 if (jj == j and kk == k) else 0 for jj in range(m) for kk in range(p)]
                W_pat = [1 if (kk == k and ii == i) else 0 for kk in range(p) for ii in range(n)]
                terms.append((U_pat, V_pat, W_pat))
    
    # Strassen-like combinations
    # For each pair of indices, try to combine
    for i1 in range(n):
        for i2 in range(n):
            if i1 == i2:
                continue
            for j1 in range(m):
                for j2 in range(m):
                    if j1 == j2:
                        continue
                    for k1 in range(p):
                        for k2 in range(p):
                            if k1 == k2:
                                continue
                            # Try a combined term
                            U_pat = [0] * (n * m)
                            V_pat = [0] * (m * p)
                            W_pat = [0] * (p * n)
                            
                            U_pat[i1 * m + j1] = 1
                            U_pat[i2 * m + j2] = 1
                            V_pat[j1 * p + k1] = 1
                            V_pat[j2 * p + k2] = 1
                            W_pat[k1 * n + i1] = 1
                            W_pat[k2 * n + i2] = 1
                            
                            terms.append((U_pat, V_pat, W_pat))
    
    return terms

def _greedy_construct(n, m, p, max_rank, time_limit, rng):
    """
    Greedy construction: build decomposition column by column.
    At each step, find the best rank-1 term that reduces the residual.
    """
    start_time = time.time()
    
    # Initialize residual to all ones
    residual = np.ones((n, m, p), dtype=np.float64)
    
    U = []
    V = []
    W = []
    
    for step in range(max_rank):
        if time.time() - start_time > time_limit:
            break
        
        # Try all possible simple rank-1 terms
        best_term = None
        best_reduction = 0.0
        
        # Try all combinations of small integer patterns
        # Limit search space for efficiency
        max_coord = 2
        
        for ui in range(n):
            for uj in range(m):
                for vi in range(m):
                    for vk in range(p):
                        for wi in range(p):
                            for wj in range(n):
                                # Try term: U has 1 at (ui,uj), V at (vi,vk), W at (wi,wj)
                                # This only helps if ui==wj and the indices align
                                if ui != wj:
                                    continue
                                
                                # Compute contribution
                                contrib = np.zeros((n, m, p), dtype=np.float64)
                                contrib[ui, uj, vk] = 1.0 if (vi == uj) else 0.0
                                
                                # Only valid if vi == uj (V index j must match U index j)
                                if vi != uj:
                                    continue
                                
                                # Compute reduction in residual
                                reduction = np.sum(np.abs(residual) - np.abs(residual - contrib))
                                
                                if reduction > best_reduction:
                                    best_reduction = reduction
                                    best_term = (ui, uj, vi, vk, wi, wj)
        
        if best_term is None or best_reduction <= 1e-12:
            break
        
        ui, uj, vi, vk, wi, wj = best_term
        
        # Add this term
        U.append([0] * (n * m))
        V.append([0] * (m * p))
        W.append([0] * (p * n))
        
        U[-1][ui * m + uj] = 1
        V[-1][vi * p + vk] = 1
        W[-1][wi * n + wj] = 1
        
        # Update residual
        residual[ui, uj, vk] -= 1.0
        
        # Check if we're done
        if np.allclose(residual, 0, atol=1e-9):
            break
    
    if len(U) == 0:
        return None
    
    # Convert to required format
    r = len(U)
    U_mat = [[0] * r for _ in range(n * m)]
    V_mat = [[0] * r for _ in range(m * p)]
    W_mat = [[0] * r for _ in range(p * n)]
    
    for c in range(r):
        for i in range(n * m):
            U_mat[i][c] = U[c][i]
        for j in range(m * p):
            V_mat[j][c] = V[c][j]
        for k in range(p * n):
            W_mat[k][c] = W[c][k]
    
    return U_mat, V_mat, W_mat, r

def _strassen_style_3x4x4(n, m, p):
    """
    Try a Strassen-inspired decomposition for 3x4x4.
    Use block structure to reduce rank.
    """
    # For 3x4x4, we can try to exploit the rectangular structure
    # Split A into 3 rows, B into 4x4
    
    # Try rank = 20, 24, 28, etc.
    # Use a structured approach based on polynomial multiplication
    
    best = None
    
    for target_rank in range(16, 49):
        # Try to construct a valid decomposition
        # Use a heuristic based on index sums
        
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        # Pattern: for each (i,j,k), assign to a column based on (i+j+k) mod target_rank
        # But this won't work directly. Need more careful design.
        
        # Alternative: use a basis of monomials
        # U[i,j,c] = coefficient of x^i y^j in basis_c
        # V[j,k,c] = coefficient of y^j z^k in basis_c  
        # W[k,i,c] = coefficient of z^k x^i in basis_c
        
        # Try: basis_c = (x + a_c*y + b_c*z)^c for various a_c, b_c
        # This is too complex. Skip for now.
        
        pass
    
    return best

def _targeted_local_search(n, m, p, initial_U, initial_V, initial_W, time_limit, rng):
    """
    Local search around an initial solution.
    Try small perturbations to reduce rank.
    """
    start_time = time.time()
    best = (initial_U, initial_V, initial_W, len(initial_U[0]) if initial_U else 0)
    
    # Try removing columns one at a time
    r = len(initial_U[0])
    
    for remove_col in range(r):
        if time.time() - start_time > time_limit:
            break
        
        # Try removing column remove_col
        U_new = [[val for c, val in enumerate(row) if c != remove_col] for row in initial_U]
        V_new = [[val for c, val in enumerate(row) if c != remove_col] for row in initial_V]
        W_new = [[val for c, val in enumerate(row) if c != remove_col] for row in initial_W]
        
        if _verify_fast(U_new, V_new, W_new, n, m, p):
            best = (U_new, V_new, W_new, r - 1)
    
    return best

def _exact_greedy(n, m, p, max_rank, time_limit, rng):
    """
    Exact greedy: at each step, find a rank-1 term with integer/half-integer
    entries that exactly covers some of the remaining tensor entries.
    """
    start_time = time.time()
    
    # Track which (i,j,k) entries are still uncovered
    # Target: T[i,j,k] = 1 for all
    uncovered = set((i, j, k) for i in range(n) for j in range(m) for k in range(p))
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for step in range(max_rank):
        if time.time() - start_time > time_limit:
            break
        
        if not uncovered:
            break
        
        # Find a rank-1 term that covers some uncovered entries
        # A rank-1 term (u, v, w) covers (i,j,k) if u[i,j]*v[j,k]*w[k,i] != 0
        # And the sum over all terms must equal 1
        
        # For simplicity, try terms with entries in {-1, 0, 1}
        best_term = None
        best_coverage = 0
        
        # Limit search: try patterns based on index relationships
        for i_val in range(n):
            for j_val in range(m):
                for k_val in range(p):
                    # Term: U[i_val, j_val] = 1, V[j_val, k_val] = 1, W[k_val, i_val] = 1
                    # This covers exactly (i_val, j_val, k_val)
                    if (i_val, j_val, k_val) in uncovered:
                        coverage = 1
                        if coverage > best_coverage:
                            best_coverage = coverage
                            best_term = (i_val, j_val, k_val, 1, 1, 1)
        
        if best_term is None:
            break
        
        i_val, j_val, k_val, u_val, v_val, w_val = best_term
        
        U_cols.append([u_val if (i == i_val and j == j_val) else 0 for i in range(n) for j in range(m)])
        V_cols.append([v_val if (j == j_val and k == k_val) else 0 for j in range(m) for k in range(p)])
        W_cols.append([w_val if (k == k_val and i == i_val) else 0 for k in range(p) for i in range(n)])
        
        uncovered.discard((i_val, j_val, k_val))
    
    if not U_cols:
        return None
    
    r = len(U_cols)
    U_mat = [[U_cols[c][i] for c in range(r)] for i in range(n * m)]
    V_mat = [[V_cols[c][j] for c in range(r)] for j in range(m * p)]
    W_mat = [[W_cols[c][k] for c in range(r)] for k in range(p * n)]
    
    return U_mat, V_mat, W_mat, r

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    rng = random.Random(42)
    np.random.seed(42)
    
    # Baseline: schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    best = (U, V, W, rank)
    best_rank = rank
    
    # Theoretical lower bound
    min_possible = max(n * p, m * p, n * m)  # max(12, 16, 12) = 16
    
    # Strategy 1: Greedy exact construction
    greedy_time = 30
    result = _exact_greedy(n, m, p, min_possible + 10, greedy_time, rng)
    if result is not None:
        U_g, V_g, W_g, r_g = result
        if _verify_fast(U_g, V_g, W_g, n, m, p) and r_g < best_rank:
            best = (U_g, V_g, W_g, r_g)
            best_rank = r_g
    
    # Strategy 2: Local search around schoolbook (try removing columns)
    local_time = 30
    result = _targeted_local_search(n, m, p, U, V, W, local_time, rng)
    if result is not None:
        U_l, V_l, W_l, r_l = result
        if r_l < best_rank:
            best = (U_l, V_l, W_l, r_l)
            best_rank = r_l
    
    # Strategy 3: Structured search with small integer entries
    # Focus on ranks near the lower bound
    search_time_budget = 60
    elapsed = time.time() - start_time
    remaining = search_time_budget - elapsed
    
    if remaining > 10:
        for target_rank in range(min_possible, min(best_rank, min_possible + 15)):
            if time.time() - start_time > deadline - 5:
                break
            
            # Try random structured patterns
            max_attempts = 50
            
            for attempt in range(max_attempts):
                if time.time() - start_time > deadline - 5:
                    break
                
                # Generate structured random matrices
                # Use patterns based on index relationships
                U_rand = [[0] * target_rank for _ in range(n * m)]
                V_rand = [[0] * target_rank for _ in range(m * p)]
                W_rand = [[0] * target_rank for _ in range(p * n)]
                
                # Fill with small random integers
                for c in range(target_rank):
                    # Choose a pattern
                    pattern_type = rng.randint(0, 3)
                    
                    if pattern_type == 0:
                        # Diagonal-like
                        i_idx = rng.randint(0, n - 1)
                        j_idx = rng.randint(0, m - 1)
                        k_idx = rng.randint(0, p - 1)
                        U_rand[i_idx * m + j_idx][c] = rng.choice([-1, 1])
                        V_rand[j_idx * p + k_idx][c] = rng.choice([-1, 1])
                        W_rand[k_idx * n + i_idx][c] = rng.choice([-1, 1])
                    
                    elif pattern_type == 1:
                        # Row-based
                        i_idx = rng.randint(0, n - 1)
                        for j_idx in range(m):
                            U_rand[i_idx * m + j_idx][c] = rng.choice([-1, 0, 1])
                        for j_idx in range(m):
                            for k_idx in range(p):
                                V_rand[j_idx * p + k_idx][c] = rng.choice([-1, 0, 1])
                        for k_idx in range(p):
                            W_rand[k_idx * n + i_idx][c] = rng.choice([-1, 0, 1])
                    
                    elif pattern_type == 2:
                        # Column-based
                        j_idx = rng.randint(0, m - 1)
                        for i_idx in range(n):
                            U_rand[i_idx * m + j_idx][c] = rng.choice([-1, 0, 1])
                        for k_idx in range(p):
                            V_rand[j_idx * p + k_idx][c] = rng.choice([-1, 0, 1])
                        for k_idx in range(p):
                            for i_idx in range(n):
                                W_rand[k_idx * n + i_idx][c] = rng.choice([-1, 0, 1])
                    
                    else:
                        # Fully random small integers
                        for idx in range(n * m):
                            U_rand[idx][c] = rng.randint(-1, 1)
                        for idx in range(m * p):
                            V_rand[idx][c] = rng.randint(-1, 1)
                        for idx in range(p * n):
                            W_rand[idx][c] = rng.randint(-1, 1)
                
                if _verify_fast(U_rand, V_rand, W_rand, n, m, p):
                    best = (U_rand, V_rand, W_rand, target_rank)
                    best_rank = target_rank
                    break
    
    # Final verification
    U_best, V_best, W_best, _ = best
    
    # Ensure we're within time limit
    if time.time() - start_time > 140:
        U, V, W, rank = _schoolbook(n, m, p)
        U_best, V_best, W_best = U, V, W
    
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
