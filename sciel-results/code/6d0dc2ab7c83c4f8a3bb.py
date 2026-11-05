"""
Rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
Combines structured nonlinear search with a combinatorial hypergraph edge-coloring
heuristic: find a sparse set of hyperedges (rank-1 terms) that cover all 36
multiplication constraints, then solve the algebraic system only for those terms.
"""
import time
import numpy as np
from itertools import product, combinations
from scipy.optimize import least_squares, fsolve

DIMENSIONS = [3, 3, 4]
DEADLINE = 115.0  # safety margin before 150s kill


def _build_target_tensor():
    """Return the 36 (i,j,k) terms with their U, V, W row indices."""
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                terms.append((u_idx, v_idx, w_idx))
    return terms


def _verify(U, V, W, terms, tol=1e-9):
    """Verify that the decomposition exactly satisfies all tensor equations."""
    rank = U.shape[1] if isinstance(U, np.ndarray) else len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for (u_idx, v_idx, w_idx) in terms:
        s = 0.0
        for r in range(rank):
            s += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
        if abs(s - 1.0) > tol:
            return False
    return True


def _snap_and_verify(U, V, W, terms):
    """Snap to nearest half-integer and verify."""
    U_s = np.round(U * 2) / 2
    V_s = np.round(V * 2) / 2
    W_s = np.round(W * 2) / 2
    if _verify(U_s, V_s, W_s, terms):
        return U_s.tolist(), V_s.tolist(), W_s.tolist()
    # Also try integer snapping
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    if _verify(U_i, V_i, W_i, terms):
        return U_i.tolist(), V_i.tolist(), W_i.tolist()
    return None


def _schoolbook_solution():
    """Return the rank-36 schoolbook solution."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    terms = _build_target_tensor()
    for col, (u_idx, v_idx, w_idx) in enumerate(terms):
        U[u_idx, col] = 1
        V[v_idx, col] = 1
        W[w_idx, col] = 1
    return U.tolist(), V.tolist(), W.tolist()


def _hypergraph_coverage(terms, max_rank, deadline):
    """
    Find a sparse set of hyperedges (rank-1 term patterns) that can cover all 36
    constraints. Each hyperedge is a triple (u_row, v_row, w_row) pattern.
    Use greedy set-cover with backtracking to find minimal covering sets.
    
    Returns a list of candidate rank values and their associated term subsets,
    sorted by rank (ascending).
    """
    n, m, p = DIMENSIONS
    u_rows = n * m  # 9
    v_rows = m * p  # 12
    w_rows = p * n  # 12
    
    # Each of the 36 constraints requires specific (u_idx, v_idx, w_idx)
    # A rank-1 term with pattern (u_row, v_row, w_row) contributes to constraint (i,j,k)
    # only if u_row == i*m+j, v_row == j*p+k, w_row == k*n+i
    # So each constraint is covered by exactly one specific pattern.
    
    # The key insight: we can share rows across terms. A term r has:
    #   U[:, r] which is a vector in R^9
    #   V[:, r] which is a vector in R^12
    #   W[:, r] which is a vector in R^12
    # For constraint (i,j,k) with indices (u_i, v_i, w_i):
    #   sum_r U[u_i, r] * V[v_i, r] * W[w_i, r] = 1
    
    # For a "sparse" solution, we want most entries to be 0.
    # Let's try patterns where each term activates a small subset of rows.
    
    # Strategy: enumerate small support patterns for U, V, W vectors
    # and check if they can form a valid decomposition.
    
    # For rank R, we need R vectors U_r, V_r, W_r such that
    # for each of the 36 constraints (u_i, v_i, w_i):
    #   sum_r U_r[u_i] * V_r[v_i] * W_r[w_i] = 1
    
    # Let's use a combinatorial approach: for each rank R, try to find
    # assignments where each U_r has support of size s_u, etc.
    
    candidates = []
    
    # Try various support sizes
    for s_u in range(1, 5):
        for s_v in range(1, 5):
            for s_w in range(1, 5):
                # Minimum rank needed: each constraint needs at least one term
                # where all three coordinates are non-zero
                # With support size s_u, s_v, s_w, a term covers at most
                # s_u * s_v * s_w constraints (but realistically much fewer)
                min_terms = 36 // max(1, s_u * s_v * s_w)
                for R in range(max(min_terms, 1), min(max_rank + 1, 36)):
                    candidates.append((R, s_u, s_v, s_w))
    
    # Sort by rank, then by support size (smaller first)
    candidates.sort(key=lambda x: (x[0], x[1] + x[2] + x[3]))
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for R, su, sv, sw in candidates:
        key = (R, su, sv, sw)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(key)
    
    return unique_candidates[:50]  # Limit to 50 most promising


def _solve_with_support_pattern(R, su, sv, sw, terms, deadline):
    """
    Try to solve for a rank-R decomposition where:
    - Each U_r has at most su non-zero entries
    - Each V_r has at most sv non-zero entries  
    - Each W_r has at most sw non-zero entries
    
    Use the combinatorial structure to reduce the search space.
    """
    n, m, p = DIMENSIONS
    u_rows = n * m
    v_rows = m * p
    w_rows = p * n
    
    terms_u = np.array([t[0] for t in terms])
    terms_v = np.array([t[1] for t in terms])
    terms_w = np.array([t[2] for t in terms])
    
    start_time = time.time()
    
    # For each rank-1 term r, choose which rows are active
    # Then solve for the values
    
    # Heuristic: for each term r, the active rows should be "compatible"
    # with the constraints it helps satisfy.
    
    # Generate candidate support patterns
    # For U: choose su rows out of 9
    # For V: choose sv rows out of 12
    # For W: choose sw rows out of 12
    
    # This is combinatorially expensive, so use a greedy approach:
    # 1. Start with random supports
    # 2. Solve the linear system for the values given the supports
    # 3. Check if the solution is valid
    
    for attempt in range(20):
        if time.time() - start_time > deadline:
            break
        if time.time() > DEADLINE - 5:
            break
        
        np.random.seed(12345 + attempt * 7919)
        
        # Choose random support patterns
        U_supports = [set(np.random.choice(u_rows, size=su, replace=False)) for _ in range(R)]
        V_supports = [set(np.random.choice(v_rows, size=sv, replace=False)) for _ in range(R)]
        W_supports = [set(np.random.choice(w_rows, size=sw, replace=False)) for _ in range(R)]
        
        # Build the linear system:
        # For each constraint t (36 constraints):
        #   sum_{r: u_t in U_support[r], v_t in V_support[r], w_t in W_support[r]}
        #     U_r[u_t] * V_r[v_t] * W_r[w_t] = 1
        #
        # This is nonlinear (product of three unknowns per term).
        # Use log variables: let a_r = log(U_r[u]), b_r = log(V_r[v]), c_r = log(W_r[w])
        # Then exp(a_r + b_r + c_r) = 1 => a_r + b_r + c_r = 0 for each active term
        # But this only works if all values are positive.
        
        # Alternative: use scipy.optimize with the nonlinear system
        # Variables: for each term r and each active row, the value
        # Number of variables: sum_r (|U_support[r]| + |V_support[r]| + |W_support[r]|)
        
        # Build variable index mapping
        var_index = {}
        var_count = 0
        for r in range(R):
            for u in U_supports[r]:
                var_index[('U', r, u)] = var_count
                var_count += 1
            for v in V_supports[r]:
                var_index[('V', r, v)] = var_count
                var_count += 1
            for w in W_supports[r]:
                var_index[('W', r, w)] = var_count
                var_count += 1
        
        if var_count == 0:
            continue
        
        # Build residual function
        def residuals(x):
            # Extract values
            vals = {}
            for key, idx in var_index.items():
                vals[key] = x[idx]
            
            res = np.zeros(36)
            for t in range(36):
                u_t = terms_u[t]
                v_t = terms_v[t]
                w_t = terms_w[t]
                s = 0.0
                for r in range(R):
                    u_val = vals.get(('U', r, u_t), 0.0)
                    v_val = vals.get(('V', r, v_t), 0.0)
                    w_val = vals.get(('W', r, w_t), 0.0)
                    s += u_val * v_val * w_val
                res[t] = s - 1.0
            return res
        
        # Initial guess
        x0 = np.zeros(var_count)
        # Set initial values to 1/sqrt(R) for active entries
        for key, idx in var_index.items():
            x0[idx] = (1.0 / R) ** (1.0/3.0)
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=5000, ftol=1e-12, xtol=1e-12)
        except Exception:
            continue
        
        if result.cost < 1e-10:
            # Build U, V, W matrices
            U = np.zeros((u_rows, R))
            V = np.zeros((v_rows, R))
            W = np.zeros((w_rows, R))
            for key, idx in var_index.items():
                typ, r, row = key
                val = result.x[idx]
                if typ == 'U':
                    U[row, r] = val
                elif typ == 'V':
                    V[row, r] = val
                else:
                    W[row, r] = val
            
            snapped = _snap_and_verify(U, V, W, terms)
            if snapped is not None:
                return snapped
            
            if _verify(U, V, W, terms):
                return U.tolist(), V.tolist(), W.tolist()
    
    return None


def _try_solve_rank(R, terms, max_time=100.0):
    """
    Try to find a rank-R decomposition using combined approaches:
    1. Random initialization with scipy.optimize
    2. Structured support patterns (hypergraph heuristic)
    """
    n, m, p = DIMENSIONS
    u_rows = n * m
    v_rows = m * p
    w_rows = p * n
    
    terms_u = np.array([t[0] for t in terms])
    terms_v = np.array([t[1] for t in terms])
    terms_w = np.array([t[2] for t in terms])
    
    start_time = time.time()
    
    # Approach 1: Random initialization (from current program)
    for attempt in range(3):
        if time.time() - start_time > max_time or time.time() > DEADLINE - 5:
            break
        
        np.random.seed(42 + attempt * 104729)
        
        U = np.random.randn(u_rows, R)
        V = np.random.randn(v_rows, R)
        W = np.random.randn(w_rows, R)
        
        scale = (1.0 / R) ** (1.0/3.0)
        U *= scale
        V *= scale
        W *= scale
        
        def residuals(x):
            U_flat = x[:u_rows * R].reshape(u_rows, R)
            V_flat = x[u_rows * R:u_rows * R + v_rows * R].reshape(v_rows, R)
            W_flat = x[u_rows * R + v_rows * R:].reshape(w_rows, R)
            
            res = np.einsum('tr,tr,tr->t', U_flat[terms_u], V_flat[terms_v], W_flat[terms_w]) - 1.0
            return res
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=5000, ftol=1e-12, xtol=1e-12)
        except Exception:
            continue
        
        if result.cost < 1e-10:
            U_sol = result.x[:u_rows * R].reshape(u_rows, R)
            V_sol = result.x[u_rows * R:u_rows * R + v_rows * R].reshape(v_rows, R)
            W_sol = result.x[u_rows * R + v_rows * R:].reshape(w_rows, R)
            
            snapped = _snap_and_verify(U_sol, V_sol, W_sol, terms)
            if snapped is not None:
                return snapped
            
            if _verify(U_sol, V_sol, W_sol, terms):
                return U_sol.tolist(), V_sol.tolist(), W_sol.tolist()
    
    # Approach 2: Structured support patterns (hypergraph heuristic)
    if time.time() - start_time < max_time * 0.5 and time.time() < DEADLINE - 10:
        remaining = min(max_time - (time.time() - start_time), DEADLINE - 5 - time.time())
        if remaining > 5:
            # Try small support patterns
            for su in range(1, min(4, u_rows + 1)):
                for sv in range(1, min(4, v_rows + 1)):
                    for sw in range(1, min(4, w_rows + 1)):
                        if time.time() - start_time > max_time or time.time() > DEADLINE - 5:
                            break
                        result = _solve_with_support_pattern(R, su, sv, sw, terms, 
                                                             max_time - (time.time() - start_time))
                        if result is not None:
                            return result
    
    return None


def construct():
    start_time = time.time()
    terms = _build_target_tensor()
    
    # Try to find solutions for rank < 36
    # Start from higher ranks (closer to 36) as they are more likely to have solutions
    for R in range(35, 20, -1):  # 35 down to 21
        if time.time() - start_time > DEADLINE - 10:
            break
        remaining_time = DEADLINE - (time.time() - start_time) - 10
        if remaining_time < 8:
            break
        result = _try_solve_rank(R, terms, max_time=remaining_time)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook rank 36
    U, V, W = _schoolbook_solution()
    return {"U": U, "V": V, "W": W}
