"""Combined: rank-18 base with aggressive rank-16/17 ALS search."""
import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [2, 2, 5]

def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1
    return T

def _check_decomposition(U, V, W, T):
    """Check if sum_r U_u,r * V_v,r * W_w,r == T_u,v,w for all u,v,w."""
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    result = np.zeros_like(T, dtype=np.float64)
    for r in range(U_a.shape[1]):
        result += np.einsum('u,v,w->uvw', U_a[:,r], V_a[:,r], W_a[:,r])
    return np.allclose(result, T.astype(np.float64), atol=1e-9)

def _build_rank18(n, m, p):
    """Build the known rank-18 Strassen+schoolbook decomposition."""
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    return U, V, W

def _als_solve_U(T, V, W, n, m, p, R):
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    # Precompute V^T W for each pair (v, w)
    VvW = np.einsum('vr,wr->vwr', V_a, W_a)
    VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
    
    # Solve all u rows at once using batched lstsq
    U = np.zeros((n*m, R), dtype=np.float64)
    # VvW_flat is (m*p*p*n, R), T_a is (n*m, m*p, p*n)
    # For each u, we need to solve VvW_flat @ x = T_a[u].reshape(-1)
    # This is equivalent to solving a single system with multiple RHS
    try:
        # Stack all b vectors
        B = T_a.reshape(n*m, -1)  # (n*m, m*p*p*n)
        # Solve VvW_flat^T @ VvW_flat @ X = VvW_flat^T @ B^T
        A = VvW_flat.T @ VvW_flat  # (R, R)
        # Regularize for numerical stability
        A += 1e-8 * np.eye(R)
        BT = B.T  # (m*p*p*n, n*m)
        # Solve A @ X = VvW_flat.T @ BT
        RHS = VvW_flat.T @ BT  # (R, n*m)
        X = np.linalg.solve(A, RHS)  # (R, n*m)
        U = X.T  # (n*m, R)
    except np.linalg.LinAlgError:
        # Fallback to individual lstsq
        for u in range(n*m):
            b = T_a[u].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
            U[u] = sol
    return U

def _als_solve_V(T, U, W, n, m, p, R):
    U_a = np.array(U, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    Uw = np.einsum('ur,wr->uwr', U_a, W_a)
    Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
    
    V = np.zeros((m*p, R), dtype=np.float64)
    try:
        B = T_a.transpose(1, 0, 2).reshape(m*p, -1)  # (m*p, n*m*p*n)
        A = Uw_flat.T @ Uw_flat  # (R, R)
        A += 1e-8 * np.eye(R)
        BT = B.T  # (n*m*p*n, m*p)
        RHS = Uw_flat.T @ BT  # (R, m*p)
        X = np.linalg.solve(A, RHS)  # (R, m*p)
        V = X.T  # (m*p, R)
    except np.linalg.LinAlgError:
        for v in range(m*p):
            b = T_a[:, v, :].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
            V[v] = sol
    return V

def _als_solve_W(T, U, V, n, m, p, R):
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    UV = np.einsum('ur,vr->uvr', U_a, V_a)
    UV_flat = UV.reshape(-1, R)
    
    W = np.zeros((p*n, R), dtype=np.float64)
    try:
        B = T_a.transpose(2, 0, 1).reshape(p*n, -1)  # (p*n, n*m*m*p)
        A = UV_flat.T @ UV_flat  # (R, R)
        A += 1e-8 * np.eye(R)
        BT = B.T  # (n*m*m*p, p*n)
        RHS = UV_flat.T @ BT  # (R, p*n)
        X = np.linalg.solve(A, RHS)  # (R, p*n)
        W = X.T  # (p*n, R)
    except np.linalg.LinAlgError:
        for w in range(p*n):
            b = T_a[:, :, w].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
            W[w] = sol
    return W

def _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, R, max_iters=300, tol=1e-12):
    U = np.array(U_init, dtype=np.float64)
    V = np.array(V_init, dtype=np.float64)
    W = np.array(W_init, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    prev_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = np.inf
    
    for it in range(max_iters):
        U = _als_solve_U(T_a, V, W, n, m, p, R)
        V = _als_solve_V(T_a, U, W, n, m, p, R)
        W = _als_solve_W(T_a, U, V, n, m, p, R)
        
        # Compute error efficiently
        result = np.zeros_like(T_a, dtype=np.float64)
        for r in range(R):
            result += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
        err = np.sqrt(np.sum((result - T_a)**2) / (n*m * m*p * p*n))
        
        # Track best solution
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    # Use the best solution found
    U, V, W = best_U, best_V, best_W
    
    # Check if we have a valid decomposition
    if _check_decomposition(U, V, W, T_a):
        # Try to snap to rational values with more denominators
        for denom in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            U_s = np.round(U * denom) / denom
            V_s = np.round(V * denom) / denom
            W_s = np.round(W * denom) / denom
            if _check_decomposition(U_s, V_s, W_s, T_a):
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _search_rank(n, m, p, T, R, deadline, U18, V18, W18):
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    # Strategy 1: For small R, try systematic backtracking over sparse integer patterns
    if R <= 17:
        result = _backtracking_search(T, n, m, p, R, deadline)
        if result is not None:
            return result
    
    # Strategy 2: Remove terms from rank 18 to get to rank R
    if R < 18:
        terms_to_remove = 18 - R
        schoolbook_indices = [14, 15, 16, 17]
        
        # Specific heuristic: remove schoolbook terms first
        if terms_to_remove <= len(schoolbook_indices):
            keep_indices = [i for i in range(18) if i not in schoolbook_indices[:terms_to_remove]]
            U_init = U18_a[:, keep_indices].copy()
            V_init = V18_a[:, keep_indices].copy()
            W_init = W18_a[:, keep_indices].copy()
            
            # Try multiple perturbation scales
            for scale in [0.01, 0.05, 0.1, 0.2]:
                for seed in range(8):
                    if time.time() > deadline:
                        return None
                    np.random.seed(seed + 1000 + int(scale * 100))
                    U_pert = U_init + np.random.randn(4, R) * scale
                    V_pert = V_init + np.random.randn(10, R) * scale
                    W_pert = W_init + np.random.randn(10, R) * scale
                    
                    result = _try_rank_als_from_init(U_pert, V_pert, W_pert, T, n, m, p, R, max_iters=100)
                    if result is not None:
                        return result

        # Try other combinations of removals with fewer random seeds
        for seed in range(20):
            if time.time() > deadline:
                return None
            np.random.seed(seed + 2000)
            keep_indices = np.random.choice(18, R, replace=False)
            U_init = U18_a[:, keep_indices].copy()
            V_init = V18_a[:, keep_indices].copy()
            W_init = W18_a[:, keep_indices].copy()
            
            for scale in [0.05, 0.1]:
                if time.time() > deadline:
                    return None
                np.random.seed(seed + 2100 + int(scale * 100))
                U_pert = U_init + np.random.randn(4, R) * scale
                V_pert = V_init + np.random.randn(10, R) * scale
                W_pert = W_init + np.random.randn(10, R) * scale
                
                result = _try_rank_als_from_init(U_pert, V_pert, W_pert, T, n, m, p, R, max_iters=80)
                if result is not None:
                    return result

    # Strategy 3: Random initialization with varied scales
    for seed in range(40):
        if time.time() > deadline:
            return None
        np.random.seed(seed + 3000)
        # Use different scales for diversity
        scale = [0.5, 1.0, 2.0][seed % 3]
        U_init = np.random.randn(4, R) * scale
        V_init = np.random.randn(10, R) * scale
        W_init = np.random.randn(10, R) * scale
        
        result = _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, R, max_iters=80)
        if result is not None:
            return result
            
    return None

def _backtracking_search(T, n, m, p, R, deadline):
    """Systematic backtracking search for sparse integer/half-integer rank-R decomposition."""
    T_a = np.array(T, dtype=np.float64)
    nm = n * m  # 4
    mp = m * p  # 10
    pn = p * n  # 10
    
    # Candidate values for entries (sparse: mostly 0, with -1, 1, -0.5, 0.5)
    candidate_vals = [0, 1, -1, 0.5, -0.5]
    
    # For efficiency, we'll build columns one at a time
    # U has shape (nm, R), V has shape (mp, R), W has shape (pn, R)
    
    U = np.zeros((nm, R), dtype=np.float64)
    V = np.zeros((mp, R), dtype=np.float64)
    W = np.zeros((pn, R), dtype=np.float64)
    
    # Precompute which (u,v,w) positions have T[u,v,w] = 1
    target_positions = []
    for u in range(nm):
        for v in range(mp):
            for w in range(pn):
                if T_a[u, v, w] != 0:
                    target_positions.append((u, v, w, T_a[u, v, w]))
    
    # We need: sum_r U[u,r] * V[v,r] * W[w,r] = T[u,v,w] for all (u,v,w)
    # Strategy: build column by column. For each column r, choose U[:,r], V[:,r], W[:,r]
    # from candidate values, then check if the partial sum is getting closer to T.
    
    # Better strategy: Use the fact that each target position (u,v,w) with T=1 
    # must be "covered" by at least one rank-1 term where U[u,r]*V[v,r]*W[w,r] != 0
    # and the sum over all such r equals 1.
    
    # Even better: enumerate possible support patterns for each column
    # A column r is defined by:
    #   support_U = {u : U[u,r] != 0}
    #   support_V = {v : V[v,r] != 0}  
    #   support_W = {w : W[w,r] != 0}
    #   values: for each u in support_U, U[u,r] in {±1, ±0.5}
    #           for each v in support_V, V[v,r] in {±1, ±0.5}
    #           for each w in support_W, W[w,r] in {±1, ±0.5}
    
    # The contribution of column r to position (u,v,w) is U[u,r]*V[v,r]*W[w,r]
    # which is 0 if u not in support_U or v not in support_V or w not in support_W
    
    # For a valid decomposition, for each target position (u,v,w) with T=1:
    #   sum over r of U[u,r]*V[v,r]*W[w,r] = 1
    # For each non-target position (u,v,w) with T=0:
    #   sum over r of U[u,r]*V[v,r]*W[w,r] = 0
    
    # This is a constraint satisfaction problem. Let's use a smarter approach:
    # Since the tensor is small, we can try to find a decomposition by solving
    # a system of linear equations given a fixed support pattern.
    
    # Approach: Enumerate support patterns (which entries are nonzero) and solve
    # for the actual values.
    
    # For rank R, each column has a support pattern. The total number of 
    # support pattern combinations is huge, so we need pruning.
    
    # Key insight: For the 2x2x5 case, the known optimal rank is 14.
    # Let's try a more targeted approach: use the structure of the problem.
    
    # The multiplication tensor for (2,2,5) has the property that:
    # - U has 4 rows, V has 10 rows, W has 10 rows
    # - Each "output" (i,k) pair for i in {0,1}, k in {0,1,2,3,4} 
    #   depends on inputs (i,j) and (j,k) for j in {0,1}
    
    # Let me try a different approach: use ALS with integer/half-integer constraints
    # by running ALS and then snapping to the nearest valid value, then verifying.
    
    # Actually, let's try a more direct approach: 
    # For small R, we can enumerate subsets of the 20 schoolbook terms and see
    # if we can combine them to get a lower rank.
    
    # The schoolbook decomposition has 20 terms. Each term r corresponds to
    # a unique (u,v,w) position where T=1. To get rank R < 20, we need to
    # "merge" some terms, which means some rank-1 terms must cover multiple
    # target positions.
    
    # A rank-1 term with U[u,r]=a, V[v,r]=b, W[w,r]=c contributes a*b*c to
    # position (u,v,w). If a=b=c=1, it contributes 1 to exactly one position.
    # If a=1, b=1, c=1 but we have multiple nonzero entries, it contributes to
    # multiple positions.
    
    # For example, if U[0,r]=1, U[1,r]=1, V[0,r]=1, W[0,r]=1, W[1,r]=1,
    # then this term contributes 1 to (0,0,0), (0,0,1), (1,0,0), (1,0,1).
    
    # This is getting complex. Let me try a simpler approach:
    # Use random restarts with ALS but enforce that the final solution has
    # entries in {0, ±0.5, ±1} by projecting after each iteration.
    
    # Projected ALS: after solving for U, project each entry to nearest value
    # in {0, ±0.5, ±1}. Same for V and W.
    
    best_result = None
    best_err = np.inf
    
    for seed in range(50):
        if time.time() > deadline:
            break
        np.random.seed(seed + 5000)
        
        # Initialize with random sparse patterns
        U_init = np.zeros((nm, R), dtype=np.float64)
        V_init = np.zeros((mp, R), dtype=np.float64)
        W_init = np.zeros((pn, R), dtype=np.float64)
        
        for r in range(R):
            # Randomly choose a few nonzero entries
            u_support = np.random.choice(nm, size=np.random.randint(1, 3), replace=False)
            v_support = np.random.choice(mp, size=np.random.randint(1, 3), replace=False)
            w_support = np.random.choice(pn, size=np.random.randint(1, 3), replace=False)
            
            for u in u_support:
                U_init[u, r] = np.random.choice([1, -1, 0.5, -0.5])
            for v in v_support:
                V_init[v, r] = np.random.choice([1, -1, 0.5, -0.5])
            for w in w_support:
                W_init[w, r] = np.random.choice([1, -1, 0.5, -0.5])
        
        # Run projected ALS
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for it in range(200):
            if time.time() > deadline:
                break
            
            # Solve for U
            U = _als_solve_U(T_a, V, W, n, m, p, R)
            # Project to nearest valid value
            U = np.round(U * 2) / 2
            U[np.abs(U) < 0.1] = 0
            U[U > 1] = 1
            U[U < -1] = -1
            
            # Solve for V
            V = _als_solve_V(T_a, U, W, n, m, p, R)
            V = np.round(V * 2) / 2
            V[np.abs(V) < 0.1] = 0
            V[V > 1] = 1
            V[V < -1] = -1
            
            # Solve for W
            W = _als_solve_W(T_a, U, V, n, m, p, R)
            W = np.round(W * 2) / 2
            W[np.abs(W) < 0.1] = 0
            W[W > 1] = 1
            W[W < -1] = -1
        
        # Check if we have a valid decomposition
        if _check_decomposition(U, V, W, T_a):
            # Snap to exact values
            U_s = np.round(U * 2) / 2
            V_s = np.round(V * 2) / 2
            W_s = np.round(W * 2) / 2
            if _check_decomposition(U_s, V_s, W_s, T_a):
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
        
        # Track best error
        result = np.zeros_like(T_a, dtype=np.float64)
        for r in range(R):
            result += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
        err = np.sqrt(np.sum((result - T_a)**2) / (nm * mp * pn))
        if err < best_err:
            best_err = err
            best_result = (U.copy(), V.copy(), W.copy())
    
    # If we didn't find an exact solution, try one more approach:
    # Use the rank-18 solution and try to zero out columns
    if best_result is not None and best_err < 1e-6:
        U_b, V_b, W_b = best_result
        # Try to identify which columns can be zeroed
        for r in range(R):
            U_temp = U_b.copy()
            V_temp = V_b.copy()
            W_temp = W_b.copy()
            U_temp[:, r] = 0
            V_temp[:, r] = 0
            W_temp[:, r] = 0
            # Remove zero columns
            nonzero_cols = [c for c in range(R) if np.any(U_b[:, c] != 0) or 
                           np.any(V_b[:, c] != 0) or np.any(W_b[:, c] != 0)]
            if len(nonzero_cols) < R:
                U_s = U_b[:, nonzero_cols]
                V_s = V_b[:, nonzero_cols]
                W_s = W_b[:, nonzero_cols]
                if _check_decomposition(U_s, V_s, W_s, T_a):
                    return U_s.tolist(), V_s.tolist(), W_s.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    # Use 130s deadline to ensure we finish well before the 150s kill
    deadline = time.time() + 130
    
    T = _build_target()
    
    # First, build the known rank-18 decomposition
    U18, V18, W18 = _build_rank18(n, m, p)
    
    # Verify rank-18
    if not _check_decomposition(U18, V18, W18, T):
        # This should not happen, but just in case
        return {"U": U18, "V": V18, "W": W18}

    # Try to find rank-14 first (known optimal rank)
    result14 = _search_rank(n, m, p, T, 14, deadline, U18, V18, W18)
    if result14 is not None:
        U, V, W = result14
        if _check_decomposition(U, V, W, T):
            return {"U": U, "V": V, "W": W}

    # Try to find rank-15
    result15 = _search_rank(n, m, p, T, 15, deadline, U18, V18, W18)
    if result15 is not None:
        U, V, W = result15
        if _check_decomposition(U, V, W, T):
            return {"U": U, "V": V, "W": W}

    # Try to find rank-16
    result16 = _search_rank(n, m, p, T, 16, deadline, U18, V18, W18)
    if result16 is not None:
        U, V, W = result16
        if _check_decomposition(U, V, W, T):
            return {"U": U, "V": V, "W": W}

    # Try to find rank-17
    result17 = _search_rank(n, m, p, T, 17, deadline, U18, V18, W18)
    if result17 is not None:
        U, V, W = result17
        if _check_decomposition(U, V, W, T):
            return {"U": U, "V": V, "W": W}

    # Fallback: return rank-18
    return {
        "U": U18,
        "V": V18,
        "W": W18
    }
