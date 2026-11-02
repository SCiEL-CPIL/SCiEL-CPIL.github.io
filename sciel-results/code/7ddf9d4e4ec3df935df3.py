"""Shared-sparse skeleton approach for 3x3 x 3x8 matrix multiplication."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE = 110.0  # safety margin before 120s wall-clock limit
_start = None


def _elapsed():
    global _start
    if _start is None:
        _start = time.time()
    return time.time() - _start


def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    r = len(U[0])
    T = np.zeros((n, m, p))
    for c in range(r):
        Uc = np.array(U[:, c]).reshape(n, m)
        Vc = np.array(V[:, c]).reshape(m, p)
        Wc = np.array(W[:, c]).reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += Uc[i, j] * Vc[j, k] * Wc[k, i]
    return np.max(np.abs(T - 1.0)) < tol


def _rank_of(W, tol=1e-7):
    r = len(W)
    if r == 0:
        return 0
    Warr = np.array(W, dtype=float)
    sv = np.linalg.svd(Warr, compute_uv=False)
    return int(np.sum(sv > tol))


def _try_fixed_UV(U, V, n, m, p, tol=1e-7):
    """Given fixed U (nm x r) and V (mp x r), solve for W and return (W, rank) or None."""
    r = len(U[0])
    # Build linear system: for each (i,k), unknowns are W[k*n+i, c] for c in 0..r-1
    # Equation: sum_j sum_c V[j*p+k,c] * W[k*n+i,c] = A[i,j] * U[i*m+j,c]
    # Group RHS by c: for each (i,k,c): sum_j A[i,j] * U[i*m+j,c]
    # LHS: for each (i,k): sum_c V[j*p+k,c] * W[k*n+i,c]  (sum over j for each c)
    # So coefficient of W[k*n+i, c] in equation (i,k) is: sum_j A[i,j] * V[j*p+k, c]
    # where A[i,j]=1 for all i,j (identity-ish? no, A is arbitrary 3x3 input)
    # Actually the tensor identity is: for ALL input A, output B = A @ X.
    # The decomposition must satisfy: B[i,k] = sum_j sum_c A[i,j] * U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For this to equal sum_j A[i,j] * X[j,k] for arbitrary A, we need:
    #   sum_j A[i,j] * X[j,k] = sum_j sum_c A[i,j] * U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Matching coefficients of A[i,j]: X[j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # But X[j,k] must be independent of i! So the constraint is:
    #   For all i, j, k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = X[j,k] (independent of i)
    # This is a system where W is the unknown. Let me set up the linear system properly.
    #
    # Unknowns: W[k*n+i, c] for k in 0..p-1, i in 0..n-1, c in 0..r-1. Total: p*n*r unknowns.
    # Constraints: for all i, j, k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = X[j,k]
    # But X[j,k] is also unknown. We can eliminate X by requiring consistency:
    # For fixed (j,k), the value sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] must be the same for all i.
    # So constraints: for all i1 != i2, j, k:
    #   sum_c [U[i1*m+j,c] - U[i2*m+j,c]] * V[j*p+k,c] * W[k*n+i1,c] ... 
    # Hmm, this is getting complex. Let me use a different approach.
    #
    # Simpler: treat X as additional unknowns. 
    # Unknowns: W[k*n+i,c] (p*n*r of them) and X[j,k] (m*p of them).
    # Constraints: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - X[j,k] = 0
    # Total constraints: n*m*p = 72. Total unknowns: p*n*r + m*p.
    # This is underdetermined when r is small. We can solve and then minimize rank(W).
    #
    # Let me set up the linear system Ax = 0 where x = [W flattened, X flattened].
    n, m, p = n, m, p
    # Number of unknowns
    num_W = p * n * r
    num_X = m * p
    total_unknowns = num_W + num_X
    # Number of equations: n*m*p
    num_eq = n * m * p
    A = np.zeros((num_eq, total_unknowns))
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Equation: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - X[j,k] = 0
                for c in range(r):
                    coeff = U[i * m + j][c] * V[j * p + k][c]
                    if coeff != 0:
                        A[eq_idx, k * n * r + i * r + c] = coeff
                A[eq_idx, num_W + j * p + k] = -1.0
                eq_idx += 1
    # Solve Ax = 0. Find null space.
    # Use SVD to get null space basis
    U_svd, s, Vt = np.linalg.svd(A, full_matrices=True)
    # Null space: columns of Vt corresponding to zero singular values
    tol_ns = max(A.shape) * np.max(s) * 1e-10 if len(s) > 0 else 1e-10
    null_dim = len(s) - int(np.sum(s > tol_ns))
    if null_dim <= 0:
        return None
    null_space = Vt[int(np.sum(s > tol_ns)):, :].T  # shape (total_unknowns, null_dim)
    # General solution: x = sum_{a} lambda_a * null_space[:, a]
    # We want to choose lambdas to minimize rank(W).
    # For small null_dim, we can try random linear combinations and pick the one with lowest W rank.
    best_W = None
    best_rank = float('inf')
    rng = np.random.default_rng(42)
    n_trials = min(500, null_dim * 100 + 50)
    for trial in range(n_trials):
        if _elapsed() > DEADLINE:
            break
        lambdas = rng.standard_normal(null_dim)
        x = null_space @ lambdas
        W_flat = x[:num_W]
        W = W_flat.reshape(p * n, r)
        # Check if this W is valid (it should be by construction, but verify rank)
        wr = _rank_of(W)
        if wr < best_rank:
            best_rank = wr
            best_W = W.tolist()
    if best_W is None:
        return None
    return (best_W, best_rank)


def _try_fixed_UV_direct(U, V, n, m, p, tol=1e-7):
    """Direct approach: for fixed U, V, the constraint is that for each (j,k),
    the vector w_{j,k} = [sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for i] must be constant.
    Actually let me just set up and solve the full system."""
    r = len(U[0])
    # Unknowns: W[k*n+i, c] for all k,i,c. That's p*n*r unknowns.
    # Constraints: for each (j,k) and for each pair (i1, i2), the value
    #   sum_c U[i1*m+j,c]*V[j*p+k,c]*W[k*n+i1,c] = sum_c U[i2*m+j,c]*V[j*p+k,c]*W[k*n+i2,c]
    # This ensures independence from i.
    # Number of constraints: m * p * C(n,2) = 3*8*3 = 72
    # Number of unknowns: p*n*r = 24*r
    num_unknowns = p * n * r
    constraints = []
    for j in range(m):
        for k in range(p):
            for i1 in range(n):
                for i2 in range(i1 + 1, n):
                    row = np.zeros(num_unknowns)
                    for c in range(r):
                        row[k * n * r + i1 * r + c] = U[i1 * m + j][c] * V[j * p + k][c]
                        row[k * n * r + i2 * r + c] = -U[i2 * m + j][c] * V[j * p + k][c]
                    constraints.append(row)
    if not constraints:
        # No constraints, W is free. Use all zeros? But then output is 0, not valid.
        # Actually if there are no constraints, it means n=1, so no issue.
        W = [[0] * r for _ in range(p * n)]
        return (W, 0)
    A = np.array(constraints)
    # Solve Ax = 0
    try:
        U_svd, s, Vt = np.linalg.svd(A, full_matrices=True)
        tol_ns = max(A.shape) * np.max(s) * 1e-10 if len(s) > 0 else 1e-10
        null_dim = len(s) - int(np.sum(s > tol_ns))
        if null_dim <= 0:
            return None
        null_space = Vt[int(np.sum(s > tol_ns)):, :].T
        rng = np.random.default_rng(123)
        best_W = None
        best_rank = float('inf')
        n_trials = min(300, null_dim * 50 + 30)
        for trial in range(n_trials):
            if _elapsed() > DEADLINE:
                break
            lambdas = rng.standard_normal(null_dim)
            x = null_space @ lambdas
            W = x.reshape(p * n, r)
            wr = _rank_of(W)
            if wr < best_rank:
                best_rank = wr
                best_W = W
        if best_W is None:
            return None
        return (best_W.tolist(), best_rank)
    except Exception:
        return None


def _generate_skeletons(n, m, p, max_rank, time_limit):
    """Generate candidate (U, V) skeletons with low rank."""
    candidates = []
    # Strategy: U has nm rows, V has mp rows. Each has r columns.
    # We want U and V to be "sparse" in some sense.
    # Try: U[i*m+j, c] nonzero only for specific (i,j,c) patterns.
    # Simplest: U[i*m+j, c] = delta_{j,c} for c in 0..m-1 (i.e., U depends only on j)
    # and V[j*p+k, c] = delta_{k,c} for c in 0..p-1 (V depends only on k)
    # Then r = max(m, p) and W[k*n+i, c] must satisfy:
    #   For all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = X[j,k]
    #   = delta_{j,c} * delta_{k,c} * W[k*n+i,c] summed over c
    #   = W[k*n+i, j] * delta_{k,j}  (since c must equal both j and k, so j=k)
    # This only works if j=k, which isn't general.
    
    # Let me try a different skeleton: U[i*m+j, c] where c indexes something.
    # Actually, let me just enumerate small-rank U and V with entries in {-1, 0, 1}
    # and check solvability.
    
    # For efficiency, let's focus on r from 1 upward and try structured patterns.
    
    # Pattern 1: U[i*m+j, c] = 1 if c == j, else 0. (U depends only on j, rank m)
    # V[j*p+k, c] = 1 if c == j, else 0. (V depends only on j, rank m)
    # Then sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = W[k*n+i, j]
    # We need W[k*n+i, j] = X[j,k] for all i. So W[k*n+i, j] must be independent of i.
    # This is always satisfiable: set W[k*n+i, j] = X[j,k] for any i.
    # Rank of W: W is p*n x m. W[k*n+i, j] = X[j,k]. 
    # The rows of W: for fixed k, the row block [W[k*n+0, :], W[k*n+1, :], W[k*n+2, :]] 
    # all equal [X[0,k], X[1,k], X[2,k]]. So the rank is at most p (one row per k).
    # Actually the distinct rows are at most p, so rank <= p = 8.
    # Can we do better? The matrix W has rows: for each k, the same 3 rows [X[0,k], X[1,k], X[2,k]].
    # So W = [v_k; v_k; v_k] stacked for k=0..7, where v_k = [X[0,k], X[1,k], X[2,k]].
    # Rank of W = rank of [v_0; v_1; ...; v_7] which is rank of an 8x3 matrix = at most 3.
    # So rank(W) <= 3! Total rank = r = m = 3.
    # Let me verify: r = 3, U is 9x3, V is 24x3, W is 24x3.
    # U[i*m+j, c] = delta_{j,c}: U[i*3+j, c] = 1 if j==c else 0.
    # V[j*p+k, c] = delta_{j,c}: V[j*8+k, c] = 1 if j==c else 0.
    # W[k*n+i, c] = X[c, k] where X is arbitrary 3x8. Let X = identity-like or just pick X[j,k] = delta_{j,k} for k<3, 0 else.
    # Actually X can be anything since it's the "intermediate" product. Let's pick X[j,k] = 1 for all j,k.
    # Then W[k*n+i, c] = 1 for all k, i, c. So W is all ones, rank 1!
    # Let me verify the full decomposition with r=3:
    # T[i,j,k] = sum_c U[i*3+j,c] * V[j*8+k,c] * W[k*3+i,c]
    # = sum_c delta_{j,c} * delta_{j,c} * 1 = sum_c delta_{j,c} = 1. 
    # So this works with rank 3!
    
    candidates.append(('skeleton_A', 3))
    
    # Can we do rank 2 or 1?
    # Rank 1: U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1 for all i,j,k.
    # This requires U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = 1.
    # For fixed j,k: U[i*m+j,0] * W[k*n+i,0] = 1/V[j*p+k,0].
    # This must hold for all i. So U[i*m+j,0] * W[k*n+i,0] is constant in i.
    # Let U[i*m+j,0] = a_j (independent of i), W[k*n+i,0] = b_k (independent of i).
    # Then a_j * b_k = 1/V[j*p+k,0]. So V[j*p+k,0] = 1/(a_j * b_k).
    # This works! Set a_j = 1, b_k = 1, V[j*p+k,0] = 1.
    # Then U[i*m+j,0] = 1, V[j*p+k,0] = 1, W[k*n+i,0] = 1.
    # T[i,j,k] = 1*1*1 = 1. Rank 1!
    
    candidates.append(('skeleton_B', 1))
    
    return candidates


def construct():
    global _start
    _start = time.time()
    n, m, p = DIMENSIONS
    
    # Try rank 1 first
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    if _validate(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Try rank 2
    # Let me search for rank 2 decompositions.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] for c=0,1.
    # T[i,j,k] = U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] + U[i*m+j,1]*V[j*p+k,1]*W[k*n+i,1] = 1
    
    # Let me try a structured approach.
    # Try: U[i*m+j, 0] = 1, U[i*m+j, 1] = 0 (U only uses first component)
    # V[j*p+k, 0] = 1, V[j*p+k, 1] = 0
    # Then T = W[k*n+i, 0]. Need W[k*n+i, 0] = 1 for all k,i. Rank(W) = 1.
    # But then we only used 1 component effectively. Let me check: this is just rank 1.
    
    # Let me do a more systematic search for small ranks.
    # For rank r, we need to find U (nm x r), V (mp x r), W (pn x r) with entries in {-1, 0, 1, 0.5}
    # such that T[i,j,k] = 1 for all i,j,k.
    
    # Use the skeleton approach: fix U and V, solve for W.
    # Let me enumerate simple U and V patterns.
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Try the known rank-3 skeleton
    r = 3
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    for i in range(n):
        for j in range(m):
            U[i * m + j][j] = 1
    for j in range(m):
        for k in range(p):
            V[j * p + k][j] = 1
    # W[k*n+i, c] = X[c, k]. Pick X[c,k] = 1 for all.
    W = [[1] * r for _ in range(p * n)]
    if _validate(U, V, W, n, m, p):
        best_rank = r
        best_result = {"U": U, "V": V, "W": W}
    
    # Try to find rank 1 or 2
    # Rank 1 was verified above. Let me double-check.
    U1 = [[1] for _ in range(n * m)]
    V1 = [[1] for _ in range(m * p)]
    W1 = [[1] for _ in range(p * n)]
    if _validate(U1, V1, W1, n, m, p):
        return {"U": U1, "V": V1, "W": W1}
    
    # The rank 1 solution works! Return it.
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
