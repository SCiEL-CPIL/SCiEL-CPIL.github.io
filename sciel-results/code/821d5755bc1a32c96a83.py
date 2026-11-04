"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _verify_full(U, V, W, n, m, p):
    """Full verification using Fraction for exactness."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                if s != 1:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135  # 135s deadline, 15s safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 6
    rank_schoolbook = n * m * p  # 90
    
    # Build Strassen-based decomposition for rank 84
    # Partition:
    # n (rows of A): [2, 1]
    # m (cols of A / rows of B): [2, 2, 1]
    # p (cols of B): [2, 2, 2]
    
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2]
    
    # Track offsets for each dimension
    n_offsets = [0]
    for i in range(len(n_parts) - 1):
        n_offsets.append(n_offsets[-1] + n_parts[i])
    
    m_offsets = [0]
    for i in range(len(m_parts) - 1):
        m_offsets.append(m_offsets[-1] + m_parts[i])
    
    p_offsets = [0]
    for i in range(len(p_parts) - 1):
        p_offsets.append(p_offsets[-1] + p_parts[i])
    
    terms = []  # each term: (list of (u_row, coeff), list of (v_row, coeff), list of (w_row, coeff))
    
    def add_strassen(a_rows, c_cols, d_cols):
        """Add 7 Strassen terms for a 2x2x2 block."""
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0, d1 = d_cols
        
        def U(la, lc): return a_rows[la]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # C[0,0] += 1, C[1,1] += 1
        terms.append((
            [(U(0,0), 1), (U(1,1), 1)],
            [(V(0,0), 1), (V(1,1), 1)],
            [(W(0,0), 1), (W(1,1), 1)]
        ))
        
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        # C[1,0] += 1, C[1,1] += -1
        terms.append((
            [(U(1,0), 1), (U(1,1), 1)],
            [(V(0,0), 1)],
            [(W(0,1), 1), (W(1,1), -1)]
        ))
        
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        # C[0,1] += 1, C[1,1] += 1
        terms.append((
            [(U(0,0), 1)],
            [(V(0,1), 1), (V(1,1), -1)],
            [(W(1,0), 1), (W(1,1), 1)]
        ))
        
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        # C[0,0] += 1, C[1,0] += 1
        terms.append((
            [(U(1,1), 1)],
            [(V(1,0), 1), (V(0,0), -1)],
            [(W(0,0), 1), (W(0,1), 1)]
        ))
        
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        # C[0,0] += -1, C[0,1] += 1
        terms.append((
            [(U(0,0), 1), (U(0,1), 1)],
            [(V(1,1), 1)],
            [(W(0,0), -1), (W(1,0), 1)]
        ))
        
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        # C[1,1] += 1
        terms.append((
            [(U(1,0), 1), (U(0,0), -1)],
            [(V(0,0), 1), (V(0,1), 1)],
            [(W(1,1), 1)]
        ))
        
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        # C[0,0] += 1
        terms.append((
            [(U(0,1), 1), (U(1,1), -1)],
            [(V(1,0), 1), (V(1,1), 1)],
            [(W(0,0), 1)]
        ))
    
    def add_schoolbook(a_rows, c_cols, d_cols):
        """Add schoolbook terms for a block."""
        for la in range(len(a_rows)):
            for lc in range(len(c_cols)):
                for ld in range(len(d_cols)):
                    terms.append((
                        [(a_rows[la]*m + c_cols[lc], 1)],
                        [(c_cols[lc]*p + d_cols[ld], 1)],
                        [(d_cols[ld]*n + a_rows[la], 1)]
                    ))
    
    # Now we need to tile the 3x5x6 problem with these blocks
    # n=3: [2, 1], m=5: [2, 2, 1], p=6: [2, 2, 2]
    
    # For each combination of n-block, m-block, p-block:
    for ni in range(len(n_parts)):
        for mi in range(len(m_parts)):
            for pi in range(len(p_parts)):
                a_rows = list(range(n_offsets[ni], n_offsets[ni] + n_parts[ni]))
                c_cols = list(range(m_offsets[mi], m_offsets[mi] + m_parts[mi]))
                d_cols = list(range(p_offsets[pi], p_offsets[pi] + p_parts[pi]))
                
                if n_parts[ni] == 2 and m_parts[mi] == 2 and p_parts[pi] == 2:
                    add_strassen(a_rows, c_cols, d_cols)
                else:
                    add_schoolbook(a_rows, c_cols, d_cols)
    
    # Count total rank
    total_rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for r, (u_entries, v_entries, w_entries) in enumerate(terms):
        for (row, coeff) in u_entries:
            U[row][r] = coeff
        for (row, coeff) in v_entries:
            V[row][r] = coeff
        for (row, coeff) in w_entries:
            W[row][r] = coeff
    
    # Verify the decomposition
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook if Strassen fails
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W = U, V, W
    best_rank = total_rank
    
    # Now try to reduce rank further using algebraic search
    import numpy as np
    from scipy.linalg import svd
    
    def try_rank_reduction(U_init, V_init, W_init, n, m, p, max_time=120):
        """
        Try to find a lower-rank exact decomposition.
        
        Strategy:
        1. Start with the given decomposition (rank R)
        2. For candidate ranks r < R, try to find U', V', W' of rank r
           that exactly reproduce the tensor
        3. Use the fact that the tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        
        We use a linear algebra approach:
        - Fix W (or U or V) and solve for the others
        - Use random initializations and local optimization
        """
        R = len(U_init[0])
        best_U, best_V, best_W = U_init, V_init, W_init
        best_rank = R
        
        nm, mp, pn = n*m, m*p, p*n
        
        # Convert to numpy
        U_np = np.array(U_init, dtype=float)
        V_np = np.array(V_init, dtype=float)
        W_np = np.array(W_init, dtype=float)
        
        # The tensor T is defined by T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want to find a decomposition of rank r < R
        
        # Approach: For a given rank r, we can try to find U', V', W' such that
        # the tensor matches. This is a system of n*m*p equations in r*(nm+mp+pn) unknowns.
        # For r < R, this is generally overdetermined, but we can try random projections.
        
        # Key insight: if we fix one of the three matrices, the remaining two are
        # constrained by a linear system. Let's try fixing W and solving for U and V.
        
        # Actually, a better approach: use the structure of the problem.
        # The matrix multiplication tensor has special structure.
        
        # Let's try a different approach: look for linear dependencies among the
        # rank-1 terms. If term r1 = sum_s c_s * term_s for s != r1, we can eliminate r1.
        
        # Build the tensor in a flat representation
        # T_flat[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # For each pair of rows in U, check if one is a linear combination of others
        # in a way that's consistent with V and W
        
        # Simpler approach: try to find if any column of U is a linear combination
        # of other columns, and if so, whether the corresponding adjustment to V and W
        # preserves the tensor.
        
        # For column r of U, if U[:, r] = sum_s alpha_s * U[:, s], then we need
        # V[:, r] * W[:, r] to be adjusted... this doesn't work directly.
        
        # Better: use the fact that the tensor is bilinear in each factor.
        # If we have a rank-R decomposition, we can try to find a rank-r decomposition
        # by solving a system of equations.
        
        # Let's try: for target rank r, set up the system and solve it.
        # T[i,j,k] = sum_{s=1}^{r} U'[i,j,s] * V'[j,k,s] * W'[k,i,s]
        
        # This is nonlinear. But we can fix W' and solve for U' and V' linearly.
        
        # Fix W' = W_np[:, :r] (first r columns of original W)
        # Then for each (i,j,k): T[i,j,k] = sum_s U'[i,j,s] * V'[j,k,s] * W'[k,i,s]
        
        # This is still nonlinear in U' and V'. But if we fix U', it's linear in V'.
        
        # Alternative: use iterative approach. Start with rank-r truncation and refine.
        
        # Let's try a more practical approach: 
        # 1. Compute the tensor T as a 3D array
        # 2. Try various rank-r approximations using truncated SVD on matricizations
        # 3. Check if any give an exact match
        
        # Build the full tensor
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1  # This is the matrix multiplication tensor!
        
        # Wait, the matrix multiplication tensor T[i,j,k] = delta_{i,j,k} in the sense
        # that (A*B)[i,k] = sum_j A[i,j]*B[j,k], so the tensor is T[i,j,k] = 1 for all i,j,k
        # in the standard basis. No wait, that's not right.
        
        # The tensor for matrix multiplication is:
        # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
        # So T[i,j,k] = 1 for all valid i,j,k (it's the identity tensor in some sense)
        # No, T[i,j,k] is the coefficient of A[i,j]*B[j,k] in (A*B)[i,k], which is 1.
        
        # So the tensor is all-ones: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        # This means we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        # This is a constraint on U, V, W.
        
        # For a rank-r decomposition, we need to find U (nm x r), V (mp x r), W (pn x r)
        # such that for all i,j,k: sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1
        
        # This is a system of n*m*p = 90 equations in r*(nm+mp+pn) = r*(15+30+18) = 63r unknowns.
        # For r=84: 63*84 = 5292 unknowns, 90 equations. Very underdetermined.
        # For r=15: 63*15 = 945 unknowns, 90 equations. Still underdetermined.
        # For r=2: 63*2 = 126 unknowns, 90 equations. Underdetermined.
        
        # So the constraint is not that the tensor has low rank in the usual sense.
        # The tensor T[i,j,k] = 1 is a specific tensor, and we're looking for its
        # minimal rank decomposition.
        
        # The rank of the all-ones tensor of size n x m x p is known to be related
        # to the tensor rank of matrix multiplication, which is what we're computing.
        
        # Let me reconsider. The matrix multiplication tensor M_{n,m,p} has entries
        # M_{i,j,k} = 1 if the indices are valid, and the multiplication is:
        # C = A * B where C[i,k] = sum_j A[i,j] * B[j,k]
        
        # So the tensor T satisfies: C[i,k] = sum_{j} sum_{r} A[i,j] * B[j,k] * T[i,j,k]
        # Wait, no. The standard form is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So T[i,j,k] = 1 for all i,j,k.
        
        # The rank of this tensor is the minimum number of rank-1 tensors needed to
        # represent it. A rank-1 tensor is u_i * v_j * w_k.
        
        # So we need: 1 = sum_{s=1}^{r} u_s[i,j] * v_s[j,k] * w_s[k,i] for all i,j,k
        # where u_s[i,j] = U[i*m+j, s], v_s[j,k] = V[j*p+k, s], w_s[k,i] = W[k*n+i, s]
        
        # This is exactly what we have. The question is: what is the minimum r?
        
        # Known results: for 2x2x2, the rank is 7 (Strassen). For 3x3x3, it's at most 23.
        # For 3x5x6, we found 84 using block Strassen.
        
        # To improve, we need to find a better decomposition. This is a hard open problem
        # in general, but for small sizes we can try exhaustive or semi-exhaustive search.
        
        # Practical approach: try to find a rank-r decomposition by solving the system
        # for various r, starting from r = best_rank - 1 and going down.
        
        # For a fixed r, the system is:
        # For all i,j,k: sum_{s=1}^{r} U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1
        
        # This is a system of 90 polynomial equations of degree 3 in 63r variables.
        # Very hard to solve directly.
        
        # Instead, let's use a heuristic: 
        # 1. Start with the rank-84 solution
        # 2. Try to merge pairs of terms
        # 3. Or try random rank-r initializations and use least squares
        
        # Let's try: for target rank r, randomly initialize U, V, W with small entries,
        # then iterate: fix two, solve for the third using least squares.
        
        def solve_for_U(V_fixed, W_fixed, r, n, m, p):
            """Given V and W of rank r, solve for U such that the tensor matches."""
            # For each (i,j): sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1 for all k
            # This is a linear system in U[i*m+j, :] for each (i,j)
            
            U_sol = np.zeros((n*m, r))
            for i in range(n):
                for j in range(m):
                    # Build the system: for each k, sum_s U[i*m+j, s] * (V[j*p+k, s] * W[k*n+i, s]) = 1
                    # Let A_k = V[j*p+k, :] * W[k*n+i, :] (element-wise product)
                    # Then we need: U[i*m+j, :] dot A_k = 1 for all k
                    # This is: sum_s U[i*m+j, s] * A_k[s] = 1 for k = 0, ..., p-1
                    # So we have a system: M @ U[i*m+j, :] = 1 where M[k, s] = A_k[s]
                    
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V_fixed[j*p+k, :] * W_fixed[k*n+i, :]
                    
                    # Solve M @ x = ones(p) for x = U[i*m+j, :]
                    # Least squares solution
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U_sol[i*m+j, :] = x
                    except:
                        pass
            return U_sol
        
        def solve_for_V(U_fixed, W_fixed, r, n, m, p):
            """Given U and W of rank r, solve for V."""
            V_sol = np.zeros((m*p, r))
            for j in range(m):
                for k in range(p):
                    # For each i: sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1
                    # M[i, s] = U[i*m+j, s] * W[k*n+i, s]
                    # M @ V[j*p+k, :] = ones(n)
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U_fixed[i*m+j, :] * W_fixed[k*n+i, :]
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V_sol[j*p+k, :] = x
                    except:
                        pass
            return V_sol
        
        def solve_for_W(U_fixed, V_fixed, r, n, m, p):
            """Given U and V of rank r, solve for W."""
            W_sol = np.zeros((p*n, r))
            for k in range(p):
                for i in range(n):
                    # For each j: sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1
                    # M[j, s] = U[i*m+j, s] * V[j*p+k, s]
                    # M @ W[k*n+i, :] = ones(m)
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U_fixed[i*m+j, :] * V_fixed[j*p+k, :]
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W_sol[k*n+i, :] = x
                    except:
                        pass
            return W_sol
        
        def check_exact(U_c, V_c, W_c, n, m, p, tol=1e-10):
            """Check if the decomposition is exact (within tolerance)."""
            r = len(U_c[0])
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for s_idx in range(r):
                            s += U_c[i*m+j][s_idx] * V_c[j*p+k][s_idx] * W_c[k*n+i][s_idx]
                        if abs(s - 1.0) > tol:
                            return False
            return True
        
        def check_exact_frac(U_c, V_c, W_c, n, m, p):
            """Check exactness using Fraction."""
            r = len(U_c[0])
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for s_idx in range(r):
                            s += Fraction(U_c[i*m+j][s_idx]) * Fraction(V_c[j*p+k][s_idx]) * Fraction(W_c[k*n+i][s_idx])
                        if s != 1:
                            return False
            return True
        
        # Try to reduce rank from best_rank down to some minimum
        # Use alternating least squares with random restarts
        
        min_rank_to_try = max(n*m, m*p, n*p)  # Lower bound: at least max dimension product
        # Actually the lower bound is more subtle. For matrix multiplication, the rank
        # is at least max(n*m, m*p, n*p) / something, but let's be conservative.
        min_rank_to_try = 20  # Try down to rank 20
        
        random.seed(42)
        
        for target_rank in range(best_rank - 1, min_rank_to_try - 1, -1):
            if time.time() > deadline - 10:
                break
            
            found = False
            num_restarts = 5
            
            for restart in range(num_restarts):
                if time.time() > deadline - 10:
                    break
                
                # Random initialization
                U_r = np.random.randn(n*m, target_rank) * 0.1
                V_r = np.random.randn(m*p, target_rank) * 0.1
                W_r = np.random.randn(p*n, target_rank) * 0.1
                
                # Alternating least squares
                for iteration in range(200):
                    if time.time() > deadline - 10:
                        break
                    
                    U_r = solve_for_U(V_r, W_r, target_rank, n, m, p)
                    V_r = solve_for_V(U_r, W_r, target_rank, n, m, p)
                    W_r = solve_for_W(U_r, V_r, target_rank, n, m, p)
                
                # Check if we found an exact solution
                if check_exact(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol=1e-9):
                    # Verify with Fraction
                    if check_exact_frac(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                        best_U = U_r.tolist()
                        best_V = V_r.tolist()
                        best_W = W_r.tolist()
                        best_rank = target_rank
                        found = True
                        break
            if found:
                # Continue trying to reduce further
                continue
            else:
                # If we couldn't find rank target_rank, try lower ranks
                # But break if we've spent too much time
                if time.time() > deadline - 15:
                    break
        
        return best_U, best_V, best_W, best_rank
    
    # Try to reduce rank
    try:
        best_U, best_V, best_W, best_rank = try_rank_reduction(
            best_U, best_V, best_W, n, m, p
        )
    except Exception as e:
        # If rank reduction fails, return the Strassen decomposition
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
