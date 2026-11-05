"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]
import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 150s kill

    # Schoolbook decomposition as fallback
    best_rank = n * m * p  # 36

    # Initialize schoolbook
    U = [[0] * best_rank for _ in range(n * m)]
    V = [[0] * best_rank for _ in range(m * p)]
    W = [[0] * best_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    def verify_exact(U_mat, V_mat, W_mat, rank):
        """Exact verification using Fraction arithmetic.
        Condition: for all i in [0,n), k in [0,p), j,j2 in [0,m):
        sum_c W[k*n+i,c] * U[i*m+j,c] * V[j2*p+k,c] = 1 if j==j2 else 0
        """
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for j2 in range(m):
                        total = Fraction(0)
                        for c in range(rank):
                            wc = W_mat[k*n+i][c]
                            uc = U_mat[i*m+j][c]
                            vc = V_mat[j2*p+k][c]
                            if wc and uc and vc:
                                total += Fraction(wc) * Fraction(uc) * Fraction(vc)
                        expected = 1 if j == j2 else 0
                        if total != expected:
                            return False
        return True

    def verify_fast(U_mat, V_mat, W_mat, rank):
        """Fast numpy verification; if it passes, do exact check."""
        U_np = np.array(U_mat, dtype=np.float64)
        V_np = np.array(V_mat, dtype=np.float64)
        W_np = np.array(W_mat, dtype=np.float64)
        for i in range(n):
            for k in range(p):
                T = np.zeros((m, m))
                for c in range(rank):
                    wc = W_mat[k*n+i][c]
                    if wc:
                        u_col = [U_mat[i*m+j][c] for j in range(m)]
                        v_col = [V_mat[j2*p+k][c] for j2 in range(m)]
                        if any(u_col) and any(v_col):
                            T += wc * np.outer(u_col, v_col)
                if not np.allclose(T, np.eye(m), atol=1e-9):
                    return False
        return verify_exact(U_mat, V_mat, W_mat, rank)

    def try_eliminate(U_mat, V_mat, W_mat, rank, target):
        """Try to eliminate column `target` by adding a multiple of it to another column.
        
        Idea: pick a pivot column `pivot` (different from target) such that
        for some scalar alpha, after replacing column `pivot` with
        (U_p + alpha*U_t, V_p + alpha*V_t, W_p + alpha*W_t) and zeroing out
        column `target`, the tensor is unchanged.
        
        The condition: for every tensor entry (i,k,j,j2), the change must be zero.
        The change in entry (i,k,j,j2) when we do this:
          - We remove column target's contribution: -W_t[i,k]*U_t[i,j]*V_t[j2,k]
          - We change column pivot's contribution:
            (W_p + a*W_t)(U_p + a*U_t)(V_p + a*V_t) - W_p*U_p*V_p
            = W_p*U_p*a*V_t + W_p*a*U_t*V_p + a*W_t*U_p*V_p
              + a^2*(W_t*U_p*V_t + W_p*U_t*V_t + W_t*U_t*V_p)
              + a^3*W_t*U_t*V_t
        
        This is a cubic in a. In general hard. But there's a simpler approach:
        
        Border-basis / "merge" trick: if there exist two columns a and b such that
        for ALL entries, the product pattern satisfies:
          U_a[i,j]*V_b[j2,k]*W_b[i,k] == U_b[i,j]*V_a[j2,k]*W_a[i,k]
        for all (i,j,j2,k), then we can merge them (set one to zero after combining).
        
        More generally, the standard trick: if column b is "redundant" in the sense that
        there exists a linear combination of the other columns that reproduces its
        tensor contribution, we can eliminate it. This is a linear system in the
        coefficients.
        
        Let's use the linear approach: we want to find coefficients alpha_c for
        c != target such that for all (i,k,j,j2):
          sum_{c != target} alpha_c * U_c[i,j]*V_c[j2,k]*W_c[i,k]
          = W_t[i,k]*U_t[i,j]*V_t[j2,k]
        
        This is a system of n*p*m*m equations in (rank-1) unknowns.
        For rank=36, that's 35 unknowns and 3*4*3*3=108 equations. Overdetermined.
        But the tensor has structure, so there may be solutions.
        
        Actually, a cleaner approach: use the "rank-1 update" trick.
        If we can find a single column c != target and scalar alpha such that
          alpha * U_c[i,j]*V_c[j2,k]*W_c[i,k] == W_t[i,k]*U_t[i,j]*V_t[j2,k]
        for all (i,j,j2,k), then we can eliminate target by adding alpha*target to c.
        This requires the tensor contributions to be proportional, which is very restrictive.
        
        Better: use the standard matrix multiplication tensor structure.
        The schoolbook columns are indexed by (i,j,k). Column (i,j,k) has:
          U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1, rest 0.
        
        To eliminate column (i0,j0,k0), we need to find other columns and coefficients
        that reproduce its contribution. The contribution of column (i0,j0,k0) to
        tensor entry (i,k,j,j2) is:
          1 if (i==i0 and k==k0 and j==j0 and j2==j0) else 0
        
        So we need: for the specific entry (i0,k0,j0,j0), the sum of contributions
        from other columns must equal 1, and for all other entries, the sum must be 0.
        
        This is exactly the condition that the column is in the span of the others
        in the tensor sense. Let's set up the linear system.
        """
        r = rank
        # Unknowns: alpha_0, ..., alpha_{r-1} for c != target
        # Equations: for each (i,k,j,j2), sum_{c != target} alpha_c * U_c[i,j]*V_c[j2,k]*W_c[i,k] = RHS
        # where RHS = W_t[i,k]*U_t[i,j]*V_t[j2,k]
        
        # Build the system using numpy for speed, then verify with exact arithmetic
        # Number of equations: n*p*m*m
        n_eq = n * p * m * m
        n_unknowns = r - 1
        
        # Build A (n_eq x n_unknowns) and b (n_eq)
        A = np.zeros((n_eq, n_unknowns), dtype=np.float64)
        b = np.zeros(n_eq, dtype=np.float64)
        
        # Map from column index to unknown index
        col_to_unknown = {}
        uidx = 0
        for c in range(r):
            if c != target:
                col_to_unknown[c] = uidx
                uidx += 1
        
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for j2 in range(m):
                        eq_idx = ((i * p + k) * m + j) * m + j2
                        # RHS
                        b[eq_idx] = W_mat[k*n+i][target] * U_mat[i*m+j][target] * V_mat[j2*p+k][target]
                        # LHS
                        for c in range(r):
                            if c == target:
                                continue
                            val = U_mat[i*m+j][c] * V_mat[j2*p+k][c] * W_mat[k*n+i][c]
                            if val != 0:
                                A[eq_idx, col_to_unknown[c]] = val
        
        # Solve least squares
        try:
            alpha, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            return None
        
        # Check residual
        residual = np.linalg.norm(A @ alpha - b)
        if residual > 1e-8:
            return None
        
        # Verify with exact arithmetic
        # Build new matrices
        new_U = [row[:] for row in U_mat]
        new_V = [row[:] for row in V_mat]
        new_W = [row[:] for row in W_mat]
        
        # For each c != target, add alpha_c * column target to column c
        for c in range(r):
            if c == target:
                continue
            ac = alpha[col_to_unknown[c]]
            if abs(ac) < 1e-14:
                continue
            for i in range(n):
                for j in range(m):
                    new_U[i*m+j][c] += ac * U_mat[i*m+j][target]
            for j2 in range(m):
                for k in range(p):
                    new_V[j2*p+k][c] += ac * V_mat[j2*p+k][target]
            for i in range(n):
                for k in range(p):
                    new_W[k*n+i][c] += ac * W_mat[k*n+i][target]
        
        # Zero out column target
        for i in range(n*m):
            new_U[i][target] = 0
        for i in range(m*p):
            new_V[i][target] = 0
        for i in range(p*n):
            new_W[i][target] = 0
        
        # Exact verification
        if not verify_exact(new_U, new_V, new_W, r):
            return None
        
        return new_U, new_V, new_W, r - 1

    # Local search
    current_U = [row[:] for row in U]
    current_V = [row[:] for row in V]
    current_W = [row[:] for row in W]
    current_rank = best_rank
    
    best_result = {"U": current_U, "V": current_V, "W": current_W}
    best_score = current_rank
    
    # Try to eliminate columns
    max_iterations = 500
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        if current_rank <= 1:
            break
        
        # Try each column as a target
        found = False
        # Shuffle column order for diversity
        cols = list(range(current_rank))
        random.shuffle(cols)
        
        for target in cols:
            if time.time() > deadline:
                break
            
            result = try_eliminate(current_U, current_V, current_W, current_rank, target)
            if result is not None:
                current_U, current_V, current_W, current_rank = result
                found = True
                if current_rank < best_score:
                    best_score = current_rank
                    best_result = {
                        "U": [row[:] for row in current_U],
                        "V": [row[:] for row in current_V],
                        "W": [row[:] for row in current_W]
                    }
                break  # Restart with new configuration
        
        if not found:
            # No more eliminations possible; try a different strategy
            # Try random perturbation: add a small random vector to a random column
            # and see if we can then eliminate another column
            if current_rank > 2:
                # Pick a random column to perturb
                pert_col = random.randint(0, current_rank - 1)
                # Add a small random perturbation to this column
                for i in range(n*m):
                    current_U[i][pert_col] += random.choice([-1, 0, 1])
                for i in range(m*p):
                    current_V[i][pert_col] += random.choice([-1, 0, 1])
                for i in range(p*n):
                    current_W[i][pert_col] += random.choice([-1, 0, 1])
                
                # Verify it's still valid
                if not verify_fast(current_U, current_V, current_W, current_rank):
                    # Revert
                    for i in range(n*m):
                        current_U[i][pert_col] -= random.choice([-1, 0, 1])
                    # This won't work cleanly; just break
                    break
            else:
                break
    
    # Ensure we return a valid result
    if verify_fast(best_result["U"], best_result["V"], best_result["W"], best_score):
        return best_result
    
    # Fallback: return schoolbook
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
