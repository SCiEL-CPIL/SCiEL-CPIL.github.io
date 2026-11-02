import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    def strassen_2x2x2_cols(i0, j0, k0, n_dim, m_dim, p_dim):
        """Returns list of 7 (U_dict, V_dict, W_dict) for Strassen 2x2x2 block."""
        cols = []
        # M1 = (A11+A22)(B11+B22)
        cols.append((
            {i0*m_dim+j0: 1, (i0+1)*m_dim+(j0+1): 1},
            {j0*p_dim+k0: 1, (j0+1)*p_dim+(k0+1): 1},
            {k0*n_dim+i0: 1, (k0+1)*n_dim+(i0+1): 1}
        ))
        # M2 = (A21+A22)B11
        cols.append((
            {(i0+1)*m_dim+j0: 1, (i0+1)*m_dim+(j0+1): 1},
            {j0*p_dim+k0: 1},
            {k0*n_dim+(i0+1): 1, (k0+1)*n_dim+(i0+1): -1}
        ))
        # M3 = A11(B12-B22)
        cols.append((
            {i0*m_dim+j0: 1},
            {j0*p_dim+(k0+1): 1, (j0+1)*p_dim+(k0+1): -1},
            {(k0+1)*n_dim+i0: 1, (k0+1)*n_dim+(i0+1): 1}
        ))
        # M4 = A22(B21-B11)
        cols.append((
            {(i0+1)*m_dim+(j0+1): 1},
            {(j0+1)*p_dim+k0: 1, j0*p_dim+k0: -1},
            {k0*n_dim+i0: 1, k0*n_dim+(i0+1): 1}
        ))
        # M5 = (A11+A12)B22
        cols.append((
            {i0*m_dim+j0: 1, i0*m_dim+(j0+1): 1},
            {(j0+1)*p_dim+(k0+1): 1},
            {k0*n_dim+i0: -1, (k0+1)*n_dim+i0: 1}
        ))
        # M6 = (A21-A11)(B11+B12)
        cols.append((
            {(i0+1)*m_dim+j0: 1, i0*m_dim+j0: -1},
            {j0*p_dim+k0: 1, j0*p_dim+(k0+1): 1},
            {(k0+1)*n_dim+(i0+1): 1}
        ))
        # M7 = (A12-A22)(B21+B22)
        cols.append((
            {i0*m_dim+(j0+1): 1, (i0+1)*m_dim+(j0+1): -1},
            {(j0+1)*p_dim+k0: 1, (j0+1)*p_dim+(k0+1): 1},
            {k0*n_dim+i0: 1}
        ))
        return cols

    def direct_decompose(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols):
        for di in range(nc):
            for dj in range(mc):
                for dk in range(pc):
                    u_idx = (ii + di) * m_dim + (jj + dj)
                    v_idx = (jj + dj) * p_dim + (kk + dk)
                    w_idx = (kk + dk) * n_dim + (ii + di)
                    cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))

    def decompose_recursive(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols):
        if time.time() > deadline:
            return
        if nc == 1 and mc == 1 and pc == 1:
            u_idx = ii * m_dim + jj
            v_idx = jj * p_dim + kk
            w_idx = kk * n_dim + ii
            cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))
            return
        if nc == 1 or mc == 1 or pc == 1:
            direct_decompose(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
            return
        if nc >= 2 and mc >= 2 and pc >= 2:
            cols.extend(strassen_2x2x2_cols(ii, jj, kk, n_dim, m_dim, p_dim))
            if nc > 2:
                decompose_recursive(nc - 2, mc, pc, ii + 2, jj, kk, n_dim, m_dim, p_dim, cols)
            if mc > 2:
                decompose_recursive(2, mc - 2, pc, ii, jj + 2, kk, n_dim, m_dim, p_dim, cols)
            if pc > 2:
                decompose_recursive(2, 2, pc - 2, ii, jj, kk + 2, n_dim, m_dim, p_dim, cols)
            return
        # Fallback split
        if nc > 1:
            decompose_recursive(1, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
            decompose_recursive(nc - 1, mc, pc, ii + 1, jj, kk, n_dim, m_dim, p_dim, cols)
        elif mc > 1:
            decompose_recursive(nc, 1, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
            decompose_recursive(nc, mc - 1, pc, ii, jj + 1, kk, n_dim, m_dim, p_dim, cols)
        elif pc > 1:
            decompose_recursive(nc, mc, 1, ii, jj, kk, n_dim, m_dim, p_dim, cols)
            decompose_recursive(nc, mc, pc - 1, ii, jj, kk + 1, n_dim, m_dim, p_dim, cols)

    def cols_to_matrices(cols):
        """Convert list of (u_dict, v_dict, w_dict) to U, V, W matrices."""
        rank = len(cols)
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for r, (u_dict, v_dict, w_dict) in enumerate(cols):
            for idx, val in u_dict.items():
                U[idx, r] = val
            for idx, val in v_dict.items():
                V[idx, r] = val
            for idx, val in w_dict.items():
                W[idx, r] = val
        return U, V, W

    def verify(U, V, W, tol=1e-9):
        """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
        rank = U.shape[1]
        # Build tensor
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    T[i, j, k] = s
        return np.allclose(T, 1.0, atol=tol)

    def solve_for_W(U, V):
        """
        Given U and V, solve for W such that the tensor identity holds.
        For each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        This is a linear system in W.
        """
        rank = U.shape[1]
        # W has shape (p*n, rank). W[k*n+i, r] for k in 0..p-1, i in 0..n-1
        # Variables: W[k*n+i, r] for all k,i,r -> p*n*rank variables
        # Equations: n*m*p equations
        # Let's set up the linear system.
        
        # For equation (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This only involves W[k*n+i, r] for r=0..rank-1
        # So each equation (i,j,k) involves only the row W[k*n+i, :]
        # Specifically: sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
        
        # This means: for each (k, i), we have multiple equations (one for each j)
        # that all involve the same row of W: W[k*n+i, :]
        # For fixed (k,i): for each j in 0..m-1:
        #   sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
        # This is a system of m equations in rank unknowns (W[k*n+i, 0..rank-1])
        
        # Build the system for each (k, i):
        # A_{j, r} = U[i*m+j, r] * V[j*p+k, r]
        # b_j = 1
        # Solve A * w = b where w = W[k*n+i, :]
        
        W = np.zeros((p * n, rank), dtype=float)
        for k in range(p):
            for i in range(n):
                # Build A: m x rank
                A = np.zeros((m, rank))
                for j in range(m):
                    for r in range(rank):
                        A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                b = np.ones(m)
                # Solve A * w = b (least squares if overdetermined)
                try:
                    w, residuals, rank_s, sv = np.linalg.lstsq(A, b, rcond=None)
                    # Check if solution is valid
                    if residuals is not None and len(residuals) > 0:
                        if residuals[0] > 1e-6:
                            return None  # No valid solution
                    W[k*n+i, :] = w
                except np.linalg.LinAlgError:
                    return None
        return W

    def solve_for_W_exact(U_frac, V_frac):
        """
        Exact rational solve for W given U and V as Fraction matrices.
        U_frac: list of lists of Fraction, shape (n*m, rank)
        V_frac: list of lists of Fraction, shape (m*p, rank)
        Returns W as list of lists of Fraction, or None if no solution.
        """
        rank = len(U_frac[0])
        W = [[Fraction(0) for _ in range(rank)] for _ in range(p*n)]
        
        for k in range(p):
            for i in range(n):
                # Build system: for each j, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # This is m equations in rank unknowns
                # Use Cramer's rule or Gaussian elimination with fractions
                # For simplicity, use numpy with rationals converted to float for check,
                # but we need exact. Let's use a simple Gaussian elimination.
                
                # Build augmented matrix: m x (rank+1)
                # Actually, this is a system A * w = b where A is m x rank
                # If m <= rank, we can solve exactly.
                # If m > rank, we need least squares (which may not be exact).
                
                # For our case: m=4, rank >= 7 typically, so m < rank usually.
                # Let's solve using exact Gaussian elimination.
                
                # Build A (m x rank) and b (m)
                A = [[Fraction(0) for _ in range(rank)] for _ in range(m)]
                b = [Fraction(1) for _ in range(m)]
                for j in range(m):
                    for r in range(rank):
                        A[j][r] = U_frac[i*m+j][r] * V_frac[j*p+k][r]
                
                # Solve A * w = b using Gaussian elimination
                # w has rank components
                # We have m equations, rank unknowns
                # If m < rank, infinitely many solutions; pick the one with free vars = 0
                # If m == rank, unique solution if det != 0
                # If m > rank, overdetermined; may not have solution
                
                # For simplicity, if m <= rank, solve the first m equations exactly
                # and set remaining free variables to 0
                # Actually, let's just solve using numpy for the float version and verify
                # For exactness, we'll use a different approach:
                
                # Use the fact that we can pick any solution. Let's solve the
                # first min(m, rank) equations and set the rest to 0.
                
                num_eq = min(m, rank)
                if num_eq == 0:
                    continue
                    
                # Extract first num_eq equations
                A_sub = [row[:num_eq] for row in A[:num_eq]]
                b_sub = b[:num_eq]
                
                # Gaussian elimination on A_sub * w_sub = b_sub
                # w_sub is the first num_eq components of w
                # Remaining components (if rank > num_eq) set to 0
                
                # Perform Gaussian elimination
                augmented = [A_sub[i][:] + [b_sub[i]] for i in range(num_eq)]
                
                # Eliminate
                for col in range(num_eq):
                    # Find pivot
                    pivot = None
                    for row in range(col, num_eq):
                        if augmented[row][col] != 0:
                            pivot = row
                            break
                    if pivot is None:
                        # Singular; try next column or return None
                        return None
                    # Swap
                    augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
                    # Normalize
                    pivot_val = augmented[col][col]
                    augmented[col] = [x / pivot_val for x in augmented[col]]
                    # Eliminate other rows
                    for row in range(num_eq):
                        if row != col and augmented[row][col] != 0:
                            factor = augmented[row][col]
                            augmented[row] = [augmented[row][t] - factor * augmented[col][t] for t in range(num_eq + 1)]
                
                # Extract solution
                w_sub = [augmented[i][num_eq] for i in range(num_eq)]
                # Remaining components are 0
                w_full = w_sub + [Fraction(0) for _ in range(rank - num_eq)]
                W[k*n+i] = w_full
        
        return W

    def verify_exact(U_frac, V_frac, W_frac):
        """Verify exact tensor decomposition with Fraction arithmetic."""
        rank = len(U_frac[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r]
                    if s != Fraction(1):
                        return False
        return True

    def generate_strassen_decomposition():
        """Generate the Strassen-based decomposition."""
        cols = []
        decompose_recursive(n, m, p, 0, 0, 0, n, m, p, cols)
        return cols

    def try_symbolic_elimination(cols):
        """
        Try to reduce rank by fixing U and V structure and solving for W exactly.
        Strategy: Start with the Strassen decomposition, then try to remove columns
        by solving the resulting linear system.
        """
        if time.time() > deadline:
            return None
        
        U, V, W = cols_to_matrices(cols)
        rank = len(cols)
        
        # Convert to Fraction for exact arithmetic
        U_frac = [[Fraction(U[i, r]).limit_denominator(10**6) for r in range(rank)] for i in range(n*m)]
        V_frac = [[Fraction(V[i, r]).limit_denominator(10**6) for r in range(rank)] for i in range(m*p)]
        
        # Verify the original decomposition is correct
        if not verify_exact(U_frac, V_frac, [[Fraction(W[i, r]).limit_denominator(10**6) for r in range(rank)] for i in range(p*n)]):
            return None
        
        # Try to remove one column at a time
        # For each column to remove, check if the remaining system is solvable
        for col_to_remove in range(rank):
            if time.time() > deadline:
                break
            
            # Create U and V without this column
            U_new_frac = []
            for i in range(n*m):
                row = [U_frac[i][r] for r in range(rank) if r != col_to_remove]
                U_new_frac.append(row)
            
            V_new_frac = []
            for i in range(m*p):
                row = [V_frac[i][r] for r in range(rank) if r != col_to_remove]
                V_new_frac.append(row)
            
            new_rank = rank - 1
            # Solve for W
            W_new_frac = solve_for_W_exact(U_new_frac, V_new_frac)
            if W_new_frac is None:
                continue
            
            # Verify
            if verify_exact(U_new_frac, V_new_frac, W_new_frac):
                # Success! Build new cols
                new_cols = []
                for r in range(new_rank):
                    orig_r = r if r < col_to_remove else r + 1
                    u_dict = {i: U_frac[i][orig_r] for i in range(n*m) if U_frac[i][orig_r] != 0}
                    v_dict = {i: V_frac[i][orig_r] for i in range(m*p) if V_frac[i][orig_r] != 0}
                    w_dict = {i: W_new_frac[i][r] for i in range(p*n) if W_new_frac[i][r] != 0}
                    new_cols.append((u_dict, v_dict, w_dict))
                return new_cols
        
        return None

    def try_direct_symbolic():
        """
        Try a direct symbolic approach: build U and V with a structured sparse form,
        then solve for W.
        """
        if time.time() > deadline:
            return None
        
        # Start with the Strassen decomposition and try various column removals
        cols = generate_strassen_decomposition()
        result = try_symbolic_elimination(cols)
        if result is not None:
            return result
        return None

    # Generate the base Strassen decomposition
    cols = generate_strassen_decomposition()
    U, V, W = cols_to_matrices(cols)
    rank = len(cols)
    
    # Verify base decomposition
    if not verify(U, V, W):
        # Fallback to schoolbook
        cols = []
        direct_decompose(n, m, p, 0, 0, 0, n, m, p, cols)
        U, V, W = cols_to_matrices(cols)
        rank = len(cols)
    
    # Try symbolic elimination to reduce rank
    optimized_cols = try_direct_symbolic()
    if optimized_cols is not None and len(optimized_cols) < rank:
        cols = optimized_cols
        rank = len(cols)
        U, V, W = cols_to_matrices(cols)
        # Final verification
        if not verify(U, V, W):
            # Revert
            cols = generate_strassen_decomposition()
            U, V, W = cols_to_matrices(cols)
            rank = len(cols)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
