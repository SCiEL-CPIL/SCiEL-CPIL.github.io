"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s wall-clock

    # Try to find rank-83 via rational elimination on the 84-term Strassen partition
    result83 = _try_rank83_rational_elimination(n, m, p)
    if result83 is not None:
        U, V, W = result83
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build Strassen-based decomposition (rank 84)
    result = _build_strassen_3x5x6(n, m, p)
    
    if result is not None:
        U, V, W, rank = result
        # Verify
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}


def _try_rank83_rational_elimination(n, m, p):
    """
    Attempt to find a rank-83 decomposition by:
    1. Building the rank-84 Strassen partition
    2. Finding a linear dependency among the 84 terms
    3. Eliminating one term to get rank 83
    """
    from fractions import Fraction
    
    start_time = time.time()
    deadline = start_time + 125
    
    # Build the rank-84 Strassen partition
    result84 = _build_strassen_3x5x6(n, m, p)
    if result84 is None:
        return None
    U84, V84, W84, rank84 = result84
    
    if rank84 != 84:
        return None
    
    # The tensor equation is: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all (i,j,k)
    # We have 90 equations (3*5*6) and 84 unknowns (the 84 terms are fixed)
    # Actually, the 84 terms ARE the decomposition. We want to check if there's a linear dependency.
    
    # Strategy: We know the 84-term decomposition works. We want to see if we can express
    # one term as a linear combination of the others, then eliminate it.
    
    # For each term c, check if it can be written as a linear combination of the other 83 terms.
    # This is equivalent to checking if the 84 vectors in the tensor space are linearly dependent.
    
    # Each term c defines a rank-1 tensor: T_c[i,j,k] = U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We need to check if these 84 tensors are linearly dependent in R^{3*5*6} = R^90.
    
    # Build the 90x84 matrix where column c is the vectorized tensor T_c
    A = np.zeros((n*m*p, rank84))
    for c in range(rank84):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    A[i*m*p + j*p + k, c] = U84[i*m+j, c] * V84[j*p+k, c] * W84[k*n+i, c]
    
    # Check if columns are linearly dependent: find null space
    # Use SVD to find the null space
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    # Check if rank < 84
    tol = 1e-10
    rank_A = sum(1 for sv in s_vals if sv > tol)
    
    if rank_A >= rank84:
        # No linear dependency found in floating point
        # Try with rational arithmetic for a specific elimination strategy
        return _try_targeted_elimination(n, m, p, U84, V84, W84, rank84)
    
    # Found a dependency. The null space vector gives coefficients.
    # The last row of Vt_svd (or rows corresponding to small singular values) gives the dependency.
    # For rank 83, there should be exactly one null vector.
    if rank_A != rank84 - 1:
        # More than one dependency or something unexpected
        return _try_targeted_elimination(n, m, p, U84, V84, W84, rank84)
    
    # Get the null space vector
    null_vec = Vt_svd[-1, :]
    
    # Find a term with a large coefficient to eliminate
    abs_coeffs = np.abs(null_vec)
    max_idx = np.argmax(abs_coeffs)
    
    if abs_coeffs[max_idx] < tol:
        return None
    
    # The dependency is: sum_c null_vec[c] * T_c = 0
    # So T_{max_idx} = -sum_{c != max_idx} (null_vec[c] / null_vec[max_idx]) * T_c
    
    # Build new decomposition with 83 terms by eliminating term max_idx
    U_new = np.zeros((n*m, rank84 - 1))
    V_new = np.zeros((m*p, rank84 - 1))
    W_new = np.zeros((p*n, rank84 - 1))
    
    new_col = 0
    for c in range(rank84):
        if c == max_idx:
            continue
        # Scale the term so that the coefficient becomes 1 in the elimination
        scale = -null_vec[c] / null_vec[max_idx]
        U_new[:, new_col] = U84[:, c] * scale
        V_new[:, new_col] = V84[:, c] * scale
        W_new[:, new_col] = W84[:, c] * scale
        new_col += 1
    
    # Verify
    if not _verify_fast(U_new, V_new, W_new, n, m, p):
        return None
    
    # Try to clean up to rational numbers
    U_new = _rationalize(U_new)
    V_new = _rationalize(V_new)
    W_new = _rationalize(W_new)
    
    if not _verify_fast(U_new, V_new, W_new, n, m, p):
        return None
    
    return U_new, V_new, W_new


def _try_targeted_elimination(n, m, p, U84, V84, W84, rank84):
    """
    Targeted elimination: try to express one specific term as a linear combination
    of others by solving a linear system.
    """
    from fractions import Fraction
    
    start_time = time.time()
    deadline = start_time + 120
    
    # Count non-zeros in each term
    nonzeros = []
    for c in range(rank84):
        nz = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if U84[i*m+j, c] != 0 and V84[j*p+k, c] != 0 and W84[k*n+i, c] != 0:
                        nz += 1
        nonzeros.append(nz)
    
    # Try eliminating terms with the most non-zeros first (most "complex" terms)
    sorted_terms = sorted(range(rank84), key=lambda c: -nonzeros[c])
    
    for target_c in sorted_terms[:20]:  # Try top 20 candidates
        if time.time() > deadline:
            break
        
        B = np.zeros((n*m*p, rank84 - 1))
        b = np.zeros(n*m*p)
        
        new_c = 0
        for c in range(rank84):
            if c == target_c:
                continue
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        B[i*m*p + j*p + k, new_c] = U84[i*m+j, c] * V84[j*p+k, c] * W84[k*n+i, c]
            new_c += 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    b[i*m*p + j*p + k] = U84[i*m+j, target_c] * V84[j*p+k, target_c] * W84[k*n+i, target_c]
        
        # Solve least squares
        try:
            alpha, residuals, rank_B, s_vals_B = np.linalg.lstsq(B, b, rcond=1e-10)
        except np.linalg.LinAlgError:
            continue
        
        # Check if solution is exact
        residual = np.linalg.norm(B @ alpha - b)
        if residual > 1e-8:
            continue
        
        # Found a solution! Build the new decomposition
        U_new = np.zeros((n*m, rank84 - 1))
        V_new = np.zeros((m*p, rank84 - 1))
        W_new = np.zeros((p*n, rank84 - 1))
        
        new_col = 0
        for c in range(rank84):
            if c == target_c:
                continue
            U_new[:, new_col] = U84[:, c] * alpha[new_col]
            V_new[:, new_col] = V84[:, c] * alpha[new_col]
            W_new[:, new_col] = W84[:, c] * alpha[new_col]
            new_col += 1
        
        if _verify_fast(U_new, V_new, W_new, n, m, p):
            U_new = _rationalize(U_new)
            V_new = _rationalize(V_new)
            W_new = _rationalize(W_new)
            if _verify_fast(U_new, V_new, W_new, n, m, p):
                return U_new, V_new, W_new
    
    return None


def _rationalize(mat, tol=1e-10):
    """
    Try to convert floating point values to simple rationals (integers or half-integers).
    """
    from fractions import Fraction
    result = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if abs(val) < tol:
                result[i, j] = 0.0
                continue
            # Try small denominators
            found = False
            for denom in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30]:
                num = round(val * denom)
                if abs(val - num / denom) < tol:
                    result[i, j] = float(num) / denom
                    found = True
                    break
            if not found:
                # Keep the floating point value
                result[i, j] = val
    return result


def _get_strassen_2x2x2():
    """
    Return U, V, W factors for 2x2x2 matrix multiplication (Strassen, rank 7).
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # U: A factors
    # M0: A00, A11
    U[0, 0] = 1
    U[3, 0] = 1
    # M1: A10, A11
    U[2, 1] = 1
    U[3, 1] = 1
    # M2: A00
    U[0, 2] = 1
    # M3: A11
    U[3, 3] = 1
    # M4: A00, A01
    U[0, 4] = 1
    U[1, 4] = 1
    # M5: A10, -A00
    U[2, 5] = 1
    U[0, 5] = -1
    # M6: A01, -A11
    U[1, 6] = 1
    U[3, 6] = -1
    
    # V: B factors
    # M0: B00, B11
    V[0, 0] = 1
    V[3, 0] = 1
    # M1: B00
    V[0, 1] = 1
    # M2: B01, -B11
    V[1, 2] = 1
    V[3, 2] = -1
    # M3: B10, -B00
    V[2, 3] = 1
    V[0, 3] = -1
    # M4: B11
    V[3, 4] = 1
    # M5: B00, B01
    V[0, 5] = 1
    V[1, 5] = 1
    # M6: B10, B11
    V[2, 6] = 1
    V[3, 6] = 1
    
    # W: C factors (W[k*2+i, c])
    # C00 = M0 + M3 - M4 + M6
    W[0, 0] = 1
    W[0, 3] = 1
    W[0, 4] = -1
    W[0, 6] = 1
    
    # C01 = M2 + M4
    # Index 1 is k=0, i=1
    W[1, 2] = 1
    W[1, 4] = 1
    
    # C10 = M1 + M3
    # Index 2 is k=1, i=0
    W[2, 1] = 1
    W[2, 3] = 1
    
    # C11 = M0 - M1 + M2 + M5
    # Index 3 is k=1, i=1
    W[3, 0] = 1
    W[3, 1] = -1
    W[3, 2] = 1
    W[3, 5] = 1
    
    return U, V, W


def _build_strassen_3x5x6(n, m, p):
    """
    Build tensor decomposition for 3x5 @ 5x6 using Strassen block structure.
    """
    sU, sV, sW = _get_strassen_2x2x2()
    
    # Total rank: 3 k-blocks * (2 Strassen blocks + 1 schoolbook for j=4)
    # Per k-block: 18 (top) + 10 (bottom) = 28
    # 3 k-blocks: 84
    
    rank = 84
    U = np.zeros((n * m, rank))  # 15 x 84
    V = np.zeros((m * p, rank))  # 30 x 84
    W = np.zeros((p * n, rank))  # 18 x 84
    
    col = 0
    
    for k_block in range(3):
        k0 = k_block * 2  # start k index
        k1 = k0 + 1       # end k index
        
        # Top 2 rows (i=0,1), j in {0,1}: Strassen
        for c in range(7):
            U[0*5 + 0, col] = sU[0, c]
            U[0*5 + 1, col] = sU[1, c]
            U[1*5 + 0, col] = sU[2, c]
            U[1*5 + 1, col] = sU[3, c]
            V[0*6 + k0, col] = sV[0, c]
            V[0*6 + k1, col] = sV[1, c]
            V[1*6 + k0, col] = sV[2, c]
            V[1*6 + k1, col] = sV[3, c]
            W[k0*3 + 0, col] = sW[0, c]
            W[k0*3 + 1, col] = sW[1, c]
            W[k1*3 + 0, col] = sW[2, c]
            W[k1*3 + 1, col] = sW[3, c]
            col += 1
        
        # Top 2 rows (i=0,1), j in {2,3}: Strassen
        for c in range(7):
            U[0*5 + 2, col] = sU[0, c]
            U[0*5 + 3, col] = sU[1, c]
            U[1*5 + 2, col] = sU[2, c]
            U[1*5 + 3, col] = sU[3, c]
            V[2*6 + k0, col] = sV[0, c]
            V[2*6 + k1, col] = sV[1, c]
            V[3*6 + k0, col] = sV[2, c]
            V[3*6 + k1, col] = sV[3, c]
            W[k0*3 + 0, col] = sW[0, c]
            W[k0*3 + 1, col] = sW[1, c]
            W[k1*3 + 0, col] = sW[2, c]
            W[k1*3 + 1, col] = sW[3, c]
            col += 1
        
        # Top 2 rows (i=0,1), j=4: schoolbook 2x1x2
        for i in range(2):
            for k in range(2):
                U[i*5 + 4, col] = 1
                V[4*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + i, col] = 1
                col += 1
        
        # Bottom row (i=2), j in {0,1}: schoolbook 1x2x2
        for j in range(2):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1
        
        # Bottom row (i=2), j in {2,3}: schoolbook 1x2x2
        for j in range(2, 4):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1
        
        # Bottom row (i=2), j=4: schoolbook 1x1x2
        for k in range(2):
            U[2*5 + 4, col] = 1
            V[4*6 + k0 + k, col] = 1
            W[(k0 + k)*3 + 2, col] = 1
            col += 1
    
    if col != rank:
        # Truncate or pad
        if col > rank:
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
        else:
            rank = col
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
    
    return U, V, W, rank


def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

# EVOLVE-BLOCK-END
