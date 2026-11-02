"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def _build_rank14():
    """Return the verified rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # C indices (W): C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11, C[4]=C02, C[5]=C12, C[6]=C03, C[7]=C13
    
    # Left block (cols 0,1) - Strassen
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Right block (cols 2,3) - Strassen
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, tol=1e-10):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n, m, p = 2, 2, 4
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    r = U.shape[1]
    
    if U.shape != (n * m, r):
        return False
    if V.shape != (m * p, r):
        return False
    if W.shape != (p * n, r):
        return False
    
    max_err = 0.0
    for i in range(n):
        for j in range(p):
            for k in range(m):
                val = 0.0
                for rr in range(r):
                    val += U[i * m + k, rr] * V[k * p + j, rr] * W[j * n + i, rr]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    return max_err < tol


def _check_rank1_residual(residual, A_set, B_set, C_set):
    """
    Check if the residual tensor can be represented as a single rank-1 term
    with support restricted to A_set x B_set x C_set.
    
    The rank-1 term is: a[i] * b[j] * c[k] for i in A_set, j in B_set, k in C_set
    and 0 otherwise.
    
    We need to find a, b, c such that:
    residual[i,j,k] = a[i]*b[j]*c[k] for all (i,j,k) in A_set x B_set x C_set
    residual[i,j,k] = 0 for all (i,j,k) not in A_set x B_set x C_set
    """
    n, m, p = 2, 2, 4
    A_set = set(A_set)
    B_set = set(B_set)
    C_set = set(C_set)
    
    # First check: residual must be zero outside the support
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                if i not in A_set or j not in B_set or k not in C_set:
                    if abs(residual[i, j, k]) > 1e-12:
                        return None
    
    # Now check if residual is rank-1 on the support
    # Pick a reference point (i0, j0, k0) in the support
    i0 = next(iter(A_set))
    j0 = next(iter(B_set))
    k0 = next(iter(C_set))
    
    ref_val = residual[i0, j0, k0]
    if abs(ref_val) < 1e-12:
        # All values must be zero
        for i in A_set:
            for j in B_set:
                for k in C_set:
                    if abs(residual[i, j, k]) > 1e-12:
                        return None
        return None  # Degenerate case, skip
    
    # For a rank-1 tensor: residual[i,j,k] = a[i]*b[j]*c[k]
    # We can set a[i0]=1, b[j0]=1, then c[k0]=ref_val
    # Then: a[i] = residual[i,j0,k0] / (b[j0]*c[k0]) = residual[i,j0,k0] / ref_val
    #       b[j] = residual[i0,j,k0] / (a[i0]*c[k0]) = residual[i0,j,k0] / ref_val
    #       c[k] = residual[i0,j0,k] / (a[i0]*b[j0]) = residual[i0,j0,k] / ref_val
    
    a = {}
    b = {}
    c = {}
    
    # Compute a
    for i in A_set:
        a[i] = residual[i, j0, k0] / ref_val
    
    # Compute b
    for j in B_set:
        b[j] = residual[i0, j, k0] / ref_val
    
    # Compute c
    for k in C_set:
        c[k] = residual[i0, j0, k] / ref_val
    
    # Verify the rank-1 structure
    for i in A_set:
        for j in B_set:
            for k in C_set:
                expected = a[i] * b[j] * c[k]
                if abs(residual[i, j, k] - expected) > 1e-10:
                    return None
    
    return a, b, c


def _search_rank13(deadline):
    """
    Search for a rank-13 decomposition by trying to merge pairs of Strassen terms.
    """
    n, m, p = 2, 2, 4
    
    # Build the expected tensor
    T_expected = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                T_expected[i * m + k, k * p + j, j * n + i] = 1.0
    
    # Get the rank-14 solution
    rank14 = _build_rank14()
    U14 = np.array(rank14['U'], dtype=float)
    V14 = np.array(rank14['V'], dtype=float)
    W14 = np.array(rank14['W'], dtype=float)
    
    # The 14 Strassen patterns with their supports
    patterns = [
        ({0, 3}, {0, 5}, {0, 3}),      # M1
        ({2, 3}, {0}, {1, 3}),          # M2
        ({0}, {1, 5}, {2, 3}),          # M3
        ({3}, {4, 0}, {0, 1}),          # M4
        ({0, 1}, {5}, {0, 2}),          # M5
        ({2, 0}, {0, 1}, {3}),          # M6
        ({1, 3}, {4, 5}, {0}),          # M7
        ({0, 3}, {2, 7}, {4, 7}),       # N1
        ({2, 3}, {2}, {5, 7}),          # N2
        ({0}, {3, 7}, {6, 7}),          # N3
        ({3}, {6, 2}, {4, 5}),          # N4
        ({0, 1}, {7}, {4, 6}),          # N5
        ({2, 0}, {2, 3}, {7}),          # N6
        ({1, 3}, {6, 7}, {4}),          # N7
    ]
    
    # Try all pairs to merge
    for pair_idx in range(len(patterns)):
        if time.time() > deadline:
            break
        for other_idx in range(pair_idx + 1, len(patterns)):
            if time.time() > deadline:
                break
            
            p1 = patterns[pair_idx]
            p2 = patterns[other_idx]
            
            merged_A = p1[0] | p2[0]
            merged_B = p1[1] | p2[1]
            merged_C = p1[2] | p2[2]
            
            # Compute residual: T_expected - sum of remaining 12 terms
            residual = T_expected.copy()
            for t in range(14):
                if t == pair_idx or t == other_idx:
                    continue
                for i in range(n * m):
                    for j in range(m * p):
                        for k in range(p * n):
                            residual[i, j, k] -= U14[i, t] * V14[j, t] * W14[k, t]
            
            # Check if residual is rank-1 on the merged support
            result = _check_rank1_residual(residual, merged_A, merged_B, merged_C)
            if result is not None:
                a_coeff, b_coeff, c_coeff = result
                
                # Build the rank-13 solution
                U13 = [[0] * 13 for _ in range(n * m)]
                V13 = [[0] * 13 for _ in range(m * p)]
                W13 = [[0] * 13 for _ in range(p * n)]
                
                # Copy the 12 remaining terms
                new_idx = 0
                for t in range(14):
                    if t == pair_idx or t == other_idx:
                        continue
                    for i in range(n * m):
                        U13[i][new_idx] = U14[i, t]
                    for j in range(m * p):
                        V13[j][new_idx] = V14[j, t]
                    for k in range(p * n):
                        W13[k][new_idx] = W14[k, t]
                    new_idx += 1
                
                # Add the merged term
                for i in merged_A:
                    U13[i][12] = a_coeff[i]
                for j in merged_B:
                    V13[j][12] = b_coeff[j]
                for k in merged_C:
                    W13[k][12] = c_coeff[k]
                
                # Verify
                if _verify_decomposition(U13, V13, W13):
                    return {"U": U13, "V": V13, "W": W13}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a rank-13 decomposition
    rank13 = _search_rank13(_DEADLINE)
    if rank13 is not None:
        return rank13
    
    # Fallback to verified rank-14 Strassen-based decomposition
    return _build_rank14()

# EVOLVE-BLOCK-END
