"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 115.0

    # Rank 50 decomposition:
    # 6 Strassen blocks of rank 7 (2x2x2) = 42
    # 8 rank-1 terms for 2x4x1 = 8
    # Total 50
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A11+A22)(B11+B22)
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1

        # M2: (A21+A22)B11
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1

        # M3: A11(B12-B22)
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1

        # M4: A22(B21-B11)
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1

        # M5: (A11+A12)B22
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1

        # M6: (A21-A11)(B11+B12)
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1

        # M7: (A12-A22)(B21+B22)
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    # Part 1: 2x4x4 - 4 Strassen blocks (j0=0,2 and k0=0,2)
    for k0 in (0, 2):
        for j0 in (0, 2):
            add_strassen(0, j0, k0)

    # Part 2: 2x4x2 - 2 Strassen blocks (k0=4, j0=0,2)
    for j0 in (0, 2):
        add_strassen(0, j0, 4)

    # Part 3: 2x4x1 - 8 rank-1 terms
    for i in range(n):
        for j in range(m):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1

    # Time guard
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Optional: verify and attempt rank reduction using nullspace of tensor equations
    try:
        t1 = time.time()
        if t1 > deadline:
            return {"U": U, "V": V, "W": W}

        # Build the tensor T of shape (n*m, m*p, p*n) representing matrix mult
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m + j, j*p + k, k*n + i] = 1

        # Current decomposition
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # Verify: T[i,j,k] should equal sum_r U[i,r] * V[j,r] * W[k,r]
        # This is a third-order tensor decomposition
        # Check residual
        residual = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            residual += np.einsum('i,j,k->ijk', U_np[:,r], V_np[:,r], W_np[:,r])
        err = np.max(np.abs(residual - T))
        if err > 1e-10:
            # Fallback: return schoolbook rank-56 decomposition
            sb_rank = n * m * p
            U_sb = [[0]*sb_rank for _ in range(n*m)]
            V_sb = [[0]*sb_rank for _ in range(m*p)]
            W_sb = [[0]*sb_rank for _ in range(p*n)]
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i*m+j][c] = 1
                        V_sb[j*p+k][c] = 1
                        W_sb[k*n+i][c] = 1
                        c += 1
            return {"U": U_sb, "V": V_sb, "W": W_sb}

        # Attempt to find rank reduction via linear algebra
        # Flatten: tensor T = sum_r u_r outer v_r outer w_r
        # If we have more than rank_r terms, some columns may be in the span of others
        # Build a system: for each (i,j,k), T[i,j,k] = sum_r U[i,r]V[j,r]W[k,r]
        # This is nonlinear in the factors. Instead, check if any column r can be
        # expressed as a linear combination using the bilinear structure.
        #
        # Alternative: check if the rank of the 3-mode matricizations exceeds a bound
        # Mode-1 matricization: U shape (n*m, rank), Mode-2: V (m*p, rank), Mode-3: W (p*n, rank)
        # The tensor rank <= min of numerical ranks but we need exact reduction.
        #
        # Try: fix V and W, solve for minimal U, then fix U and W, solve for minimal V, etc.
        # This is a greedy rank reduction.
        
        # Mode-1 matricization (row index i, col index (j,k)): T1[i, j*p + k] but actually
        # T1[i, (j,k)] = sum_r U[i,r] V[j,r] W[k,r]
        # Let M1[i, idx] where idx = j*(p*n) + k*(1) ... actually let's use:
        # M1[i, j*(p) + k] doesn't work directly because the third index is separate.
        # 
        # Use: for fixed i, the slice T[i,:,:] = sum_r U[i,r] * (V[:,r] * W[:,r]) elementwise
        # This is not a standard matricization.
        #
        # Skip complex reduction; return rank 50.
        if time.time() > deadline - 2:
            pass
    except Exception:
        pass

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
