"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Rank 50 decomposition:
    # 4 blocks of 2x2x2 (Strassen, rank 7 each) -> 28
    # 2 blocks of 2x2x2 (Strassen, rank 7 each) -> 14
    # 8 rank-1 terms for 2x4x1 -> 8
    # Total 50
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Helper to add a Strassen block
    # A block: rows i0, i0+1; cols j0, j0+1
    # B block: rows j0, j0+1; cols k0, k0+1
    # C block: rows i0, i0+1; cols k0, k0+1
    def add_strassen(i0, j0, k0):
        nonlocal col
        # Local A indices:
        # a00: i0, j0
        # a01: i0, j0+1
        # a10: i0+1, j0
        # a11: i0+1, j0+1

        # Local B indices:
        # b00: j0, k0
        # b01: j0, k0+1
        # b10: j0+1, k0
        # b11: j0+1, k0+1

        # Local C indices (W rows):
        # c00: k0, i0
        # c01: k0+1, i0
        # c10: k0, i0+1
        # c11: k0+1, i0+1

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

    # Part 1: 2x4x4 (k=0..3)
    # 4 blocks of 2x2x2
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)

    # Part 2: 2x4x2 (k=4..5)
    # 2 blocks of 2x2x2
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)

    # Part 3: 2x4x1 (k=6)
    # 8 rank-1 terms
    for i in range(2):
        for j in range(4):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1

    # Now try to reduce rank by finding linear dependencies among the 50 terms
    # Convert to numpy arrays for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Check if we can find a lower rank decomposition
    # The tensor T is defined by T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want to find if there's a linear combination of the rank-1 terms that sums to zero
    
    # Build the "flattened" tensor: each rank-1 term r contributes to T as
    # (U[:,r] outer V[:,r] outer W[:,r]) flattened
    # If there's a vector c such that sum_r c[r] * (U[:,r] outer V[:,r] outer W[:,r]) = 0,
    # then we can reduce the rank.
    
    # Actually, a simpler approach: try to find if any column of the "joint" matrix is linearly dependent
    # The joint matrix has rows indexed by (i,j,k) and columns by r, with entry U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    
    # Build this matrix
    joint = np.zeros((n*m*p, rank), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    joint[idx, r] = U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                idx += 1
    
    # Find the rank of the joint matrix
    # If rank(joint) < 50, we can potentially reduce the rank
    try:
        _, s, vh = np.linalg.svd(joint, full_matrices=False)
        # Count how many singular values are above tolerance
        tol = 1e-10
        num_nonzero = np.sum(s > tol)
        print(f"Joint matrix rank: {num_nonzero}, singular values: {s[:10]}")
        
        # If we can reduce, we need to find a new basis for the column space
        # and express the tensor in terms of fewer rank-1 terms
        # This is complex; for now, just report
    except Exception as e:
        print(f"Error in SVD: {e}")

    # Verify decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(lhs - 1.0) > 1e-9:
                    print(f"Verification failed at i={i},j={j},k={k}: lhs={lhs}")
                    break

    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        print(f"Time limit approaching: {elapsed}s")

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
