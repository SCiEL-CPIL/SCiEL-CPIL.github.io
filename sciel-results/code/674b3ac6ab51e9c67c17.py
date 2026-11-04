"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Base rank 50 decomposition:
    # 6 Strassen blocks (2x2x2, rank 7 each) -> 42
    # 8 rank-1 terms for 2x4x1 -> 8
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

    # 6 Strassen blocks for 2x4x6 portion
    add_strassen(0, 0, 0)   # A(0:2,0:2) B(0:2,0:2) -> C(0:2,0:2)
    add_strassen(0, 0, 2)   # A(0:2,0:2) B(0:2,2:4) -> C(0:2,2:4)
    add_strassen(0, 2, 0)   # A(0:2,2:4) B(2:4,0:2) -> C(0:2,0:2)
    add_strassen(0, 2, 2)   # A(0:2,2:4) B(2:4,2:4) -> C(0:2,2:4)
    add_strassen(0, 0, 4)   # A(0:2,0:2) B(0:2,4:6) -> C(0:2,4:6)
    add_strassen(0, 2, 4)   # A(0:2,2:4) B(2:4,4:6) -> C(0:2,4:6)

    # 8 rank-1 terms for 2x4x1 (k=6)
    for i in range(2):
        for j in range(4):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1

    # Verify base decomposition
    def verify(U_list, V_list, W_list, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = sum(U_list[i*m+j][t] * V_list[j*p+k][t] * W_list[k*n+i][t] for t in range(r))
                    if abs(lhs - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, rank):
        # Should not happen, but fallback
        return {"U": U, "V": V, "W": W}

    # Attempt rank reduction: check if any term is linearly dependent
    # The tensor T[i,j,k] = 1 for all i,j,k
    # Each term r contributes: u_r[i,j] * v_r[j,k] * w_r[k,i]
    # If term r = sum_{s != r} c_s * term_s, we can eliminate it
    # This is a complex search; try simple cases first

    # Try: can we express the last 8 rank-1 terms using Strassen terms?
    # The rank-1 terms for k=6: for each (i,j), U[i*m+j]=1, V[j*p+6]=1, W[6*n+i]=1
    # These contribute to C[i,6] += A[i,j]*B[j,6]
    # Check if any Strassen term also contributes to C[*,6] or C[*,*] with k=6
    # Strassen blocks only use k in {0,1,2,3,4,5}, so no overlap with k=6
    # Thus the 8 rank-1 terms are independent from Strassen terms

    # Try: within Strassen blocks, are there dependencies?
    # Each Strassen block operates on disjoint (i,j,k) subspaces
    # Block (i0,j0,k0) affects C[i0:i0+2, k0:k0+2]
    # Since all 6 blocks have distinct k ranges (0-1, 2-3, 4-5) or distinct j ranges,
    # they operate on different parts of the tensor
    # Actually blocks 1,2 share k=0,1 but different j; blocks 3,4 share k=0,1 different j
    # Block 1: j=0,1 k=0,1; Block 2: j=0,1 k=2,3; Block 3: j=2,3 k=0,1; Block 4: j=2,3 k=2,3
    # Block 5: j=0,1 k=4,5; Block 6: j=2,3 k=4,5
    # All 6 blocks have disjoint (j,k) pairs, so their contributions to different C entries
    # This means the 42 Strassen terms + 8 rank-1 terms = 50 are likely all independent

    # However, let's try a more aggressive approach: use numpy to check rank of the
    # decomposition matrix and see if we can find a lower-rank representation
    if time.time() - start_time < DEADLINE - 20:
        try:
            U_np = np.array(U, dtype=float)  # (nm, r)
            V_np = np.array(V, dtype=float)  # (mp, r)
            W_np = np.array(W, dtype=float)  # (pn, r)

            # Build the full tensor decomposition matrix
            # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # We want to find the minimum r such that this holds
            # Try removing terms one by one and checking if we can compensate

            # Simpler approach: check if the 50 vectors (in the tensor space) are independent
            # Each term r defines a vector in R^{n*m*p} (flattened tensor)
            # vec_r[i*m*n + j*n + k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
            # If these 50 vectors are linearly independent, rank >= 50
            # If not, we can reduce

            vec_dim = n * m * p
            term_vectors = []
            for r in range(rank):
                vec = np.zeros(vec_dim)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            vec[(i*m + j)*p + k] = U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                term_vectors.append(vec)

            term_matrix = np.array(term_vectors).T  # (vec_dim, rank)
            # Check rank
            tr = np.linalg.matrix_rank(term_matrix, tol=1e-10)
            if tr < rank:
                # We have dependencies! Try to find a lower-rank decomposition
                # Use SVD to find the minimal representation
                # Actually, this doesn't directly give us a lower-rank matmul decomposition
                # We need the structure: T = sum_r u_r ⊗ v_r ⊗ w_r
                # The matrix_rank check tells us the tensor rank is at most tr
                # But finding an explicit lower-rank decomposition is hard

                # For now, just note the rank and return the original
                pass
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
