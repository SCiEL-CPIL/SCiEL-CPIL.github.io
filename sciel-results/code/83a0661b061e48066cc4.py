"""Matrix multiplication 3x4 by 4x4 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s safety margin for 150s limit

    n, m, p = 3, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 12, 16, 12

    def build_rank44():
        """Build rank-44 decomposition using Strassen for 2x2x2 blocks + schoolbook for 1x2x2."""
        rank = 44
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]

        col = 0

        def add_strassen(X_u, Y_v, Z_w, col):
            """Add 7 Strassen terms for X(2x2) * Y(2x2) = Z(2x2)."""
            # M1 = (X00+X11)(Y00+Y11)
            U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            # M2 = (X10+X11)Y00
            U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1
            W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
            col += 1
            # M3 = X00(Y01-Y11)
            U[X_u[0]][col] += 1
            V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
            W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            # M4 = X11(Y10-Y00)
            U[X_u[3]][col] += 1
            V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
            W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
            col += 1
            # M5 = (X00+X01)Y11
            U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
            V[Y_v[3]][col] += 1
            W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
            col += 1
            # M6 = (X10-X00)(Y00+Y01)
            U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
            V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
            W[Z_w[3]][col] += 1
            col += 1
            # M7 = (X01-X11)(Y10+Y11)
            U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
            V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1
            col += 1
            return col

        # Strassen1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        # U indices: A[i,j] -> i*m+j: (0,0)=0,(0,1)=1,(1,0)=4,(1,1)=5
        # V indices: B[j,k] -> j*p+k: (0,0)=0,(0,1)=1,(1,0)=4,(1,1)=5
        # W indices: C[k,i] -> k*n+i: (0,0)=0,(0,1)=3,(1,0)=1,(1,1)=4
        col = add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4], col)

        # Strassen2: A[0:2,2:4] * B[0:2,2:4] -> C[0:2,0:2]
        # U: (0,2)=2,(0,3)=3,(1,2)=6,(1,3)=7
        # V: (0,2)=2,(0,3)=3,(1,2)=6,(1,3)=7
        # W: same as above
        col = add_strassen([2, 3, 6, 7], [2, 3, 6, 7], [0, 3, 1, 4], col)

        # Strassen3: A[0:2,0:2] * B[2:4,0:2] -> C[0:2,2:4]
        # U: (0,0)=0,(0,1)=1,(1,0)=4,(1,1)=5
        # V: (2,0)=8,(2,1)=9,(3,0)=12,(3,1)=13
        # W: (2,0)=6,(2,1)=9,(3,0)=7,(3,1)=10
        col = add_strassen([0, 1, 4, 5], [8, 9, 12, 13], [6, 9, 7, 10], col)

        # Strassen4: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
        # U: (0,2)=2,(0,3)=3,(1,2)=6,(1,3)=7
        # V: (2,2)=10,(2,3)=11,(3,2)=14,(3,3)=15
        # W: same as above
        col = add_strassen([2, 3, 6, 7], [10, 11, 14, 15], [6, 9, 7, 10], col)

        # Bottom row: A[2,0:4] * B[0:4,0:2] -> C[0:2,2]
        for j in range(4):
            u_idx = 2 * 4 + j  # 8+j
            v_idx = j * 4 + 0
            w_idx = 0 * 3 + 2  # 2
            U[u_idx][col] += 1; V[v_idx][col] += 1; W[w_idx][col] += 1
            col += 1
            v_idx = j * 4 + 1
            w_idx = 1 * 3 + 2  # 5
            U[u_idx][col] += 1; V[v_idx][col] += 1; W[w_idx][col] += 1
            col += 1

        # Bottom row: A[2,0:4] * B[0:4,2:4] -> C[2:4,2]
        for j in range(4):
            u_idx = 2 * 4 + j  # 8+j
            v_idx = j * 4 + 2
            w_idx = 2 * 3 + 2  # 8
            U[u_idx][col] += 1; V[v_idx][col] += 1; W[w_idx][col] += 1
            col += 1
            v_idx = j * 4 + 3
            w_idx = 3 * 3 + 2  # 11
            U[u_idx][col] += 1; V[v_idx][col] += 1; W[w_idx][col] += 1
            col += 1

        assert col == rank, f"Expected rank {rank}, got {col}"
        return U, V, W

    def verify(U, V, W):
        """Check exact decomposition."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_np[i * m + j] * V_np[j * p + k] * W_np[k * n + i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Build the rank-44 decomposition
    U, V, W = build_rank44()

    # Verify it's correct
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank = 48
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_U, best_V, best_W = U, V, W
    best_rank = 44

    # Try to find linear dependencies to reduce rank
    # Form combined matrix M where each column is the stacked factor vector
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    current_rank = U_np.shape[1]

    # Check if any column is linearly dependent on others
    # We'll try removing columns one at a time and see if the tensor equation still holds
    # This is a heuristic: if column c is in the span of others in a way that preserves
    # the tensor structure, we can remove it.

    # More targeted approach: for each column, check if setting it to zero
    # still allows the remaining columns to satisfy the tensor equation.
    # This is expensive, so we do it in a limited way.

    # Alternative: check numerical rank of the combined factor matrix
    M = np.vstack([U_np, V_np, W_np])  # shape (40, 44)
    # If rank(M) < 44, there might be redundancy, but it doesn't mean we can simply drop a column
    # because the tensor constraint is multiplicative, not just linear.

    # Try a greedy removal: for each column, tentatively zero it out and check
    # if the remaining columns can still produce the correct tensor.
    # We do this by checking: does there exist coefficients for the remaining columns
    # that reproduce the tensor? This is a linear system in the coefficients.

    # For efficiency, we'll only try a limited number of removals.
    if time.time() < deadline - 30:
        # Try removing each column and solving for replacement
        # The tensor equation is: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
        # If we remove column r, we need: sum_{c != r} U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
        # This is NOT simply a linear system because of the product structure.
        # Instead, we check if the column is "redundant" in the sense that
        # the products U[i,j,r]*V[j,k,r]*W[k,i,r] can be replicated by others.

        # Simpler check: if the column has all zeros in one of U, V, or W, it's already zero.
        # Otherwise, try to see if we can express it as a linear combination.

        # Given the complexity, let's just try a few random perturbations
        # to see if we can find a lower rank solution.
        # For now, the rank-44 is our best bet.
        pass

    if time.time() > deadline:
        pass  # Return what we have

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
