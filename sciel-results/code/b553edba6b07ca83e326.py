"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin for 150s limit

    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()

    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    # Strassen's algorithm for 2x2x2 matrix multiplication
    # Rank 7 decomposition
    def strassen_2x2():
        """Return U, V, W for 2x2x2 matrix multiplication with rank 7."""
        n2, m2, p2 = 2, 2, 2
        U = np.zeros((n2 * m2, 7))
        V = np.zeros((m2 * p2, 7))
        W = np.zeros((p2 * n2, 7))

        # Strassen's 7 products:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)

        # U: input A matrix indices (i*m+j)
        # A11: i=0,j=0 -> idx 0
        # A12: i=0,j=1 -> idx 1
        # A21: i=1,j=0 -> idx 2
        # A22: i=1,j=1 -> idx 3

        # V: input B matrix indices (j*p+k)
        # B11: j=0,k=0 -> idx 0
        # B12: j=0,k=1 -> idx 1
        # B21: j=1,k=0 -> idx 2
        # B22: j=1,k=1 -> idx 3

        # W: output C matrix indices (k*n+i)
        # C11: k=0,i=0 -> idx 0
        # C12: k=0,i=1 -> idx 1
        # C21: k=1,i=0 -> idx 2
        # C22: k=1,i=1 -> idx 3

        # M1 = (A11 + A22)(B11 + B22)
        U[0, 0] = 1  # A11
        U[3, 0] = 1  # A22
        V[0, 0] = 1  # B11
        V[3, 0] = 1  # B22
        W[0, 0] = 1  # C11
        W[1, 0] = 1  # C12
        W[2, 0] = 1  # C21
        W[3, 0] = 1  # C22

        # M2 = (A21 + A22)B11
        U[2, 1] = 1  # A21
        U[3, 1] = 1  # A22
        V[0, 1] = 1  # B11
        W[2, 1] = 1  # C21
        W[3, 1] = 1  # C22

        # M3 = A11(B21 - B22)
        U[0, 2] = 1  # A11
        V[2, 2] = 1  # B21
        V[3, 2] = -1 # B22
        W[1, 2] = 1  # C12
        W[3, 2] = 1  # C22

        # M4 = A22(B12 - B11)
        U[3, 3] = 1  # A22
        V[1, 3] = 1  # B12
        V[0, 3] = -1 # B11
        W[0, 3] = 1  # C11
        W[2, 3] = 1  # C21

        # M5 = (A11 + A12)B22
        U[0, 4] = 1  # A11
        U[1, 4] = 1  # A12
        V[3, 4] = 1  # B22
        W[1, 4] = 1  # C12
        W[3, 4] = 1  # C22

        # M6 = (A21 - A11)(B11 + B12)
        U[2, 5] = 1  # A21
        U[0, 5] = -1 # A11
        V[0, 5] = 1  # B11
        V[1, 5] = 1  # B12
        W[0, 5] = -1 # C11
        W[2, 5] = 1  # C21

        # M7 = (A12 - A22)(B21 + B22)
        U[1, 6] = 1  # A12
        U[3, 6] = -1 # A22
        V[2, 6] = 1  # B21
        V[3, 6] = 1  # B22
        W[0, 6] = 1  # C11
        W[1, 6] = -1 # C12

        return U, V, W

    # Block decomposition approach:
    # Split 4x5x5 into blocks and use Strassen for 2x2 blocks
    # For the 4x5 matrix, we can think of it as two 2x5 blocks
    # For the 5x5 matrix, we can think of it as 2x5 and 3x5 blocks

    # Let's try a different approach: direct construction using known results
    # For 4x5x5, the known upper bound is around 90-95

    # Try to find a better decomposition by exploiting structure
    # One approach: use the fact that we can combine some terms

    def try_block_decomposition():
        """Try to find a lower-rank decomposition using block structure."""
        if time.time() > deadline - 30:
            return None

        # Use Strassen for 2x2x2 blocks
        # Split n=4 into two groups of 2: {0,1} and {2,3}
        # Split m=5 into groups: {0,1} and {2,3,4}
        # Split p=5 into groups: {0,1} and {2,3,4}

        # This gives us a 2x2 block structure for the "large" blocks
        # But the dimensions don't match perfectly for Strassen

        # Alternative: just use schoolbook for now but try to find
        # linear dependencies among the basis tensors

        # For a 4x5x5 matrix multiplication, the tensor is T in R^{20} x R^{25} x R^{20}
        # The schoolbook decomposition has 100 terms
        # We need to find if there are linear dependencies

        # This is a hard problem in general, but for small cases we can try
        # to find a lower-rank decomposition

        # One heuristic: try to find if any subset of the schoolbook terms
        # can be replaced by fewer terms

        # For now, return the schoolbook decomposition as the best we have
        return None

    # Try the block decomposition approach
    result = try_block_decomposition()
    if result is not None:
        U, V, W = result
        if verify(U, V, W):
            if U.shape[1] < best_rank:
                best_rank = U.shape[1]
                best_U = U
                best_V = V
                best_W = W

    # Try to find a better rank by random search
    def random_search(rank_target, iterations=100):
        """Try to find a rank-rank_target decomposition by random search."""
        if time.time() > deadline - 20:
            return None

        for _ in range(iterations):
            if time.time() > deadline - 15:
                return None

            # Random initialization
            U = np.random.randn(n * m, rank_target)
            V = np.random.randn(m * p, rank_target)
            W = np.random.randn(p * n, rank_target)

            # Simple gradient-free optimization: try to minimize the error
            # This is very crude but might find something
            # For now, just check if the random guess works (very unlikely)
            if verify(U, V, W):
                return U, V, W

        return None

    # Try random search for slightly lower ranks
    for r in range(best_rank - 1, max(80, best_rank - 5), -1):
        if time.time() > deadline - 10:
            break
        result = random_search(r, iterations=10)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                best_rank = r
                best_U = U
                best_V = V
                best_W = W
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
