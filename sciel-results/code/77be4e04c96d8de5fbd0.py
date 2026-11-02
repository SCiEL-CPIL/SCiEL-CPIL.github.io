"""Rewritten program for 2x3x10 matrix multiplication tensor rank minimization."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank = n * m * p  # 60

    # Build schoolbook decomposition
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    # Verify the decomposition is correct
    def verify(U_mat, V_mat, W_mat, rnk):
        # Check that sum_c U[i,j,c] * V[j,k,c] * W[k,n+i,c] == delta_{i,j} * delta_{j,k} * delta_{k,n+i}
        # Actually the tensor is T[i,j,k] = 1 if consistent with matrix multiplication
        # For 2x3x10: T[i,j,k] where i in 0..1, j in 0..2, k in 0..9
        # T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) of C
        # In the standard convention, T[i,j,k] = 1 for all valid i,j,k (since A[i,j]*B[j,k] appears in C[i,k])
        # Wait, the tensor entries are T[i,j,k] = 1 if i,j,k form a valid multiplication triple
        # Actually for matrix multiplication, the tensor has T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
        # No, that's not right either. Let me think again.
        # The matrix multiplication tensor T satisfies:
        # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
        # where T[i,j,k] = 1 for all valid indices.
        # So the tensor is just the all-ones tensor on [n]x[m]x[p].
        # Wait no. T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k].
        # So yes, T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
        
        # Check: sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] should equal 1 for all valid i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rnk):
                        s += U_mat[i*m+j][c] * V_mat[j*p+k][c] * W_mat[k*n+i][c]
                    if s != 1:
                        return False
        return True

    if not verify(U, V, W, rank):
        # Fallback: rebuild
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
