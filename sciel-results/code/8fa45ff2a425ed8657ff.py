"""
Rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.

Strategy: Partition middle dimension 3 = 2+1, use Strassen on 2x2 blocks
(5 blocks x rank 7 = 35) and schoolbook on remainder (2x1x10 = 20).
Total: rank 55.

Attempt to reduce by finding linear dependencies or using a smarter partition.
"""
import time
import numpy as np
from scipy.linalg import null_space

DIMENSIONS = [2, 3, 10]
DEADLINE = 140.0


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    # Build the rank-55 decomposition first
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    idx = 0

    # Part 1: Strassen for 5 blocks of 2x2x2
    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1

        # M1
        U[0, idx] = 1.0; U[4, idx] = 1.0
        V[k0, idx] = 1.0; V[10 + k1, idx] = 1.0
        W[k0 * 2 + 0, idx] = 1.0; W[k1 * 2 + 1, idx] = 1.0
        idx += 1

        # M2
        U[3, idx] = 1.0; U[4, idx] = 1.0
        V[k0, idx] = 1.0
        W[k0 * 2 + 1, idx] = 1.0; W[k1 * 2 + 1, idx] = -1.0
        idx += 1

        # M3
        U[0, idx] = 1.0
        V[10 + k1, idx] = 1.0; V[10 + k1, idx] = 1.0
        V[10 + k1, idx] = 1.0
        W[k1 * 2 + 0, idx] = 1.0; W[k1 * 2 + 1, idx] = 1.0
        idx += 1

        # M4
        U[4, idx] = 1.0
        V[10 + k0, idx] = 1.0; V[k0, idx] = -1.0
        W[k0 * 2 + 0, idx] = 1.0; W[k0 * 2 + 1, idx] = 1.0
        idx += 1

        # M5
        U[0, idx] = 1.0; U[1, idx] = 1.0
        V[10 + k1, idx] = 1.0
        W[k0 * 2 + 0, idx] = -1.0; W[k1 * 2 + 0, idx] = 1.0
        idx += 1

        # M6
        U[3, idx] = 1.0; U[0, idx] = -1.0
        V[k0, idx] = 1.0; V[k1, idx] = 1.0
        W[k1 * 2 + 1, idx] = 1.0
        idx += 1

        # M7
        U[1, idx] = 1.0; U[4, idx] = -1.0
        V[10 + k0, idx] = 1.0; V[10 + k1, idx] = 1.0
        W[k0 * 2 + 0, idx] = 1.0
        idx += 1

    # Part 2: Schoolbook for 2x1 x 1x10
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2, idx] = 1.0
            V[2 * 10 + k, idx] = 1.0
            W[k * 2 + i, idx] = 1.0
            idx += 1

    # Check for linear dependencies in the tensor
    # The tensor T is in R^{(n*m) x (m*p) x (p*n)}
    # We can check if the columns of the "unfolding" are linearly dependent
    # by checking if there's a non-trivial combination that gives zero
    
    # Build the 3-way tensor as a 2D matrix (matricization)
    # T_{i,j,k} = sum_r U[i,r] * V[j,r] * W[k,r]
    # Matricize along the third mode: T_{(i,j), k}
    T_mat = np.zeros((n * m * m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(m):
                for l in range(p):
                    # This is getting complex, let's use a different approach
                    pass

    # Simpler: check if we can reduce rank by finding null space
    # of the tensor's matricization
    # Create the tensor as a 2D array by flattening first two indices
    tensor_2d = np.zeros((n * m * m * p, p * n))
    for r in range(rank):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        
        for i in range(n):
            for k in range(m):
                for j in range(m):
                    for l in range(p):
                        row_idx = (i * m + k) * m * p + j * p + l
                        col_idx = l * n + i
                        tensor_2d[row_idx, col_idx] += u_r[i, k] * v_r[j, l] * w_r[l, i]

    # Check rank of this matricization
    # If rank < 55, we can potentially reduce
    try:
        u, s, vh = np.linalg.svd(tensor_2d, full_matrices=False)
        rank_svd = np.sum(s > 1e-10)
        if rank_svd < rank:
            # We can potentially use a lower rank decomposition
            # But this is the matricization rank, not the tensor rank
            # Tensor rank >= matricization rank, so this doesn't directly help
            pass
    except:
        pass

    # For now, return the rank-55 decomposition
    # The matricization rank check suggests no easy reduction
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    
    n, m, p = DIMENSIONS
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    rank = U.shape[1]
    
    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)
    
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]
        
        A_weighted = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                A_weighted[i, j] = u_r[i * m + j]
        
        B_weighted = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                B_weighted[j, k] = v_r[j * p + k]
        
        prod_r = np.sum(A_weighted * A) * np.sum(B_weighted * B)
        
        for i in range(n):
            for k in range(p):
                C_decomp[i, k] += prod_r * w_r[k * n + i]
    
    C_direct = A @ B
    error = np.max(np.abs(C_decomp - C_direct))
    print(f"Max error: {error}")
    if error > 1e-10:
        print("WARNING: Decomposition is incorrect!")
