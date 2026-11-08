import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    start_time = time.time()
    deadline = start_time + 120.0

    def time_left():
        return deadline - time.time()

    def matmul3(U, V, W, rank):
        """Compute tensor contraction: T[i,j,k] = sum_t U[i,t]*V[j,t]*W[k,t]"""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # We want C[i,k] = sum_j A[i,j]*B[j,k]
        # T[i*m+j, j*p+k, k*n+i] should be 1
        # Actually, the tensor is: T[a,b,c] where a indexes A (i*m+j), b indexes B (j*p+k), c indexes C (k*n+i)
        # T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
        # For correct matmul: T[i*m+j, j*p+k, k*n+i] = 1
        # Let's compute the full tensor and check
        result = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for t in range(rank):
            u_col = U[:, t]
            v_col = V[:, t]
            w_col = W[:, t]
            # Outer product
            result += np.einsum('a,b,c->abc', u_col, v_col, w_col)
        return result

    def target_tensor():
        """Build the target matrix multiplication tensor"""
        T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1
        return T

    # Start with the Strassen-based rank-50 construction
    def strassen_rank50():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            e_idx = j0 * p + k0
            f_idx = j0 * p + (k0 + 1)
            g_idx = (j0 + 1) * p + k0
            h_idx = (j0 + 1) * p + (k0 + 1)
            c00_idx = k0 * n + i0
            c01_idx = (k0 + 1) * n + i0
            c10_idx = k0 * n + (i0 + 1)
            c11_idx = (k0 + 1) * n + (i0 + 1)

            U[a_idx][col] += 1; U[d_idx][col] += 1
            V[e_idx][col] += 1; V[h_idx][col] += 1
            W[c00_idx][col] += 1; W[c11_idx][col] += 1
            col += 1

            U[c_idx][col] += 1; U[d_idx][col] += 1
            V[e_idx][col] += 1
            W[c10_idx][col] += 1; W[c11_idx][col] -= 1
            col += 1

            U[a_idx][col] += 1
            V[f_idx][col] += 1; V[h_idx][col] -= 1
            W[c01_idx][col] += 1; W[c11_idx][col] += 1
            col += 1

            U[d_idx][col] += 1
            V[g_idx][col] += 1; V[e_idx][col] -= 1
            W[c00_idx][col] += 1; W[c10_idx][col] += 1
            col += 1

            U[a_idx][col] += 1; U[b_idx][col] += 1
            V[h_idx][col] += 1
            W[c00_idx][col] -= 1; W[c01_idx][col] += 1
            col += 1

            U[c_idx][col] += 1; U[a_idx][col] -= 1
            V[e_idx][col] += 1; V[f_idx][col] += 1
            W[c11_idx][col] += 1
            col += 1

            U[b_idx][col] += 1; U[d_idx][col] -= 1
            V[g_idx][col] += 1; V[h_idx][col] += 1
            W[c00_idx][col] += 1
            col += 1

        def add_schoolbook(i0, j0, k0):
            nonlocal col
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            e_idx = j0 * p + k0
            g_idx = (j0 + 1) * p + k0
            c00_idx = k0 * n + i0
            c10_idx = k0 * n + (i0 + 1)

            U[a_idx][col] += 1; V[e_idx][col] += 1; W[c00_idx][col] += 1; col += 1
            U[b_idx][col] += 1; V[g_idx][col] += 1; W[c00_idx][col] += 1; col += 1
            U[c_idx][col] += 1; V[e_idx][col] += 1; W[c10_idx][col] += 1; col += 1
            U[d_idx][col] += 1; V[g_idx][col] += 1; W[c10_idx][col] += 1; col += 1

        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                add_strassen(0, j_start, k_start)
            add_schoolbook(0, j_start, 6)

        return U, V, W, rank

    U, V, W, best_rank = strassen_rank50()
    best_result = (U, V, W, best_rank)

    # Try to find a lower-rank decomposition via random search
    # Strategy: try ranks from 48 down to 44, use random initialization
    # and a simple local search

    def verify(U, V, W, rank):
        """Check if the decomposition is exact"""
        if time_left() < 2.0:
            return False
        T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for t in range(rank):
            u_col = np.array(U[:, t], dtype=np.int64)
            v_col = np.array(V[:, t], dtype=np.int64)
            w_col = np.array(W[:, t], dtype=np.int64)
            T += np.einsum('a,b,c->abc', u_col, v_col, w_col)
        target = target_tensor()
        return np.array_equal(T, target)

    def random_decomposition(rank, seed):
        """Generate a random decomposition with entries in {-2,-1,0,1,2}"""
        rng = np.random.RandomState(seed)
        U = rng.randint(-2, 3, size=(n*m, rank))
        V = rng.randint(-2, 3, size=(m*p, rank))
        W = rng.randint(-2, 3, size=(p*n, rank))
        return U.tolist(), V.tolist(), W.tolist()

    # Try a few random searches for lower ranks
    for rank in range(48, 43, -1):
        if time_left() < 10.0:
            break
        if rank >= best_rank:
            continue

        for seed in range(200):
            if time_left() < 5.0:
                break
            U_r, V_r, W_r = random_decomposition(rank, seed)
            if verify(U_r, V_r, W_r, rank):
                best_rank = rank
                best_result = (U_r, V_r, W_r, rank)
                break
        if rank < best_rank:
            break

    # Also try a structured approach: modify the Strassen blocks
    # Try replacing one schoolbook block (4 mults) with a 3-mult decomposition
    # if possible for 2x2x1 -> not possible in general, but let's try mixed

    # Try to reduce by combining Strassen and schoolbook differently
    # Current: 2 * (3*7 + 4) = 2 * 25 = 50
    # Alternative: use 7-term for 2x2x2, but for the odd column use a different approach

    # Try a block structure: treat the 4x7 as four 1x7 blocks
    # For each j (4 values), multiply 2x1 by 1x7: 14 mults total = 56
    # That's worse.

    # Try: split into two 2x4x7 blocks, each using a custom decomposition
    # For 2x4x7: we need to compute C[i,k] = sum_j A[i,j]*B[j,k] for i=0,1, k=0..6
    # That's 14 outputs, 8 inputs from A, 28 inputs from B
    # Schoolbook: 2*4*7 = 56 per block... no wait, we're splitting j, not i.

    # Let me try a different approach: use the fact that n=2
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]

    # Group by k: for each k, we have a 2x4 * 4x1 multiplication = 8 mults
    # 7 values of k = 56 total

    # Can we do better by sharing across k values?
    # Strassen on j dimension: split j into {0,1} and {2,3}
    # For j in {0,1}: compute M1=(A00+A11)(B0k+B1k), M2=(A10+A11)B0k, M3=A00(B1k-B0k), M4=A11(B0k+B1k)... 
    # Wait, that's not quite right. Let me think about this differently.

    # For a fixed k, we're doing 2x4 * 4x1.
    # Split the 4 j-values into two groups of 2.
    # For group {0,1}: 2x2 * 2x1 = 4 mults schoolbook, or... 
    # Actually for 2x2 * 2x1, we can't use Strassen (Strassen is for 2x2*2x2).
    # For 2x2 * 2x1: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    #                C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    # That's 4 mults, no savings possible for 2x2*2x1.

    # So the current approach of using Strassen on 2x2*2x2 blocks is the right idea.
    # The question is whether we can do better than 7 mults for the 2x2*2x2 part.

    # 7 is the known minimum for 2x2*2x2 (Strassen). So 3*7 + 4 = 25 per j-block is optimal
    # given this decomposition.

    # Can we do a different decomposition?
    # What if we don't split j into {0,1} and {2,3}, but use a different grouping?
    # The 4 j-values must be handled somehow.

    # Alternative: treat the whole 4x7 as one block and find a tensor decomposition.
    # This is a much harder problem.

    # Let's try to use the remaining time for a more sophisticated search.
    # We can try to find a rank-49 or rank-48 decomposition by starting from the
    # rank-50 one and trying to merge or eliminate terms.

    # For now, return the best we have.
    U_best, V_best, W_best, rank_best = best_result
    return {"U": U_best, "V": V_best, "W": W_best}
