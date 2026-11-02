import numpy as np
import time
from fractions import Fraction

def construct():
    start_t = time.time()
    deadline = start_t + 140.0

    # Dimensions
    n, m, p = 2, 4, 4
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # U index: i*m + j  (i in [0,2), j in [0,4))
    # V index: j*p + k  (j in [0,4), k in [0,4))
    # W index: k*n + i  (k in [0,4), i in [0,2))

    # Build the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We want to decompose T as sum of rank-1 tensors
    # Rank-1: u[i,j] * v[j,k] * w[k,i]
    # i.e., sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k] = 1

    # Strategy: Start with naive rank-32 decomposition, then try to reduce rank
    # by finding linear dependencies among the rank-1 terms.
    # A better strategy: Use a structured approach based on block decomposition.

    # Let's use the block structure:
    # A is 2x4, B is 4x4, C is 2x4
    # Split A into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3)
    # Split B into four 2x2 blocks: B_TL, B_TR, B_BL, B_BR
    # C_L = A_L * B_TL + A_R * B_BL
    # C_R = A_L * B_TR + A_R * B_BR

    # For each 2x2 * 2x2 -> 2x2 multiplication, we can use Strassen (rank 7)
    # Total rank: 4 * 7 = 28

    # Let me verify this gives rank 28 and then try to find if we can do better.

    # Strassen decomposition for 2x2 matrix multiplication
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)

    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # Each is list of (local_idx, coeff)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),       # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),               # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),               # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),               # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),               # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),               # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),               # M7
    ]

    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Global index mappings for A (2x4)
    # A_L: cols 0,1 -> indices 0,1,4,5
    # A_R: cols 2,3 -> indices 2,3,6,7
    u_map_L = [0, 1, 4, 5]
    u_map_R = [2, 3, 6, 7]

    # Global index mappings for B (4x4)
    # B_TL: rows 0,1, cols 0,1 -> indices 0,1,4,5
    # B_TR: rows 0,1, cols 2,3 -> indices 2,3,6,7
    # B_BL: rows 2,3, cols 0,1 -> indices 8,9,12,13
    # B_BR: rows 2,3, cols 2,3 -> indices 10,11,14,15
    v_map_TL = [0, 1, 4, 5]
    v_map_TR = [2, 3, 6, 7]
    v_map_BL = [8, 9, 12, 13]
    v_map_BR = [10, 11, 14, 15]

    # Global index mappings for W (4x2, using W[k*n+i] = W[k*2+i])
    # C_L: cols 0,1
    #   (r,c) -> global col = c, global row = r
    #   W idx = c*2 + r
    #   (0,0)->0, (0,1)->2, (1,0)->1, (1,1)->3
    w_map_L = [0, 2, 1, 3]

    # C_R: cols 2,3
    #   (r,c) -> global col = 2+c, global row = r
    #   W idx = (2+c)*2 + r
    #   (0,0)->4, (0,1)->6, (1,0)->5, (1,1)->7
    w_map_R = [4, 6, 5, 7]

    col = 0

    # 1. A_L * B_TL -> C_L
    for u_t, v_t, w_t in strassen_terms:
        for idx, val in u_t:
            U[u_map_L[idx], col] += val
        for idx, val in v_t:
            V[v_map_TL[idx], col] += val
        for idx, val in w_t:
            W[w_map_L[idx], col] += val
        col += 1

    # 2. A_R * B_BL -> C_L
    for u_t, v_t, w_t in strassen_terms:
        for idx, val in u_t:
            U[u_map_R[idx], col] += val
        for idx, val in v_t:
            V[v_map_BL[idx], col] += val
        for idx, val in w_t:
            W[w_map_L[idx], col] += val
        col += 1

    # 3. A_L * B_TR -> C_R
    for u_t, v_t, w_t in strassen_terms:
        for idx, val in u_t:
            U[u_map_L[idx], col] += val
        for idx, val in v_t:
            V[v_map_TR[idx], col] += val
        for idx, val in w_t:
            W[w_map_R[idx], col] += val
        col += 1

    # 4. A_R * B_BR -> C_R
    for u_t, v_t, w_t in strassen_terms:
        for idx, val in u_t:
            U[u_map_R[idx], col] += val
        for idx, val in v_t:
            V[v_map_BR[idx], col] += val
        for idx, val in w_t:
            W[w_map_R[idx], col] += val
        col += 1

    # Verify the rank-28 decomposition
    def verify_decomp(U_arr, V_arr, W_arr, rnk):
        for seed in [42, 123, 999]:
            np.random.seed(seed)
            A_test = np.random.rand(n, m)
            B_test = np.random.rand(m, p)
            C_expected = A_test @ B_test

            C_computed = np.zeros((n, p))
            for ii in range(n):
                for kk in range(p):
                    acc = 0.0
                    for cc in range(rnk):
                        for jj in range(m):
                            acc += U_arr[ii * m + jj, cc] * V_arr[jj * p + kk, cc] * W_arr[kk * n + ii, cc]
                    C_computed[ii, kk] = acc

            if not np.allclose(C_expected, C_computed, atol=1e-10):
                return False
        return True

    if verify_decomp(U, V, W, rank):
        # Try to reduce rank by finding linear dependencies
        # Build the tensor from the decomposition and check if rank can be reduced
        # For now, return the rank-28 solution if it verifies
        if time.time() > deadline - 5:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        # Attempt rank reduction via linear algebra
        # The tensor T has entries T[i,j,k] = 1
        # Our decomposition gives sum_r U[:,j,r] * V[j,:,r] * W[:,i,r]
        # If there's a linear dependency among the rank-1 terms, we can reduce rank.
        # This is complex; for now, we'll try a simple approach:
        # Check if any two columns of U are proportional (which would indicate redundancy)

        # A more systematic approach: try to find a smaller set of rank-1 tensors
        # that span the same tensor. This requires solving a nonlinear problem.
        # Given time constraints, we'll stick with rank 28 unless we find an obvious reduction.

        # Let's try a different structured approach that might give rank 27 or less.
        # Idea: The 2x4x4 tensor can be viewed as 2 copies of 2x4x4 with row indices.
        # For row 0: C[0,k] = sum_j A[0,j] B[j,k]
        # For row 1: C[1,k] = sum_j A[1,j] B[j,k]
        # These are two independent 1x4 x 4x4 multiplications.
        # A 1x4 * 4x4 multiplication has rank 4*4=16 naively, but can it be lower?
        # Actually, the rank of the 1x4x4 tensor (outer product of 1x4 and 4x4)
        # is just 4*4 = 16 since it's a simple bilinear form.
        # So two of them would be 32 naively.
        # With Strassen-style sharing, we got 28.
        # Can we do better?

        # Let's try to see if rank 27 is possible by checking if the 28 rank-1 terms
        # have a linear dependency.

        # Build matrices for the decomposition
        # The tensor T is flattened as T[i,j,k] for all i,j,k
        # Our decomposition: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = T[i,j,k]

        # For a given r, the rank-1 tensor is u_r ⊗ v_r ⊗ w_r (in some sense)
        # To check linear independence, we can vectorize and check rank.

        # Vectorize: for each r, create a vector of length n*m*p = 2*4*4 = 32
        # vec_r[i*m*p + j*p + k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

        vec_dim = n * m * p
        vecs = np.zeros((rank, vec_dim))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        vecs[r, idx] = U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]

        # Check if the vectors are linearly independent
        # If rank(vecs) < 28, then there's a dependency and we might be able to reduce.
        # But reducing the tensor rank is not simply about linear independence of these vectors
        # because the decomposition is constrained to be a sum of rank-1 tensors with
        # the specific U, V, W structure.

        # For simplicity, if the vectors are linearly dependent, it suggests potential for reduction.
        # However, finding the actual reduced decomposition is non-trivial.

        # Given the complexity and time constraints, we'll return the verified rank-28 solution.
        # The rank is 28, which is an improvement over the naive 32.

        # Convert to integers if possible (the entries are already integers)
        U_int = np.round(U).astype(int)
        V_int = np.round(V).astype(int)
        W_int = np.round(W).astype(int)

        # Verify with integer entries
        if verify_decomp(U_int.astype(float), V_int.astype(float), W_int.astype(float), rank):
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
        else:
            # Return float version if integer conversion breaks verification
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to naive rank-32 decomposition
        rank_naive = n * m * p
        U_naive = np.zeros((n * m, rank_naive), dtype=int)
        V_naive = np.zeros((m * p, rank_naive), dtype=int)
        W_naive = np.zeros((p * n, rank_naive), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i * m + j, col] = 1
                    V_naive[j * p + k, col] = 1
                    W_naive[k * n + i, col] = 1
                    col += 1
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = 2, 4, 4
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)

    # Test with random matrices to ensure correctness
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B

    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc

    if not np.allclose(C, Cf):
        print("Verification Failed")
        print(C)
        print(Cf)
    else:
        print("Verification Passed")
    print("Rank:", U.shape[1])
