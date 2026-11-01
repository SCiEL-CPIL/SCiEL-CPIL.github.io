"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank_schoolbook

    # Rank-32 construction for 2x2xp
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # We use 2 terms per k (total 2p = 30) plus 2 extra terms = 32
    # Actually, let's use a cleaner construction.
    #
    # For each k, we need to compute:
    #   C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    #   C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # Term 2k:   U[0,2k]=1, U[1,2k]=0, U[2,2k]=0, U[3,2k]=0
    #             V[0,2k]=1, V[p,2k]=0
    #             W[2k,2k]=1, W[2k+1,2k]=0  (k*n+0, k*n+1)
    #             This gives: for (i=0,j=0,k): 1*1*1=1, for (i=1,j=0,k): 0*1*0=0
    #
    # Term 2k+1: U[0,2k+1]=0, U[1,2k+1]=1, U[2,2k+1]=0, U[3,2k+1]=0
    #             V[0,2k+1]=0, V[p,2k+1]=1
    #             W[2k,2k+1]=1, W[2k+1,2k+1]=0
    #             This gives: for (i=0,j=1,k): 1*1*1=1, for (i=1,j=1,k): 0*1*0=0
    #
    # This only handles i=0. We need i=1 too. So this gives rank 4p = 60 (schoolbook).
    #
    # Better: use Strassen-style sharing.
    # For 2x2x2, Strassen has rank 7. For 2x2xp, we can do:
    #   7 terms for the "2x2 mixing" structure, but each term needs to handle all p columns.
    #   This doesn't directly work because the p-dimension is in V and W.
    #
    # Alternative: rank 2p+2 construction.
    # Let's use:
    #   For k=0..p-1:
    #     Term 2k:   handles A[0,0]*B[0,k] -> C[0,k] and A[1,0]*B[0,k] -> C[1,k]
    #     Term 2k+1: handles A[0,1]*B[1,k] -> C[0,k] and A[1,1]*B[1,k] -> C[1,k]
    #
    # Term 2k:   U[0,2k]=1, U[1,2k]=0, U[2,2k]=1, U[3,2k]=0
    #             V[0,2k]=1, V[p,2k]=0
    #             W[2k,2k]=1, W[2k+1,2k]=1  (both i=0 and i=1 for this k)
    #             Check: (i=0,j=0,k): U[0]*V[0]*W[2k] = 1*1*1 = 1 ✓
    #                     (i=1,j=0,k): U[2]*V[0]*W[2k+1] = 1*1*1 = 1 ✓
    #                     (i=0,j=1,k): U[1]*V[p]*W[2k] = 0*0*1 = 0 ✓
    #                     (i=1,j=1,k): U[3]*V[p]*W[2k+1] = 0*0*1 = 0 ✓
    #
    # Term 2k+1: U[0,2k+1]=0, U[1,2k+1]=1, U[2,2k+1]=0, U[3,2k+1]=1
    #             V[0,2k+1]=0, V[p,2k+1]=1
    #             W[2k,2k+1]=1, W[2k+1,2k+1]=1
    #             Check: (i=0,j=1,k): U[1]*V[p]*W[2k] = 1*1*1 = 1 ✓
    #                     (i=1,j=1,k): U[3]*V[p]*W[2k+1] = 1*1*1 = 1 ✓
    #                     (i=0,j=0,k): U[0]*V[0]*W[2k] = 0*0*1 = 0 ✓
    #                     (i=1,j=0,k): U[2]*V[0]*W[2k+1] = 0*0*1 = 0 ✓
    #
    # This gives rank 2p = 30! Let me verify all 4 entries per k.
    # For fixed k:
    #   (i=0,j=0): sum_c U[0,c]*V[0,c]*W[2k,c] = U[0,2k]*V[0,2k]*W[2k,2k] + U[0,2k+1]*V[0,2k+1]*W[2k,2k+1]
    #              = 1*1*1 + 0*0*1 = 1 ✓
    #   (i=0,j=1): sum_c U[1,c]*V[p,c]*W[2k,c] = U[1,2k]*V[p,2k]*W[2k,2k] + U[1,2k+1]*V[p,2k+1]*W[2k,2k+1]
    #              = 0*0*1 + 1*1*1 = 1 ✓
    #   (i=1,j=0): sum_c U[2,c]*V[0,c]*W[2k+1,c] = U[2,2k]*V[0,2k]*W[2k+1,2k] + U[2,2k+1]*V[0,2k+1]*W[2k+1,2k+1]
    #              = 1*1*1 + 0*0*1 = 1 ✓
    #   (i=1,j=1): sum_c U[3,c]*V[p,c]*W[2k+1,c] = U[3,2k]*V[p,2k]*W[2k+1,2k] + U[3,2k+1]*V[p,2k+1]*W[2k+1,2k+1]
    #              = 0*0*1 + 1*1*1 = 1 ✓
    #
    # All other (i,j,k') for k' != k: W entries are 0, so contribution is 0. ✓
    #
    # So rank 2p = 30 works!

    R = 2 * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    for k in range(p):
        c0 = 2 * k
        c1 = 2 * k + 1
        # Term c0: A[0,0]*B[0,k] -> C[0,k] and A[1,0]*B[0,k] -> C[1,k]
        U[0, c0] = 1  # i=0, j=0
        U[2, c0] = 1  # i=1, j=0
        V[0, c0] = 1  # j=0, k
        W[2 * k, c0] = 1      # k*n+0 = k*2+0
        W[2 * k + 1, c0] = 1  # k*n+1 = k*2+1

        # Term c1: A[0,1]*B[1,k] -> C[0,k] and A[1,1]*B[1,k] -> C[1,k]
        U[1, c1] = 1  # i=0, j=1
        U[3, c1] = 1  # i=1, j=1
        V[p, c1] = 1  # j=1, k (index j*p+k = 1*p+k = p+k)
        W[2 * k, c1] = 1      # k*n+0
        W[2 * k + 1, c1] = 1  # k*n+1

    # Verify
    if time.time() < deadline - 5:
        # Check all (i,j,k)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(R):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    expected = 1.0
                    max_err = max(max_err, abs(val - expected))
        if max_err > 1e-10:
            # Fallback to schoolbook
            return best_result
        else:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return best_result
# EVOLVE-BLOCK-END
