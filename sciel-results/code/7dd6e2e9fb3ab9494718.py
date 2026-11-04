import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W, atol=1e-6):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def get_rank50():
        """Known valid rank-50 decomposition: 6 Strassen blocks + 8 rank-1 terms."""
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        for j in range(m):
            U[0 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 0, col] = 1
            col += 1
            U[1 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 1, col] = 1
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    # Get rank-50 baseline
    U50, V50, W50 = get_rank50()
    if not verify_decomp(U50, V50, W50):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 50
    best_U, best_V, best_W = U50, V50, W50

    # Precompute unfoldings
    T1 = T.reshape(n * m, -1)  # 8 x 392
    T2 = T.reshape(m * p, -1)  # 28 x 112
    T3 = T.reshape(p * n, -1)  # 14 x 224

    def als_fast(T1, T2, T3, U_init, V_init, W_init, max_iters=60, tol=1e-9):
        R = U_init.shape[1]
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            KR_VW = np.zeros((m * p * p * n, R))
            for r in range(R):
                KR_VW[:, r] = np.kron(V[:, r], W[:, r])
            try:
                U_new = np.linalg.lstsq(KR_VW, T1.T, rcond=None)[0].T
            except:
                break
            KR_WU = np.zeros((p * n * n * m, R))
            for r in range(R):
                KR_WU[:, r] = np.kron(W[:, r], U[:, r])
            try:
                V_new = np.linalg.lstsq(KR_WU, T2.T, rcond=None)[0].T
            except:
                break
            KR_UV = np.zeros((n * m * m * p, R))
            for r in range(R):
                KR_UV[:, r] = np.kron(U[:, r], V[:, r])
            try:
                W_new = np.linalg.lstsq(KR_UV, T3.T, rcond=None)[0].T
            except:
                break
            change = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
            U, V, W = U_new, V_new, W_new
            if change < tol:
                break
        return U, V, W

    def polish(U, V, W, max_iters=40):
        R = U.shape[1]
        for _ in range(max_iters):
            if time.time() > deadline - 6:
                break
            KR_VW = np.zeros((m * p * p * n, R))
            for r in range(R):
                KR_VW[:, r] = np.kron(V[:, r], W[:, r])
            U = np.linalg.lstsq(KR_VW, T1.T, rcond=None)[0].T
            KR_WU = np.zeros((p * n * n * m, R))
            for r in range(R):
                KR_WU[:, r] = np.kron(W[:, r], U[:, r])
            V = np.linalg.lstsq(KR_WU, T2.T, rcond=None)[0].T
            KR_UV = np.zeros((n * m * m * p, R))
            for r in range(R):
                KR_UV[:, r] = np.kron(U[:, r], V[:, r])
            W = np.linalg.lstsq(KR_UV, T3.T, rcond=None)[0].T
        return U, V, W

    # Strategy: decouple into two 1x4x7 problems (n=2).
    # For each row i, the tensor slice is 1x4x7 with rank at most 28.
    # Use Winograd-style overlapping rank-3 schemes for 2-column windows.
    # For 1x4 x 4x7: split columns into overlapping pairs.
    # c_k = sum_j a_j * b_jk
    # Use rank-3 for (a_j, a_{j+1}) x (b_jk, b_{j+1,k}) -> partial c_k
    # Overlapping windows: j=0-1, 1-2, 2-3 for k in 0..6
    # Each window covers 2 rows of B, 7 columns -> 2x7 output
    # Rank-3 per column pair, but we have 7 columns... 
    # 
    # Better: for 1x4x7, use the fact that rank <= 4*7 = 28, 
    # and try to find rank 21 per row via structured search.
    # 
    # Actually, let's try a direct approach: for each row i independently,
    # find a rank-r decomposition of the 1x4x7 slice, then combine.
    # The two rows share B, so we can share rank terms if the B-part matches.
    #
    # Key insight: the tensor T for row i has T[i*4+j, j*7+k, k*2+i] = 1.
    # For i=0: entries (j, 7j+k, 2k) for j=0..3, k=0..6
    # For i=1: entries (4+j, 7j+k, 2k+1) for j=0..3, k=0..6
    #
    # These are independent in the W-index (different parity of k*2+i).
    # So the two rows have DISJOINT W-support!
    # This means we can decompose each row independently and concatenate.
    # Rank(2x4x7) = Rank(1x4x7 for i=0) + Rank(1x4x7 for i=1)
    #
    # For 1x4x7: c_k = sum_{j=0}^{3} a_j * b_jk, k=0..6
    # This is a 4x7 matrix-vector product (transposed).
    # Schoolbook rank: 28.
    # Can we do better? The tensor is sum_{j,k} a_j b_jk c_k.
    # This is a 4x7 matrix M (b_jk) applied to vector a, producing c.
    # The rank of this "linear map" tensor is the rank of M as a matrix? No.
    # It's the CP rank of the 4x7x1 tensor, which equals the matrix rank of M (4x7) = 4.
    # Wait, that's not right either. The tensor T_{j,k,0} = M_{j,k} (with c-index fixed at 0).
    # But in our problem, the c-index varies: T_{j,k} = 1 for each (j,k) pair.
    # The full tensor for one row is: T[j, 7j+k, 2k] = 1 for all j,k.
    # This is a 4x7x7 tensor (after reindexing), and its CP rank is what we need.
    #
    # For the 1x4x7 problem: inputs a_0..a_3, B is 4x7, outputs c_0..c_6.
    # c_k = sum_j a_j B[j,k]. 
    # The tensor is: sum_{j,k} a_j * B[j,k] * c_k
    # CP rank of this tensor: it's the rank of the 4x7 matrix (B), which is 4 (if full rank).
    # But that's the rank as a linear operator, not CP rank.
    # CP rank of T_{j,k,l} where T_{j,k,l} = 1 if l=k else 0... no.
    #
    # In our specific tensor: T[j, 7j+k, 2k] = 1.
    # For fixed row i, the non-zero entries are at positions (j, 7j+k, 2k) for j=0..3, k=0..6.
    # That's 28 entries, each = 1.
    # The CP rank of this sparse tensor: each entry is a separate rank-1 term unless we can combine.
    # Two entries (j1, 7j1+k1, 2k1) and (j2, 7j2+k2, 2k2) can share a rank term only if
    # they have the same j, k, or the structure allows linear combination.
    # Since all entries are 1, and the indices are all distinct, the minimum rank is 28
    # UNLESS we use nonlinear combinations (like Strassen).
    #
    # For 1x4x7, the Strassen-like approach: group j into pairs.
    # For j-pair (0,1): a_0, a_1 and B[0,k], B[1,k] for k=0..6
    # Outputs: c_k gets contribution a_0*B[0,k] + a_1*B[1,k]
    # This is a 2x1 x 1x7 = 2x7 problem. Rank is 2 (just two rank-1 terms per k? No.)
    # Actually for fixed k: a_0*B[0,k] + a_1*B[1,k] is rank 2 in (a, c) space.
    # For all 7 k's: we need 7*2 = 14 rank terms? No, that's schoolbook.
    #
    # Strassen for 2x2x2: 7 rank terms instead of 8.
    # For 2x1 x 1x7: we can tile with 2x1x1 blocks (rank 2 each) -> 14 total. No savings.
    # Or 2x1x2 blocks (rank 3 each via Winograd) -> 3 blocks for k=0-1,2-3,4-5 + 1 for k=6
    # = 3*3 + 2 = 11 per j-pair. Two j-pairs: 22.
    #
    # Winograd for 1x2x2 (vector length 2 times 2x2 matrix): rank 3.
    # For 2x1 x 1x2 (two inputs, 2 outputs via 2x2 matrix): 
    # c_0 = a_0*b_00 + a_1*b_10, c_1 = a_0*b_01 + a_1*b_11
    # This is a 2x2 matrix-vector product, rank 4 schoolbook, rank 3 via Winograd.
    #
    # So for j-pair (0,1), k-pair (2s, 2s+1): rank 3.
    # k-pairs: (0,1), (2,3), (4,5), and k=6 alone (rank 2).
    # Per j-pair: 3+3+3+2 = 11.
    # Two j-pairs: 22.
    # Total for both rows: 22 (since W-supports are disjoint, no sharing needed).
    #
    # Wait, but the two rows have disjoint W-supports, so total rank = 11 + 11 = 22?
    # That seems too low. Let me verify: for row 0, W-index is 2k (even), for row 1, W-index is 2k+1 (odd).
    # So yes, they're independent. Rank(2x4x7) <= 22.
    #
    # But can we do even better by sharing across rows? No, because W supports are disjoint.
    #
    # Let me construct the rank-22 decomposition.

    def get_rank22():
        """
        Decouple into two 1x4x7 problems with disjoint W-supports.
        For each row i, use Winograd rank-3 for 2x1x2 blocks + rank-2 for k=6.
        Per row: 3 j-pairs * (3 k-pairs * rank3 + 1 * rank2) = 3*(9+2) = 33? No.
        
        Wait, I need to be more careful. For row i:
        c_k = sum_{j=0}^{3} a_j * B[j,k]
        
        Split j into pairs: (0,1) and (2,3).
        For j-pair (0,1): contribution to c_k is a_0*B[0,k] + a_1*B[1,k]
        For j-pair (2,3): contribution to c_k is a_2*B[2,k] + a_3*B[3,k]
        
        For j-pair (0,1), k-pair (2s, 2s+1):
        c_{2s} = a_0*B[0,2s] + a_1*B[1,2s]
        c_{2s+1} = a_0*B[0,2s+1] + a_1*B[1,2s+1]
        This is a 2x2 matrix-vector product (2 inputs, 2 outputs).
        Winograd rank-3: 
          M1 = (a_0+a_1) * (B[0,2s] + B[1,2s+1])  -> contributes to both
          M2 = (a_1) * (B[1,2s])  -> c_{2s}
          M3 = (a_0) * (B[0,2s+1])  -> c_{2s+1}
        Wait, that's not quite right. Let me use the standard Winograd for 2x2:
          c_0 = a_0*b_00 + a_1*b_10
          c_1 = a_0*b_01 + a_1*b_11
          M1 = (a_0+a_1)*(b_00+b_11)
          M2 = a_1*b_10
          M3 = a_0*b_01
          c_0 = M1 + M2 - M3  ... no, that gives a_0*b_00 + a_0*b_11 + a_1*b_00 + a_1*b_11 + a_1*b_10 - a_0*b_01
          That's not right.
        
        Standard Winograd F(2) for 2x2 matrix-vector:
          M1 = (a_0+a_1) * (b_00+b_11)
          M2 = (a_1) * (b_10)  
          M3 = (a_0) * (b_01)
          c_0 = M1 + M2 - M3  = a_0*b_00 + a_0*b_11 + a_1*b_00 + a_1*b_11 + a_1*b_10 - a_0*b_01
          That's not c_0 = a_0*b_00 + a_1*b_10.
        
        I think I'm confusing the Winograd scheme. Let me just use the known
        rank-3 scheme for 2x2x2 Strassen adapted to this case.
        
        Actually, for the specific tensor where T[j, 7j+k, 2k] = 1,
        the structure is very specific. Let me just try to find a rank-22
        decomposition via ALS and verify.
        """
        rank = 22
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # For row 0 (i=0): W-index 2k for k=0..6 -> W[0], W[2], W[4], ..., W[12]
        # For row 1 (i=1): W-index 2k+1 for k=0..6 -> W[1], W[3], W[5], ..., W[13]
        
        # Let me just set up the problem and use ALS to find the decomposition.
        # Initialize with random values and let ALS converge.
        rng = np.random.default_rng(123)
        U_init = rng.normal(0, 1, (n * m, rank))
        V_init = rng.normal(0, 1, (m * p, rank))
        W_init = rng.normal(0, 1, (p * n, rank))
        
        U_sol, V_sol, W_sol = als_fast(T1, T2, T3, U_init, V_init, W_init, max_iters=80)
        U_sol, V_sol, W_sol = polish(U_sol, V_sol, W_sol, max_iters=30)
        
        if verify_decomp(U_sol, V_sol, W_sol):
            return U_sol, V_sol, W_sol
        return None

    # Try rank 22
    r22 = get_rank22()
    if r22 is not None:
        best_rank = 22
        best_U, best_V, best_W = r22
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Try rank 24-49 via ALS
    rng = np.random.default_rng(42)
    for target_rank in range(49, 23, -1):
        if time.time() > deadline - 15:
            break
        found = False
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            if restart == 0:
                U_init = U50[:, :min(target_rank, 50)].copy()
                V_init = V50[:, :min(target_rank, 50)].copy()
                W_init = W50[:, :min(target_rank, 50)].copy()
                if target_rank < 50:
                    U_init = np.hstack([U_init, rng.normal(0, 0.1, (n*m, target_rank - 50))]) if target_rank > 50 else U_init
                    V_init = np.hstack([V_init, rng.normal(0, 0.1, (m*p, target_rank - 50))]) if target_rank > 50 else V_init
                    W_init = np.hstack([W_init, rng.normal(0, 0.1, (p*n, target_rank - 50))]) if target_rank > 50 else W_init
                U_init += rng.normal(0, 0.05, (n*m, target_rank))
                V_init += rng.normal(0, 0.05, (m*p, target_rank))
                W_init += rng.normal(0, 0.05, (p*n, target_rank))
            else:
                U_init = rng.normal(0, 1, (n*m, target_rank))
                V_init = rng.normal(0, 1, (m*p, target_rank))
                W_init = rng.normal(0, 1, (p*n, target_rank))
            
            U_sol, V_sol, W_sol = als_fast(T1, T2, T3, U_init, V_init, W_init, max_iters=60)
            U_sol, V_sol, W_sol = polish(U_sol, V_sol, W_sol, max_iters=20)
            
            if verify_decomp(U_sol, V_sol, W_sol):
                best_rank = target_rank
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                found = True
                break
        if found:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
