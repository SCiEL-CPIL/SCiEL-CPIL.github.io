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

    def verify_decomp(U, V, W, atol=1e-8):
        """Check if sum_r U[:,r] V[:,r] W[:,r] == T"""
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
        """Construct a known valid rank-50 decomposition using Strassen blocks."""
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (a+d)(e+h)
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M2: (c+d)e
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            # M3: a(f-h)
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M4: d(g-e)
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            # M5: (a+b)h
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            # M6: (c-a)(e+f)
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M7: (b-d)(g+h)
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        # Tile 2x4 x 4x7:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        # Handle column 6 of B (and C) separately with rank-1 terms
        for j in range(m):
            U[0 * m + j, col] = 1  # A[0,j]
            V[j * p + 6, col] = 1  # B[j,6]
            W[6 * n + 0, col] = 1  # C[0,6]
            col += 1
            U[1 * m + j, col] = 1  # A[1,j]
            V[j * p + 6, col] = 1  # B[j,6]
            W[6 * n + 1, col] = 1  # C[1,6]
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    def get_rank42():
        """
        Decouple into two 1x4x7 vector-matrix products (n=2).
        For each row i of A (a 1x4 vector) times B (4x7), use overlapping
        2-element windows with rank-3 Strassen-like schemes.
        
        For row i, the 4 inputs a_j (j=0..3) produce 7 outputs c_k (k=0..6).
        Window decomposition:
          - c_0 = a_0*b_00  (rank 1)
          - c_1 = a_0*b_01 + a_1*b_11  (window 0-1, k=1)
          - c_2 = a_1*b_12 + a_2*b_22  (window 1-2, k=2)
          - c_3 = a_2*b_23 + a_3*b_33  (window 2-3, k=3)
          - c_4 = a_3*b_34  (rank 1)
          - c_5 = a_0*b_05 + a_1*b_15 + a_2*b_25 + a_3*b_35  (need 4 rank-1)
          - c_6 = a_0*b_06 + a_1*b_16 + a_2*b_26 + a_3*b_36  (need 4 rank-1)
        
        This gives 1+1+1+1+1+4+4 = 13 per row, 26 total.
        But we can do better by using Strassen on pairs of adjacent windows.
        
        Alternative: Use rank-3 scheme for each pair of overlapping windows.
        For windows (j, j+1) contributing to outputs (j+1, j+2):
          c_{j+1} = a_j * b_{j, j+1} + a_{j+1} * b_{j+1, j+1}
          c_{j+2} = a_{j+1} * b_{j+1, j+2} + a_{j+2} * b_{j+2, j+2}
        
        Actually, let me use a cleaner approach:
        For each row i, treat as 1x4 x 4x7.
        Use the following scheme with rank 12 per row (24 total):
        
        Split the 4x7 B into:
          B_left (4x3): columns 0,1,2
          B_mid  (4x3): columns 3,4,5  
          B_right(4x1): column 6
        
        c_0 = a_0*b_00
        c_1 = a_0*b_01 + a_1*b_11
        c_2 = a_1*b_12 + a_2*b_22
        c_3 = a_2*b_23 + a_3*b_33
        c_4 = a_3*b_34
        c_5 = sum a_j*b_j5
        c_6 = sum a_j*b_j6
        
        Rank-1 terms: c_0, c_4, and 4+4 for c_5,c_6 = 10
        Overlapping windows: c_1, c_2, c_3 each need 2 rank-1 = 6
        Total per row: 16, total 32. Still not 24.
        
        Let me try a different split. Use rank-3 Strassen for 2x2x2 blocks:
        For row i, consider B as 4x7. Group columns as [0,1], [2,3], [4,5], [6].
        For each 2-column group, use a 2-input, 2-output scheme.
        
        Actually, the cleanest: for 1x4 x 4x7, the tensor rank is at most 4*7=28,
        but can be reduced. Known result: rank(1xm x mxp) <= m*p - 2 for m,p >= 2
        using Winograd. For 1x4x4, rank is 5 (not 4). For 1x4x7, let's use
        overlapping rank-3 schemes.
        
        Scheme: Use 3 overlapping 2x2x2 Strassen blocks + rank-1 for remainder.
        Block 1: rows 0-1 of B, cols 0-1 -> outputs c_0, c_1 (partial)
        Block 2: rows 1-2 of B, cols 2-3 -> outputs c_2, c_3 (partial)
        Block 3: rows 2-3 of B, cols 4-5 -> outputs c_4, c_5 (partial)
        Plus rank-1 for c_6 and remaining partials.
        
        This is getting complex. Let me just use rank 28 per row pair = 28 total
        by sharing the 4 B columns between the two rows.
        
        Actually the key insight: since n=2, the W matrix has shape (14, rank).
        W[k*n+i, r] for i=0,1. The two rows of A share the same B.
        
        Let me construct rank 28 directly:
        For each j in 0..3 (B row) and k in 0..6 (B col), we need:
          A[0,j]*B[j,k] -> C[0,k]
          A[1,j]*B[j,k] -> C[1,k]
        
        If we use rank-1 terms: A[0,j] * B[j,k] * C[0,k] and A[1,j] * B[j,k] * C[1,k]
        That's 28 terms. But we can combine: for fixed (j,k), use one rank term
        that contributes to both rows? No, because A[0,j] and A[1,j] are independent.
        
        So minimum is 28 unless we use nonlinear combinations.
        
        Let me use a hybrid: rank-3 Strassen for 2x2 blocks in the j dimension,
        sharing across k.
        
        For j-pair (0,1) and k in 0..6:
          Use Strassen: 7 rank terms for (A[0,0],A[0,1],A[1,0],A[1,1]) x (B[0,k],B[1,k])
          -> C[0,k], C[1,k]
        Similarly for j-pair (2,3): 7 more rank terms.
        Total: 14 rank terms!
        
        Wait, Strassen 2x2x2: inputs 2x2, outputs 2x2, rank 7.
        Here for fixed k, we have 2x1 x 2x1 = 2x1 output. That's rank 2, not 7.
        
        For 2x2 (A rows) x 1 (B col) -> 2x1 (C rows): rank is 2 (just a_0*b_0 + a_1*b_1).
        
        So for each k, we need 2 rank-1 terms (one per A row). Total 2*7 = 14.
        But that's just schoolbook for the 2x4x4 case... no wait, it's 2*4*7 = 56 schoolbook.
        
        I'm confusing myself. Let me just go with rank 28 via a clean construction:
        For each (j,k) pair, create a rank term. But share structure.
        
        Actually, the simplest improvement: use the fact that for n=2,
        we can write the decomposition as a sum over j of (outer product of A[:,j] and B[j,:])
        projected onto C. Each such term has rank 1 in the (i,k) space but we need
        to represent it in the tensor format.
        
        For fixed j: A[:,j] is 2x1, B[j,:] is 1x7. Their outer product is 2x7.
        In tensor form: for each (i,k), T[i*m+j, j*p+k, k*n+i] = 1.
        This is a rank-1 tensor in (i,k) for fixed j, but we need to represent it
        as U[:,r] V[:,r] W[:,r].
        
        For fixed j, the contribution is: sum over i,k of A[i,j]*B[j,k]*C[i,k]
        In our tensor: for each i in {0,1}, k in {0..6}, the entry (i*4+j, j*7+k, k*2+i) = 1.
        This is a sum of 14 rank-1 tensors (one per (i,k) pair).
        
        So total rank = 4 * 14 = 56 (schoolbook). No savings.
        
        The only way to save rank is to use nonlinear combinations that share
        rank terms across different (i,j,k) triples. Strassen does this for 2x2x2.
        
        For n=2, m=4, p=7: we can tile with 2x2x2 Strassen blocks.
        j goes 0..3, so 2 blocks in j. k goes 0..6, so 3 blocks in k (0-1, 2-3, 4-5) + 1 leftover.
        i goes 0..1, so 1 block in i.
        
        Number of Strassen blocks: 1 * 2 * 3 = 6 blocks, each rank 7 = 42.
        Plus 1 * 2 * 1 = 2 leftover k=6 columns, each needing 2*4 = 8 rank-1 terms? No.
        For k=6, we have 2 (i) * 4 (j) = 8 rank-1 terms.
        Total: 42 + 8 = 50. That's the current rank-50.
        
        To beat 50, we need a better scheme for the leftover or a different tiling.
        
        Alternative: use 2x4x4 blocks with known rank. The rank of 2x4x4 matrix mult
        is known to be 14 (not 16). So for each 2x4x4 block, use rank 14.
        We have k in 0..6, so split into 2x4x4 (k=0..3) and 2x4x3 (k=4..6).
        Rank(2x4x4) = 14, Rank(2x4x3) <= 12 (schoolbook), or better.
        Total: 14 + 12 = 26? That seems too good.
        
        Actually rank(2x4x4) is not known to be 14. The schoolbook is 16, Strassen gives 14
        for 2x2x2 but for 2x4x4 it's more complex.
        
        Let me just try to find a rank-42 construction and verify it, falling back to 50.
        """
        rank = 42
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        # Use Strassen for 2x2x2 blocks: i=0..1, j-pairs (0,1),(2,3), k-pairs (0,1),(2,3),(4,5)
        # That's 1*2*3 = 6 blocks * 7 = 42 rank terms
        # Plus k=6: 2*4 = 8 rank-1 terms -> total 50. Still 50.
        
        # To get 42, we need to handle k=6 more cleverly.
        # Use a 2x4x1 block for k=6 with rank 4 (one per j, shared across i? No, i is independent).
        # For k=6: C[i,6] = sum_j A[i,j]*B[j,6]. This needs 4 rank-1 per i, so 8 total.
        
        # Alternative: merge k=5,6 into a 2x4x2 block with rank 5 (Winograd for 1x2x2 is rank 3,
        # so 2x4x2 might be rank 8 or 10).
        
        # Let me try: 3 Strassen blocks (k=0-1, 2-3, 4-5) = 42, plus handle k=6 with 8 terms = 50.
        # Or: 2 Strassen blocks (k=0-1, 2-3) = 28, plus 2x4x4 for k=4-7? No, only 7 cols.
        
        # Let me just return rank 50 for now and rely on ALS to find 42 or lower.
        # Actually, let me construct a valid rank-42 by using a non-standard tiling.
        
        # New approach: for each i in {0,1}, use a 1x4x7 scheme with rank 21.
        # 1x4x7: use overlapping 2-element windows.
        # c_k = sum_j a_j * b_jk
        # Use rank-3 scheme for each pair of adjacent b-columns.
        # For b-columns (k, k+1): need 2 outputs, 4 inputs (a_j for j=0..3).
        # That's a 4x1 x 1x2 = 4x2 problem, rank at most 8, but can be less.
        
        # This is getting too complex. Let me just use rank 50 as baseline and
        # let ALS try to find lower ranks.
        return None

    # Get the rank-50 decomposition as a starting point
    U50, V50, W50 = get_rank50()
    
    # Verify rank 50
    if not verify_decomp(U50, V50, W50):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 50
    best_U, best_V, best_W = U50, V50, W50

    # Precompute unfoldings of T
    T1 = T.reshape(n * m, -1)  # 8 x 392
    T2 = T.reshape(m * p, -1)  # 28 x 112
    T3 = T.reshape(p * n, -1)  # 14 x 224

    def khatri_rao(A, B):
        """
        Compute Khatri-Rao product of A (M x R) and B (N x R).
        Result is (M*N) x R.
        """
        M, R = A.shape
        N, _ = B.shape
        KR = np.zeros((M * N, R))
        for r in range(R):
            KR[:, r] = np.kron(A[:, r], B[:, r])
        return KR

    def als_decompose(T1, T2, T3, U_init, V_init, W_init, max_iters=300, tol=1e-12):
        """
        Alternating Least Squares for CP decomposition.
        """
        R = U_init.shape[1]
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for it in range(max_iters):
            # Solve for U
            KR_VW = khatri_rao(V, W)
            try:
                U_new = T1 @ np.linalg.pinv(KR_VW)
            except:
                break
            
            # Solve for V
            KR_WU = khatri_rao(W, U)
            try:
                V_new = T2 @ np.linalg.pinv(KR_WU)
            except:
                break
                
            # Solve for W
            KR_UV = khatri_rao(U, V)
            try:
                W_new = T3 @ np.linalg.pinv(KR_UV)
            except:
                break
            
            # Check convergence
            change = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
            U, V, W = U_new, V_new, W_new
            
            if change < tol:
                break
        
        return U, V, W

    def polish_decomposition(U, V, W, max_iters=200, tol=1e-14):
        """
        High-precision polish: alternate between ALS steps and a final 
        least-squares refinement on the residuals.
        """
        R = U.shape[1]
        for _ in range(max_iters):
            # Standard ALS step
            KR_VW = khatri_rao(V, W)
            U = T1 @ np.linalg.pinv(KR_VW)
            
            KR_WU = khatri_rao(W, U)
            V = T2 @ np.linalg.pinv(KR_WU)
            
            KR_UV = khatri_rao(U, V)
            W = T3 @ np.linalg.pinv(KR_UV)
            
            # Check if we're close
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(recon - T)
            if residual < tol:
                break
        
        return U, V, W

    # Try to reduce rank using faster ALS with SVD-based updates
    
    rng = np.random.default_rng(42)
    
    def als_fast(T1, T2, T3, U_init, V_init, W_init, max_iters=100, tol=1e-10):
        """
        Fast ALS using SVD for the Khatri-Rao solve.
        """
        R = U_init.shape[1]
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Solve for U: U = T1 @ (V ∘ W)^+
            # V ∘ W is (m*p * p*n) x R = 28*14 x R = 392 x R
            # Instead of forming the full KR, use the structure
            # T1 is 8 x 392, so U = T1 @ pinv(KR_VW) where KR_VW is 392 x R
            KR_VW = np.zeros((m * p * p * n, R))
            for r in range(R):
                KR_VW[:, r] = np.kron(V[:, r], W[:, r])
            
            # Use SVD for the solve
            try:
                # U = T1 @ KR_VW^+ 
                # = T1 @ (KR_VW^T @ KR_VW)^-1 @ KR_VW^T
                # Or use lstsq: solve KR_VW @ U^T = T1^T
                U_new = np.linalg.lstsq(KR_VW, T1.T, rcond=None)[0].T
            except:
                break
            
            # Solve for V
            KR_WU = np.zeros((p * n * n * m, R))
            for r in range(R):
                KR_WU[:, r] = np.kron(W[:, r], U[:, r])
            try:
                V_new = np.linalg.lstsq(KR_WU, T2.T, rcond=None)[0].T
            except:
                break
                
            # Solve for W
            KR_UV = np.zeros((n * m * m * p, R))
            for r in range(R):
                KR_UV[:, r] = np.kron(U[:, r], V[:, r])
            try:
                W_new = np.linalg.lstsq(KR_UV, T3.T, rcond=None)[0].T
            except:
                break
            
            # Check convergence
            change = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
            U, V, W = U_new, V_new, W_new
            
            if change < tol:
                break
        
        return U, V, W

    for target_rank in range(49, 34, -1):
        if time.time() > deadline - 15:
            break
            
        found = False
        for restart in range(4):
            if time.time() > deadline - 10:
                break
                
            if restart == 0:
                U_init = U50[:, :target_rank].copy()
                V_init = V50[:, :target_rank].copy()
                W_init = W50[:, :target_rank].copy()
                U_init += rng.normal(0, 0.05, U_init.shape)
                V_init += rng.normal(0, 0.05, V_init.shape)
                W_init += rng.normal(0, 0.05, W_init.shape)
            else:
                U_init = rng.normal(0, 1, (n*m, target_rank))
                V_init = rng.normal(0, 1, (m*p, target_rank))
                W_init = rng.normal(0, 1, (p*n, target_rank))
            
            U_sol, V_sol, W_sol = als_fast(T1, T2, T3, U_init, V_init, W_init, max_iters=80)
            
            recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
            residual = np.linalg.norm(recon - T)
            if residual < 1e-6:
                # Polish with higher precision
                for _ in range(50):
                    if time.time() > deadline - 8:
                        break
                    KR_VW = np.zeros((m * p * p * n, target_rank))
                    for r in range(target_rank):
                        KR_VW[:, r] = np.kron(V_sol[:, r], W_sol[:, r])
                    U_sol = np.linalg.lstsq(KR_VW, T1.T, rcond=None)[0].T
                    
                    KR_WU = np.zeros((p * n * n * m, target_rank))
                    for r in range(target_rank):
                        KR_WU[:, r] = np.kron(W_sol[:, r], U_sol[:, r])
                    V_sol = np.linalg.lstsq(KR_WU, T2.T, rcond=None)[0].T
                    
                    KR_UV = np.zeros((n * m * m * p, target_rank))
                    for r in range(target_rank):
                        KR_UV[:, r] = np.kron(U_sol[:, r], V_sol[:, r])
                    W_sol = np.linalg.lstsq(KR_UV, T3.T, rcond=None)[0].T
                
                recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
                residual = np.linalg.norm(recon - T)
            
            if verify_decomp(U_sol, V_sol, W_sol):
                best_rank = target_rank
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                found = True
                break
        
        if found:
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
