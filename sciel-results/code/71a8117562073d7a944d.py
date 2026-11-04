import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W, atol=1e-8):
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

    # Get the rank-50 decomposition as a starting point
    U50, V50, W50 = get_rank50()
    if not verify_decomp(U50, V50, W50):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 50
    best_U, best_V, best_W = U50, V50, W50

    # GKS-inspired approach: Use integer/half-integer search
    # Try to find a decomposition with small integer entries
    
    def try_integer_decomposition(rank, max_entries=2):
        """
        Try to find a rank-`rank` decomposition with entries in {-max_entries, ..., max_entries}.
        Uses a greedy/iterative approach.
        """
        if time.time() > deadline - 10:
            return None
        
        # Start from rank-50 solution and try to reduce
        if rank >= 50:
            return U50[:, :rank].copy(), V50[:, :rank].copy(), W50[:, :rank].copy()
        
        # Use the structure: tile with 2x2x2 Strassen-like blocks
        # For 2x4x7, we can think of it as:
        # - Two 2x2x4 blocks (for columns 0-3 of B)
        # - Two 2x2x3 blocks (for columns 4-6 of B)
        
        # Actually, let's try a direct construction
        # The matrix multiplication 2x4 * 4x7 can be decomposed as:
        # C[0,0] = sum_j A[0,j]*B[j,0]
        # C[0,1] = sum_j A[0,j]*B[j,1]
        # ... etc.
        
        # For a rank-r decomposition, we need r terms of the form:
        # u_a * v_b * w_c where u is length 8, v is length 28, w is length 14
        
        # Try to find sparse solutions
        # Each term contributes to at most |supp(u)|*|supp(v)|*|supp(w)| entries of T
        
        # Since T has exactly 56 non-zero entries (one for each (i,j,k) with i in {0,1}, j in {0,1,2,3}, k in {0,...,6}),
        # and each rank-1 term u⊗v⊗w can cover at most some of these,
        # we need at least 56 / (max coverage per term) terms.
        
        # A single rank-1 term with u, v, w all having support 2 covers at most 8 entries.
        # So we need at least 56/8 = 7 terms. But that's a very loose bound.
        
        # The actual rank is likely around 40-50.
        
        # Let's try to verify if rank 49, 48, etc. work by attempting optimization
        return None

    # Try to verify lower ranks by checking if the rank-50 solution can be reduced
    # by checking linear dependencies
    
    def check_rank_reduction(U, V, W, target_rank):
        """
        Check if we can reduce the rank by finding linear dependencies.
        """
        if target_rank >= U.shape[1]:
            return U, V, W, False
        
        # Try to remove columns that are linearly dependent
        # This is a heuristic approach
        
        R = U.shape[1]
        if target_rank >= R:
            return U[:, :target_rank], V[:, :target_rank], W[:, :target_rank], True
        
        # Try to find a subset of columns that spans the same tensor
        # This is hard in general, so we'll use a greedy approach
        
        return None, None, None, False

    # Use a more direct approach: try specific constructions
    
    # For 2x4x7, we can partition the 4x7 matrix into:
    # - A 4x4 block and a 4x3 block
    # The 4x4 part can use rank-14 (from 2x2x2 Strassen on two 2x2 blocks)
    # The 4x3 part needs more careful handling
    
    # Actually, let's try to use the known result that 2x4x7 has rank at most 44
    # by using a combination of Strassen and rank-1 terms
    
    def get_rank44_construction():
        """
        Attempt to construct a rank-44 decomposition.
        """
        rank = 44
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        # Use Strassen for the 2x4x4 part (columns 0-3 of B)
        def add_strassen_2x2x2(i0, j0, k0):
            nonlocal col
            # M1
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            # M2
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1
            # M3
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            # M4
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1
            # M5
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1
            # M6
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            # M7
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        # Strassen on 2x2x2 blocks for j0 in {0,2}, k0 in {0,2}
        for j0 in [0, 2]:
            for k0 in [0, 2]:
                add_strassen_2x2x2(0, j0, k0)
        
        # That's 4*7 = 28 terms for the 2x4x4 part
        # Now handle columns 4,5,6 of B (the 2x4x3 part)
        
        # For the 2x4x3 part, use rank-1 terms with some sharing
        # C[0,4] = sum_j A[0,j]*B[j,4]
        # C[0,5] = sum_j A[0,j]*B[j,5]
        # C[0,6] = sum_j A[0,j]*B[j,6]
        # C[1,4] = sum_j A[1,j]*B[j,4]
        # C[1,5] = sum_j A[1,j]*B[j,5]
        # C[1,6] = sum_j A[1,j]*B[j,6]
        
        # We can use 6 rank-1 terms (one for each output)
        for j in range(m):
            for k in [4, 5, 6]:
                U[0 * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + 0, col] = 1
                col += 1
                U[1 * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + 1, col] = 1
                col += 1
        
        # That's 2*4*3 = 24 more terms, total 52. Too many.
        # We need to share terms.
        
        # Actually, the above construction is wrong. Let me restart.
        
        return None

    # The GKS approach suggested in the directive is to solve polynomial equations.
    # This is complex, so let's use a practical approach:
    # Try to find a valid decomposition with rank < 50 by verification.
    
    # For now, return the rank-50 solution if we can't find a better one.
    # The ALS approach from the parent may find rank 49 or lower.
    
    T1 = T.reshape(n * m, -1)
    T2 = T.reshape(m * p, -1)
    T3 = T.reshape(p * n, -1)

    def khatri_rao(A, B):
        M, R = A.shape
        N, _ = B.shape
        KR = np.zeros((M * N, R))
        for r in range(R):
            KR[:, r] = np.kron(A[:, r], B[:, r])
        return KR

    def als_decompose(T1, T2, T3, U_init, V_init, W_init, max_iters=200, tol=1e-12):
        R = U_init.shape[1]
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            KR_VW = khatri_rao(V, W)
            try:
                U_new = T1 @ np.linalg.pinv(KR_VW)
            except:
                break
            KR_WU = khatri_rao(W, U)
            try:
                V_new = T2 @ np.linalg.pinv(KR_WU)
            except:
                break
            KR_UV = khatri_rao(U, V)
            try:
                W_new = T3 @ np.linalg.pinv(KR_UV)
            except:
                break
            
            change = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
            U, V, W = U_new, V_new, W_new
            if change < tol:
                break
        
        return U, V, W

    rng = np.random.default_rng(12345)
    
    # Try ranks from 49 down to 40
    for target_rank in range(49, 39, -1):
        if time.time() > deadline - 10:
            break
        
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            
            if restart < 3:
                U_init = rng.normal(0, 1, (n*m, target_rank))
                V_init = rng.normal(0, 1, (m*p, target_rank))
                W_init = rng.normal(0, 1, (p*n, target_rank))
            else:
                U_init = U50[:, :target_rank].copy()
                V_init = V50[:, :target_rank].copy()
                W_init = W50[:, :target_rank].copy()
                U_init += rng.normal(0, 0.05, U_init.shape)
                V_init += rng.normal(0, 0.05, V_init.shape)
                W_init += rng.normal(0, 0.05, W_init.shape)
            
            U_sol, V_sol, W_sol = als_decompose(T1, T2, T3, U_init, V_init, W_init, max_iters=100)
            
            recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
            residual = np.linalg.norm(recon - T)
            if residual < 1e-6:
                # Polish
                for _ in range(50):
                    if time.time() > deadline - 5:
                        break
                    KR_VW = khatri_rao(V_sol, W_sol)
                    U_sol = T1 @ np.linalg.pinv(KR_VW)
                    KR_WU = khatri_rao(W_sol, U_sol)
                    V_sol = T2 @ np.linalg.pinv(KR_WU)
                    KR_UV = khatri_rao(U_sol, V_sol)
                    W_sol = T3 @ np.linalg.pinv(KR_UV)
                
                if verify_decomp(U_sol, V_sol, W_sol):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_sol, V_sol, W_sol
                    break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
