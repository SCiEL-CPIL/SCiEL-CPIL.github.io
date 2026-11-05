import numpy as np
import time
from scipy.linalg import solve

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W):
        """Check if sum_r U[:,r] V[:,r] W[:,r] == T"""
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=1e-9)

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

    def get_winograd_rank24():
        """
        Winograd-based rank-24 decomposition exploiting n=2.
        
        For each row i of A (i=0,1), we compute C[i,:] = A[i,:] * B.
        This is a 1x4 x 4x7 matrix multiplication.
        
        Using Winograd F(4,2) style tiling: partition B into overlapping 2x7 blocks.
        Actually, let's use a cleaner approach:
        
        For 1x4 x 4x7, we can tile into overlapping 1x2 x 2x2 blocks.
        A 1x2 x 2x2 multiplication has rank 4 (schoolbook) but can be done in rank 4.
        
        Let me think differently. The tensor for 1x4 x 4x7 has entries:
        T[j, k, l] = 1 if l = j + k - something... no.
        
        Actually T[j, k, l] = 1 where the output index l corresponds to position in C.
        For row i: C[i, l] = sum_j A[i,j] * B[j, l]
        So T[i*m+j, j*p+l, l*n+i] = 1
        
        For i=0: T[j, j*p+l, l] = 1 for j in 0..3, l in 0..6
        For i=1: T[4+j, j*p+l, 7+l] = 1 for j in 0..3, l in 0..6
        
        Each is a 4x7 matrix multiplication viewed as a 1x4 x 4x7 tensor.
        
        Winograd approach: tile the 4 columns of A into overlapping pairs.
        Use F(4,2) Winograd: for computing 4 outputs from 4 inputs with 2-wide filter...
        
        Actually, let's use a simpler tiling:
        Partition the 7 output columns into: [0,1], [1,2], [2,3], [3,4], [4,5], [5,6]
        That's 6 blocks of 2 consecutive outputs, with overlaps.
        
        For each pair of consecutive outputs (l, l+1), we need:
        C[l] = A[0]*B[0,l] + A[1]*B[1,l] + A[2]*B[2,l] + A[3]*B[3,l]
        C[l+1] = A[0]*B[0,l+1] + A[1]*B[1,l+1] + A[2]*B[2,l+1] + A[3]*B[3,l+1]
        
        This is a 1x4 x 4x2 problem. We can use Winograd F(4,2):
        Input: A = [a0, a1, a2, a3]
        Filter: B[:, l:l+2] = [[b00, b01], [b10, b11], [b20, b21], [b30, b31]]
        
        Winograd F(4,2) for 1D convolution:
        d = [a0+a2, a1+a3, a0-a2, a1-a3]  (4 multiplications)
        e = [b00+b10, b01+b11, b20+b30, b21+b31]  (4 multiplications)
        Wait, that's not right. Let me use the standard F(4,2) transform.
        
        For 1x4 x 4x2, the Winograd algorithm:
        m=4 (input length), r=2 (filter/output length), so we need r+m-1 = 5 points.
        
        Actually, let me use a different approach. Since n=2, I'll handle both rows together.
        
        Let me try a direct construction:
        For the full 2x4 x 4x7 problem, I'll use overlapping 2x2 tiles in the output dimension.
        
        Output columns: 0,1,2,3,4,5,6
        Pairs: (0,1), (1,2), (2,3), (3,4), (4,5), (5,6) -> 6 pairs
        
        For each pair (l, l+1), and for each input column pair (j, j+1) for j=0,1,2:
        We have a 2x2 x 2x2 sub-problem (2 input rows, 2 input cols, 2 output cols).
        But we also have the coupling between j and j+1...
        
        Let me try yet another approach. I'll use the fact that:
        C = A * B where A is 2x4 and B is 4x7.
        
        Split A into columns: A = [A0, A1, A2, A3] where Aj is 2x1
        Split B into columns: B = [B0, B1, B2, B3, B4, B5, B6] where Bj is 4x1
        
        C[:,l] = A0*B0[l] + A1*B1[l] + A2*B2[l] + A3*B3[l]
        
        This is a sum of 4 rank-2 tensors (each Aj * Bj^T contributes to all 7 columns).
        Total rank = 4 * 2 = 8 per... no, that's not right either.
        
        Let me just try to find a valid low-rank decomposition numerically and see what rank we can achieve.
        
        For now, return a known valid decomposition.
        """
        # Use the Strassen-based rank-50 as fallback, but let's try to construct rank-24
        # Actually, let me try a simpler approach: use rank-1 terms strategically
        
        # The minimal rank for 2x4 x 4x7 is at least max(2*4, 4*7, 2*7) = max(8,28,14) = 28
        # by the trivial lower bound... no, that's not the right lower bound.
        # The rank is at least the matrix rank of any unfolding.
        # T1 is 8x392, T2 is 28x112, T3 is 14x224
        # The rank of T is at least the max matrix rank of its unfoldings.
        
        # Let me just return the rank-50 for now and use ALS to try to reduce it.
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

    def get_rank28_direct():
        """
        Direct rank-28 construction.
        
        For each output column l (0..6) and each input column j (0..3),
        we have a rank-2 term: A[:,j] * B[j,l] contributes to C[:,l].
        But that's 4*7 = 28 terms, each of rank 2 (since A[:,j] is 2x1 and we need to 
        separate the two rows).
        
        Actually, C[i,l] = sum_j A[i,j] * B[j,l]
        
        For fixed j and l, the contribution is A[:,j] * B[j,l] which is a 2x1 outer product
        scaled by B[j,l]. This gives:
        U[i*m+j, r] = A[i,j] = delta(i, anything) -- no, A is the input, not fixed.
        
        Wait, in the tensor decomposition, U, V, W are FIXED matrices that work for ALL inputs.
        The tensor T[i*m+j, j*p+l, l*n+i] = 1 means:
        C[l*n+i] = sum_{j} A[i*m+j] * B[j*p+l]
        
        So for each (j, l) pair, we need a term where:
        U[i*m+j, r] * V[j*p+l, r] * W[l*n+i, r] = 1 for all i
        
        If we set for each (j,l): U[0*m+j, r] = 1, U[1*m+j, r] = 1, V[j*p+l, r] = 1, W[l*n+0, r] = 1, W[l*n+1, r] = 1
        Then the product gives 1*1*1 = 1 for both i=0 and i=1. 
        But wait, that would give C[l] = sum_j A[0,j]*B[j,l] + sum_j A[1,j]*B[j,l] for both rows,
        which is wrong.
        
        We need to separate the rows. So for each (j,l), we need TWO terms:
        Term 1: U[0*m+j]=1, U[1*m+j]=0, V[j*p+l]=1, W[l*n+0]=1, W[l*n+1]=0
        Term 2: U[0*m+j]=0, U[1*m+j]=1, V[j*p+l]=1, W[l*n+0]=0, W[l*n+1]=1
        
        That's 2*4*7 = 56 terms (schoolbook).
        
        To reduce rank, we need to share terms across different (j,l) or (i) values.
        
        Key insight: for fixed j, we can share the V factor. For fixed l, we can share the W factor.
        
        Let's try: for each j (0..3), create 7 terms (one per l) but with shared structure.
        Or for each l (0..6), create 4 terms (one per j).
        
        Actually, the standard way: rank of matmul tensor [n,m,p] is at least max(n*m, m*p, p*n) 
        by the slice rank bound... no, that's not right.
        
        Let me just use ALS to find a low-rank decomposition.
        """
        # Return rank-50 for now
        return get_rank50()

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

    def als_decompose(T1, T2, T3, U_init, V_init, W_init, max_iters=200, tol=1e-12):
        """
        Alternating Least Squares for CP decomposition using lstsq for better conditioning.
        """
        R = U_init.shape[1]
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Solve for U
            KR_VW = khatri_rao(V, W)
            try:
                U_new, _, _, _ = np.linalg.lstsq(KR_VW, T1.T, rcond=None)
                U_new = U_new.T
            except:
                break
            
            # Solve for V
            KR_WU = khatri_rao(W, U)
            try:
                V_new, _, _, _ = np.linalg.lstsq(KR_WU, T2.T, rcond=None)
                V_new = V_new.T
            except:
                break
                
            # Solve for W
            KR_UV = khatri_rao(U, V)
            try:
                W_new, _, _, _ = np.linalg.lstsq(KR_UV, T3.T, rcond=None)
                W_new = W_new.T
            except:
                break
            
            # Check convergence
            change = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
            U, V, W = U_new, V_new, W_new
            
            if change < tol:
                break
        
        return U, V, W

    def try_reduce_rank(U_curr, V_curr, W_curr, target_rank, rng, max_restarts=10):
        """Try to find a valid decomposition at target_rank by modifying current solution."""
        R = U_curr.shape[1]
        if R == target_rank:
            return U_curr, V_curr, W_curr, True
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            if target_rank == R - 1:
                # Remove one column
                U_init = U_curr[:, :target_rank].copy()
                V_init = V_curr[:, :target_rank].copy()
                W_init = W_curr[:, :target_rank].copy()
                # Add small noise
                U_init += rng.normal(0, 0.001, U_init.shape)
                V_init += rng.normal(0, 0.001, V_init.shape)
                W_init += rng.normal(0, 0.001, W_init.shape)
            else:
                # Random init with structure: initialize with sparse patterns
                U_init = rng.normal(0, 1, (n*m, target_rank))
                V_init = rng.normal(0, 1, (m*p, target_rank))
                W_init = rng.normal(0, 1, (p*n, target_rank))
                # Make some entries zero to encourage sparsity
                mask_u = rng.random(U_init.shape) < 0.3
                mask_v = rng.random(V_init.shape) < 0.3
                mask_w = rng.random(W_init.shape) < 0.3
                U_init[mask_u] = 0
                V_init[mask_v] = 0
                W_init[mask_w] = 0
            
            U_sol, V_sol, W_sol = als_decompose(T1, T2, T3, U_init, V_init, W_init, max_iters=100)
            
            # Round to nearest half-integer and verify
            U_round = np.round(U_sol * 2) / 2
            V_round = np.round(V_sol * 2) / 2
            W_round = np.round(W_sol * 2) / 2
            
            if verify_decomp(U_round, V_round, W_round):
                return U_round, V_round, W_round, True
            
            # Also try without rounding
            if verify_decomp(U_sol, V_sol, W_sol):
                return U_sol, V_sol, W_sol, True
            
            # Try rounding to integers
            U_int = np.round(U_sol)
            V_int = np.round(V_sol)
            W_int = np.round(W_sol)
            if verify_decomp(U_int, V_int, W_int):
                return U_int, V_int, W_int, True
        
        return None, None, None, False

    rng = np.random.default_rng(42)
    
    # Try to reduce rank from 49 down to 25
    for target_rank in range(49, 24, -1):
        if time.time() > deadline - 15:
            break
        
        # Allocate more time for lower ranks (harder)
        max_restarts = max(3, 12 - (49 - target_rank))
        U_sol, V_sol, W_sol, found = try_reduce_rank(U50, V50, W50, target_rank, rng, max_restarts)
        
        if found:
            best_rank = target_rank
            best_U, best_V, best_W = U_sol, V_sol, W_sol
            # Continue trying lower ranks
    
    # Final verification
    if not verify_decomp(best_U, best_V, best_W):
        # Fallback to schoolbook
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
