import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 20, 30, 24

    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomposition(U, V, W, tol=1e-10):
        r = U.shape[1]
        # Use einsum for speed
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T_recon - T))

    # Schoolbook decomposition (rank 120)
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_rank = n * m * p
    best_decomp = (U_sb, V_sb, W_sb)

    # --- Winograd FMA for 4x4x4 core ---
    # The 4x4x4 matrix multiplication tensor can be decomposed using the
    # Winograd Fourier Transform Algorithm. For size 4, we use 4-point DFT-like
    # transforms. The rank of the 4x4x4 core is 16 (4*4 pointwise products).
    # 
    # Actually, for 4x4 matrix multiplication, the Winograd scheme uses
    # m=4 transform points. The decomposition is:
    #   A (4x4) -> transform -> 4x4 matrix A'
    #   B (4x4) -> transform -> 4x4 matrix B'  
    #   C = inverse_transform(A' * B')  (element-wise product)
    # 
    # The transform matrices for 4-point Winograd are 4x4.
    # But this gives rank 16 for the 4x4x4 core.
    #
    # Let me use the explicit 4x4 Winograd transforms.
    # For 4-point, the transform uses the roots of unity of order 4: 1, i, -1, -i
    # But we need integer/half-integer entries. The standard Winograd for even sizes
    # uses real transforms.
    #
    # Actually, let me use a simpler approach: decompose the 4x4x4 core using
    # two 2x2 Strassen-like blocks, or use the known rank-16 decomposition.
    #
    # For a 4x4x4 matrix multiplication, one approach is to view it as 4 independent
    # 2x2 multiplications (if we partition appropriately), but that's not quite right.
    #
    # Let me think more carefully. The 4x4x4 tensor T4 has entries T4[i*4+j, j*4+k, k*4+i]=1.
    # We can decompose this using the Kronecker product structure.
    # T4 = T2 ⊗ T2 where T2 is the 2x2x2 tensor.
    # The rank of T2 ⊗ T2 is at most rank(T2)^2 = 4^2 = 16.
    # Since T2 has rank 4 (schoolbook for 2x2), the Kronecker gives rank 16.
    
    # 2x2x2 schoolbook (rank 4)
    def mm2x2():
        r = 4
        U = np.zeros((4, r))
        V = np.zeros((4, r))
        W = np.zeros((4, r))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    t = i*4 + j*2 + k
                    U[i*2+j, t] = 1
                    V[j*2+k, t] = 1
                    W[k*2+i, t] = 1
        return U, V, W

    U2, V2, W2 = mm2x2()
    
    # Verify 2x2x2
    T2 = np.zeros((4,4,4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T2[i*2+j, j*2+k, k*2+i] = 1.0
    T2_recon = np.einsum('ar,br,cr->abc', U2, V2, W2)
    if np.max(np.abs(T2_recon - T2)) > 1e-10:
        # Fallback to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Build 4x4x4 from Kronecker product of 2x2x2
    # The 4x4x4 tensor is T2 ⊗ T2
    # U4[i4*4+j4, t4] = U2[i2*2+j2, t2] * U2[i2'*2+j2', t2'] where i4=2*i2+i2', j4=2*j2+j2', t4=4*t2+t2'
    # Wait, this isn't quite right. Let me think about the indexing.
    # 
    # For the 4x4 matrix A, we index A[i,j] where i,j in 0..3.
    # We can write i = 2*i2 + i2', j = 2*j2 + j2' where i2,j2,i2',j2' in 0..1.
    # Then A[i,j] = A[2*i2+i2', 2*j2+j2'].
    # 
    # The 4x4 multiplication C = A*B where C[k,i] = sum_j A[i,j]*B[j,k].
    # C[2*k2+k2', 2*i2+i2'] = sum_j A[2*i2+i2', j] * B[j, 2*k2+k2']
    # = sum_{j2,j2'} A[2*i2+i2', 2*j2+j2'] * B[2*j2+j2', 2*k2+k2']
    #
    # This is NOT simply a Kronecker product of two 2x2 multiplications because
    # the summation index j is shared. So the simple Kronecker approach doesn't work directly.
    #
    # Let me use a different approach: partition the 4x4x4 into 4 blocks of 2x2x2.
    # Actually, for 4x4 matrix multiplication, we can use the following:
    # View A as a 2x2 block matrix with 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # Similarly B = [[B00, B01], [B10, B11]]
    # Then C = A*B has blocks:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Each block is a 2x2 matrix. So we have 4 block equations, each involving 2 multiplications of 2x2 matrices.
    # Total: 8 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 4, so total rank = 8*4 = 32.
    # That's worse than schoolbook 16.
    #
    # Let me try Strassen for the 2x2 block structure:
    # Using Strassen for the 2x2 block matrices (each block is 2x2):
    # M1 = (A00+A11)*(B00+B11)  -- 2x2 * 2x2
    # M2 = (A10+A11)*B00        -- 2x2 * 2x2
    # M3 = A00*(B01-B11)        -- 2x2 * 2x2
    # M4 = A11*(B10-B00)        -- 2x2 * 2x2
    # M5 = (A00+A01)*B11        -- 2x2 * 2x2
    # M6 = (A10-A00)*(B00+B01)  -- 2x2 * 2x2
    # M7 = (A01-A11)*(B10+B11)  -- 2x2 * 2x2
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each M is a 2x2 multiplication, rank 4. Total rank = 7*4 = 28.
    # Still worse than 16.
    #
    # Let me just use the schoolbook for the 4x4x4 core: rank 16.
    # 4*4*4 = 64... no wait, that's the full 4x4x4 tensor size, not the rank.
    # Schoolbook for 4x4*4x4 is rank 4*4*4 = 64? No.
    # Schoolbook for multiplying 4x4 by 4x4: for each (i,j,k), one multiplication.
    # So rank = 4*4*4 = 64? No, that's not right either.
    # 
    # Wait, I'm confusing myself. The matrix multiplication tensor for n×m * m×p has
    # entries T[i*m+j, j*p+k, k*n+i] = 1 for i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # The rank of this tensor is at most n*m*p (schoolbook).
    # For 4x4*4x4, schoolbook rank = 4*4*4 = 64.
    # 
    # But the Winograd scheme for 4x4*4x4 should give a lower rank.
    # The known rank for 4x4 matrix multiplication is 16? No, that seems too low.
    # Actually, the rank of the 4x4 matrix multiplication tensor is known to be at most 20
    # (using Winograd) or even lower. Let me check: the rank of the 2x2 tensor is 4 (schoolbook),
    # and it's known that rank(2x2) = 4. For 3x3, rank is 7 (Strassen) or 5 (Laderman, but that's for 3x3*3x3).
    # For 4x4, the best known upper bound is 20 (using Winograd FMA).
    
    # Let me implement the Winograd FMA for 4x4*4x4.
    # The 4-point Winograd transform uses the following approach:
    # We use a 4x4 transform matrix F such that F is orthogonal (up to scaling).
    # For 4-point, we can use the DFT-4 matrix, but that involves complex numbers.
    # Instead, we use the real Winograd transform.
    #
    # Actually, for the purpose of this problem, let me use a simpler decomposition.
    # I'll partition the 4x5*5x6 problem differently.
    #
    # Let me reconsider. The problem is 4x5 * 5x6.
    # I can partition this into:
    # - A 4x4*4x4 core (rows 0-3, cols 0-3 of the 5x6, and rows 0-3 of the 4x5)
    # - Strips for the remaining parts
    #
    # Actually, let me use a cleaner partition:
    # I = {0,1,2,3} (all rows of A)
    # J = {0,1,2,3,4} (all cols of A / rows of B)
    # K = {0,1,2,3,4,5} (all cols of B)
    #
    # Partition J into J1={0,1,2,3} and J2={4}
    # Partition K into K1={0,1,2,3} and K2={4,5}
    #
    # Then C = A*B has blocks:
    # C[:, K1] = A[:, J1]*B[J1, K1] + A[:, J2]*B[J2, K1]
    # C[:, K2] = A[:, J1]*B[J1, K2] + A[:, J2]*B[J2, K2]
    #
    # The first term A[:, J1]*B[J1, K1] is a 4x4 * 4x4 multiplication (rank 64 schoolbook, or lower with Winograd)
    # The other terms are smaller.
    
    # For now, let me use the Strassen-3 approach from the current program as a baseline,
    # and try to improve it.
    
    # Strassen-3: rank-7 decomposition for 3x3 * 3x3
    def strassen_3x3():
        r = 7
        U = np.zeros((9, r))
        V = np.zeros((9, r))
        W = np.zeros((9, r))
        # U idx: i*3+j ; V idx: j*3+k ; W idx: k*3+i
        t = 0
        U[0, t] = 1
        V[1*3+1, t] = 1; V[1*3+2, t] = -1; V[2*3+1, t] = -1; V[2*3+2, t] = 1
        W[0*3+0, t] = 1; W[0*3+1, t] = 1; W[0*3+2, t] = 1
        t = 1
        U[0, t] = 1; U[1, t] = 1
        V[1*3+2, t] = 1; V[2*3+2, t] = 1; V[1*3+1, t] = -1; V[2*3+1, t] = -1
        W[1*3+0, t] = 1; W[1*3+1, t] = 1; W[1*3+2, t] = 1
        t = 2
        U[1, t] = 1; U[2, t] = 1
        V[1*3+0, t] = 1; V[2*3+0, t] = 1; V[1*3+1, t] = -1; V[2*3+1, t] = -1
        W[2*3+0, t] = 1; W[2*3+1, t] = 1; W[2*3+2, t] = 1
        t = 3
        U[8, t] = 1
        V[0*3+0, t] = 1; V[0*3+1, t] = -1; V[1*3+0, t] = -1; V[1*3+1, t] = 1
        W[0*3+0, t] = 1; W[1*3+0, t] = 1; W[2*3+0, t] = 1
        t = 4
        U[6, t] = 1; U[8, t] = 1
        V[0*3+1, t] = 1; V[1*3+1, t] = 1; V[0*3+0, t] = -1; V[1*3+0, t] = -1
        W[0*3+1, t] = 1; W[1*3+1, t] = 1; W[2*3+1, t] = 1
        t = 5
        U[7, t] = 1; U[8, t] = 1
        V[0*3+0, t] = 1; V[1*3+0, t] = 1; V[0*3+1, t] = -1; V[1*3+0, t] = -1
        W[0*3+2, t] = 1; W[1*3+2, t] = 1; W[2*3+2, t] = 1
        t = 6
        U[0, t] = 1; U[4, t] = 1; U[8, t] = 1
        V[0*3+0, t] = 1; V[1*3+1, t] = 1; V[2*3+2, t] = 1
        W[0*3+0, t] = 1; W[1*3+0, t] = 1; W[2*3+0, t] = 1
        W[0*3+1, t] = 1; W[1*3+1, t] = 1; W[2*3+1, t] = 1
        W[0*3+2, t] = 1; W[1*3+2, t] = 1; W[2*3+2, t] = 1
        return U, V, W

    U_s3, V_s3, W_s3 = strassen_3x3()
    T_33 = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T_33[i*3+j, j*3+k, k*3+i] = 1.0
    T_recon_33 = np.einsum('ar,br,cr->abc', U_s3, V_s3, W_s3)
    err_33 = np.max(np.abs(T_recon_33 - T_33))

    if err_33 > 1e-10:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Full 4x5*5x6 decomposition: Strassen-3 on 3x3x3 cores + schoolbook strips.
    r_total = 7 + 7 + 18 + 18 + 9 + 9 + 6 + 6
    U_full = np.zeros((nm, r_total))
    V_full = np.zeros((mp, r_total))
    W_full = np.zeros((pn, r_total))
    col = 0

    def place_strassen(k_off, i_off):
        nonlocal col
        for t in range(7):
            for a_idx in range(9):
                val = U_s3[a_idx, t]
                if val != 0:
                    i = a_idx // 3; j = a_idx % 3
                    U_full[i*5 + j, col] += val
            for b_idx in range(9):
                val = V_s3[b_idx, t]
                if val != 0:
                    j = b_idx // 3; k = b_idx % 3
                    V_full[j*6 + (k_off + k), col] += val
            for c_idx in range(9):
                val = W_s3[c_idx, t]
                if val != 0:
                    k = c_idx // 3; i = c_idx % 3
                    W_full[(k_off + k)*4 + (i_off + i), col] += val
            col += 1

    def place_schoolbook(i_vals, j_vals, k_vals, i_off=0, j_off=0, k_off=0):
        nonlocal col
        for i in i_vals:
            for j in j_vals:
                for k in k_vals:
                    U_full[(i_off + i)*5 + (j_off + j), col] = 1
                    V_full[(j_off + j)*6 + (k_off + k), col] = 1
                    W_full[(k_off + k)*4 + (i_off + i), col] = 1
                    col += 1

    place_strassen(k_off=0, i_off=0)
    place_strassen(k_off=3, i_off=0)
    place_schoolbook(range(3), range(3, 5), range(3))
    place_schoolbook(range(3), range(3, 5), range(3, 6))
    place_schoolbook(range(3), range(3), range(3), i_off=3)
    place_schoolbook(range(3), range(3), range(3, 6), i_off=3)
    place_schoolbook(range(3), range(3, 5), range(3), i_off=3)
    place_schoolbook(range(3), range(3, 5), range(3, 6), i_off=3)

    err_full = verify_decomposition(U_full, V_full, W_full)
    if err_full < 1e-10:
        best_rank = r_total
        best_decomp = (U_full.copy(), V_full.copy(), W_full.copy())

    # --- Try to find a lower-rank decomposition using ALS with integer/half-integer constraint ---
    # The key insight: we need integer or half-integer entries.
    # Let's try to use a different partitioning that might give lower rank.
    
    # Alternative: use 2x2x2 Strassen (rank 4) for 2x2 cores
    # Partition I={0,1,2,3}, J={0,1,2,3,4}, K={0,1,2,3,4,5}
    # Use 2x2x2 blocks:
    # I into {0,1},{2,3}; J into {0,1},{2,3},{4}; K into {0,1},{2,3},{4,5}
    
    # Actually, let me try a different approach: use the fact that we can combine
    # some of the schoolbook terms.
    
    # For now, let's try ALS to see if we can reduce the rank below 80.
    np.random.seed(42)
    
    def try_als(U0, V0, W0, target_rank, max_iter=50):
        r = target_rank
        U = U0[:, :min(U0.shape[1], r)].copy()
        V = V0[:, :min(V0.shape[1], r)].copy()
        W = W0[:, :min(W0.shape[1], r)].copy()
        if U.shape[1] < r:
            pad_u = np.random.randn(U.shape[0], r - U.shape[1]) * 0.01
            pad_v = np.random.randn(V.shape[0], r - V.shape[1]) * 0.01
            pad_w = np.random.randn(W.shape[0], r - W.shape[1]) * 0.01
            U = np.hstack([U, pad_u]); V = np.hstack([V, pad_v]); W = np.hstack([W, pad_w])
        
        for _ in range(max_iter):
            if time.time() > deadline - 15:
                return None
            # Solve W
            W_new = np.zeros_like(W)
            for c in range(pn):
                X = np.einsum('ar,br->abr', U, V).reshape(r, -1).T
                W_new[c, :] = np.linalg.lstsq(X, T[:, :, c].reshape(-1), rcond=None)[0]
            W = W_new
            # Solve U
            U_new = np.zeros_like(U)
            for a in range(nm):
                X = np.einsum('br,cr->bcr', V, W).reshape(r, -1).T
                U_new[a, :] = np.linalg.lstsq(X, T[a, :, :].reshape(-1), rcond=None)[0]
            U = U_new
            # Solve V
            V_new = np.zeros_like(V)
            for b in range(mp):
                X = np.einsum('ar,cr->acr', U, W).reshape(r, -1).T
                V_new[b, :] = np.linalg.lstsq(X, T[:, b, :].reshape(-1), rcond=None)[0]
            V = V_new
            err = verify_decomposition(U, V, W)
            if err < 1e-10:
                # Round to nearest half-integer
                U_r = np.round(2 * U) / 2
                V_r = np.round(2 * V) / 2
                W_r = np.round(2 * W) / 2
                err_r = verify_decomposition(U_r, V_r, W_r)
                if err_r < 1e-10:
                    return (U_r.copy(), V_r.copy(), W_r.copy())
        return None

    for target_rank in [78, 75, 72, 70, 68, 65, 62, 60]:
        if time.time() > deadline - 20:
            break
        if target_rank >= best_rank:
            continue
        found = None
        for attempt in range(2):
            if time.time() > deadline - 15:
                break
            found = try_als(*best_decomp, target_rank, max_iter=80)
            if found is not None:
                break
        if found is not None:
            U, V, W = found
            nz = np.where(np.any(np.abs(np.stack([U, V, W], axis=1)) > 1e-12, axis=0))[0]
            U, V, W = U[:, nz], V[:, nz], W[:, nz]
            r = U.shape[1]
            if r < best_rank and verify_decomposition(U, V, W) < 1e-10:
                best_rank = r
                best_decomp = (U.copy(), V.copy(), W.copy())

    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
