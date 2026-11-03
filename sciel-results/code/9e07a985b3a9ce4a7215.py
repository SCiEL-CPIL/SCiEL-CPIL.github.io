"""Optimized matrix multiplication decomposition for 4x4 * 4x9."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 4, 4, 9
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36

    def verify_decomposition(U, V, W, tol=1e-8):
        """Verify that the decomposition is exact."""
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return np.max(np.abs(T - 1.0)) < tol

    # Strategy: Use tensor product of Strassen decompositions
    # Decompose 4x4*4x9 as (2x2*2x9) x (2x2*2x9) tensor product
    # 2x2*2x9 can be decomposed using Strassen for 2x2*2x2 (rank 7) + schoolbook for 2x2*2x7 (rank 14) = rank 21
    # But we can do better: use Strassen structure directly

    # Strassen's 7-term decomposition for 2x2*2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 2x2*2x9, we can use:
    # - 2x2*2x2 part with Strassen (rank 7)
    # - 2x2*2x7 part with schoolbook (rank 14)
    # Total: rank 21 for 2x2*2x9
    
    # Then 4x4*4x9 = (2x2*2x9) ⊗ (2x2*2x9) has rank 21*21 = 441? No, that's wrong.
    
    # Actually, let me think more carefully.
    # 4x4*4x9: split A into 4 blocks of 2x2, B into 4 blocks of 2x9
    # C_ij = sum_k A_ik B_kj
    # Using Strassen-like approach on the 4x4 block structure...
    
    # Simpler: Just use schoolbook but try to optimize with known structures
    
    # Schoolbook rank = 144
    rank_schoolbook = n * m * p  # 144
    U_fb = np.zeros((nm, rank_schoolbook))
    V_fb = np.zeros((mp, rank_schoolbook))
    W_fb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U = U_fb.copy()
    best_V = V_fb.copy()
    best_W = W_fb.copy()

    # Try Strassen-based decomposition for 4x4*4x4 part + schoolbook for 4x4*4x5
    # 4x4*4x4 with Strassen: rank 49 (known)
    # 4x4*4x5 with schoolbook: rank 80
    # Total: 129
    
    # Let's try to construct the 4x4*4x4 Strassen decomposition explicitly
    # Strassen for 4x4 uses 49 multiplications
    
    # For 4x4*4x9, we can split into:
    # - 4x4*4x4 with Strassen (rank 49)
    # - 4x4*4x5 with schoolbook (rank 80)
    # Total rank: 129
    
    # Construct 4x4*4x4 Strassen decomposition
    # This is complex, so let's try a different approach:
    # Use the fact that 4x4*4x9 can be seen as 4 independent 1x4*4x9 problems
    # Each 1x4*4x9 has rank 36 (schoolbook)
    # But we can do better with structure
    
    # Actually, let's try a block decomposition approach:
    # Split the 9 columns into groups and use different strategies
    
    # For now, let's try to find if there's a known better decomposition
    # and verify it works
    
    # Try rank 129: 4x4*4x4 Strassen + 4x4*4x5 schoolbook
    target_rank_129 = 129
    U_129 = np.zeros((nm, target_rank_129))
    V_129 = np.zeros((mp, target_rank_129))
    W_129 = np.zeros((pn, target_rank_129))
    
    # 4x4*4x5 schoolbook part (columns 4-8 of B)
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                U_129[i * 4 + j, col] = 1
                V_129[j * 9 + k, col] = 1
                W_129[k * 4 + i, col] = 1
                col += 1
    
    # Now we need 4x4*4x4 with rank 49 for the first 4 columns
    # This requires Strassen's algorithm for 4x4 matrices
    # Let's implement Strassen for 4x4*4x4
    
    # Strassen 4x4: split into 2x2 blocks
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For tensor decomposition, we need to express this in terms of rank-1 tensors
    # Each M term involves products of linear combinations
    
    # This is getting complex. Let's try a simpler approach:
    # Use the schoolbook for now but try to optimize numerically
    
    def try_optimize(n_dim, m_dim, p_dim, target_rank, max_steps=2000, n_trials=2, lr=0.01):
        """Try to find a rank-target decomposition."""
        nm_d = n_dim * m_dim
        mp_d = m_dim * p_dim
        pn_d = p_dim * n_dim

        for trial in range(n_trials):
            if time.time() > deadline:
                return None

            np.random.seed(42 + trial)
            scale = 0.1
            U_d = np.random.randn(nm_d, target_rank) * scale
            V_d = np.random.randn(mp_d, target_rank) * scale
            W_d = np.random.randn(pn_d, target_rank) * scale

            # Adam optimizer
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            m_U = np.zeros_like(U_d)
            v_U = np.zeros_like(U_d)
            m_V = np.zeros_like(V_d)
            v_V = np.zeros_like(V_d)
            m_W = np.zeros_like(W_d)
            v_W = np.zeros_like(W_d)
            t = 0

            for step in range(max_steps):
                if time.time() > deadline:
                    return None

                t += 1
                U_3d = U_d.reshape(n_dim, m_dim, target_rank)
                V_3d = V_d.reshape(m_dim, p_dim, target_rank)
                W_3d = W_d.reshape(p_dim, n_dim, target_rank)

                UV = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)
                T = np.einsum('ijkr,kir->ijk', UV, W_3d)
                err = T - 1.0

                grad_U = np.einsum('ijk,jkr,kir->ijr', err, V_3d, W_3d)
                grad_V = np.einsum('ijk,ijr,kir->jkr', err, U_3d, W_3d)
                grad_W = np.einsum('ijk,ijr,jkr->kir', err, U_3d, V_3d)

                m_U = beta1 * m_U + (1 - beta1) * grad_U
                v_U = beta2 * v_U + (1 - beta2) * grad_U**2
                U_d -= lr * (m_U / (1 - beta1**t)) / (np.sqrt(v_U / (1 - beta2**t)) + eps)

                m_V = beta1 * m_V + (1 - beta1) * grad_V
                v_V = beta2 * v_V + (1 - beta2) * grad_V**2
                V_d -= lr * (m_V / (1 - beta1**t)) / (np.sqrt(v_V / (1 - beta2**t)) + eps)

                m_W = beta1 * m_W + (1 - beta1) * grad_W
                v_W = beta2 * v_W + (1 - beta2) * grad_W**2
                W_d -= lr * (m_W / (1 - beta1**t)) / (np.sqrt(v_W / (1 - beta2**t)) + eps)

                if step % 100 == 0:
                    max_err = np.max(np.abs(T - 1.0))
                    if max_err < 1e-8:
                        return U_d, V_d, W_d

            U_3d = U_d.reshape(n_dim, m_dim, target_rank)
            V_3d = V_d.reshape(m_dim, p_dim, target_rank)
            W_3d = W_d.reshape(p_dim, n_dim, target_rank)
            UV = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)
            T = np.einsum('ijkr,kir->ijk', UV, W_3d)
            if np.max(np.abs(T - 1.0)) < 1e-8:
                return U_d, V_d, W_d

        return None

    # Try to find a decomposition with rank < 144
    # Start from 129 and go down
    for target_rank in range(129, 100, -1):
        if time.time() > deadline - 30:
            break

        result = try_optimize(4, 4, 9, target_rank, max_steps=1000, n_trials=1, lr=0.01)
        if result is not None:
            U_d, V_d, W_d = result
            if verify_decomposition(U_d, V_d, W_d):
                best_rank = target_rank
                best_U = U_d.copy()
                best_V = V_d.copy()
                best_W = W_d.copy()
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
