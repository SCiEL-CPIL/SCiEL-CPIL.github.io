import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS
    NM, MP, PN = n * m, m * p, p * n
    
    # Build the correct matrix multiplication tensor T for 3x4x6
    T = np.zeros((NM, MP, PN), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    T_flat = T.flatten()

    def check_exact(U, V, W, rank, tol=1e-8):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec.flatten(), T_flat, atol=tol)

    def get_schoolbook_solution():
        rank = n * m * p
        U = np.zeros((NM, rank))
        V = np.zeros((MP, rank))
        W = np.zeros((PN, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def als_decompose_small(n_sub, m_sub, p_sub, rank, max_iter=100, time_limit=5.0):
        """
        ALS for M(n_sub, m_sub, p_sub) with given rank.
        Returns U, V, W or None.
        """
        NM_s = n_sub * m_sub
        MP_s = m_sub * p_sub
        PN_s = p_sub * n_sub
        
        T_s = np.zeros((NM_s, MP_s, PN_s), dtype=np.float64)
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    a = i * m_sub + j
                    b = j * p_sub + k
                    c = k * n_sub + i
                    T_s[a, b, c] = 1.0
        T_s_flat = T_s.flatten()

        def check_exact_s(U, V, W, rank, tol=1e-8):
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            return np.allclose(T_rec.flatten(), T_s_flat, atol=tol)

        def als_run(U, V, W, rank, max_iter=100, tol=1e-12, time_limit=5.0):
            U = U.copy()
            V = V.copy()
            W = W.copy()
            start = time.time()
            
            for it in range(max_iter):
                if time.time() - start > time_limit:
                    break
                
                # Solve for U
                HW = np.einsum('br,cr->bcr', V, W).reshape(rank, MP_s * PN_s)
                T_reshaped = T_s.reshape(NM_s, MP_s * PN_s)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(HW.T, T_reshaped.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                
                # Solve for V
                UW = np.einsum('ar,cr->acr', U, W).reshape(rank, NM_s * PN_s)
                T_reshaped_v = T_s.reshape(MP_s, NM_s * PN_s).T
                try:
                    V_new, _, _, _ = np.linalg.lstsq(UW.T, T_reshaped_v, rcond=None)
                    V = V_new.T
                except:
                    pass
                
                # Solve for W
                UV = np.einsum('ar,br->abr', U, V).reshape(rank, NM_s * MP_s)
                T_reshaped_w = T_s.reshape(PN_s, NM_s * MP_s).T
                try:
                    W_new, _, _, _ = np.linalg.lstsq(UV.T, T_reshaped_w, rcond=None)
                    W = W_new.T
                except:
                    pass
                
                T_rec = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.sum((T_rec - T_s) ** 2)
                if err < tol:
                    break
            
            return U, V, W, err

        # Try multiple seeds
        for seed in range(20):
            if time.time() > deadline - 10:
                return None
            rng = np.random.RandomState(seed * 123 + 456)
            U = rng.randn(NM_s, rank) * 0.5
            V = rng.randn(MP_s, rank) * 0.5
            W = rng.randn(PN_s, rank) * 0.5
            
            U_opt, V_opt, W_opt, err = als_run(U, V, W, rank, max_iter=50, time_limit=2.0)
            
            if err < 1e-8:
                if check_exact_s(U_opt, V_opt, W_opt, rank):
                    return U_opt, V_opt, W_opt
                
                # Try rounding
                U_r = np.round(U_opt * 2) / 2
                V_r = np.round(V_opt * 2) / 2
                W_r = np.round(W_opt * 2) / 2
                if check_exact_s(U_r, V_r, W_r, rank):
                    return U_r, V_r, W_r
                
                U_i = np.round(U_opt)
                V_i = np.round(V_opt)
                W_i = np.round(W_opt)
                if check_exact_s(U_i, V_i, W_i, rank):
                    return U_i, V_i, W_i
        
        return None

    def build_full_from_sub(sub_U, sub_V, sub_W, sub_rank):
        """
        Builds full 3x4x6 decomposition from two 3x2x6 decompositions.
        """
        # sub_U: (6, sub_rank), sub_V: (12, sub_rank), sub_W: (18, sub_rank)
        # Full U: (12, 2*sub_rank)
        # Full V: (24, 2*sub_rank)
        # Full W: (18, 2*sub_rank)
        
        full_rank = 2 * sub_rank
        full_U = np.zeros((12, full_rank))
        full_V = np.zeros((24, full_rank))
        full_W = np.zeros((18, full_rank))
        
        # First part (A1, B1)
        full_U[0:6, 0:sub_rank] = sub_U
        full_V[0:12, 0:sub_rank] = sub_V
        full_W[0:18, 0:sub_rank] = sub_W
        
        # Second part (A2, B2) - identical structure, just shifted indices
        full_U[6:12, sub_rank:2*sub_rank] = sub_U
        full_V[12:24, sub_rank:2*sub_rank] = sub_V
        full_W[0:18, sub_rank:2*sub_rank] = sub_W
        
        return {"U": full_U.tolist(), "V": full_V.tolist(), "W": full_W.tolist()}

    # Search for M(3,2,6) rank
    # Try 32, 31, 30
    best_sub = None
    for r in [32, 31, 30]:
        if time.time() > deadline - 20:
            break
        res = als_decompose_small(3, 2, 6, r)
        if res is not None:
            best_sub = res
            break
    
    if best_sub is not None:
        U_sub, V_sub, W_sub = best_sub
        result = build_full_from_sub(U_sub, V_sub, W_sub, U_sub.shape[1])
        if check_exact(np.array(result['U']), np.array(result['V']), np.array(result['W']), len(result['U'][0])):
            return result

    # Fallback: Rank 66 construction using M(3,4,2) rank 22
    # Or just return schoolbook if time is tight, but 66 is known.
    # Let's try to find M(3,4,2) rank 22 quickly if we have time.
    # Actually, the previous best was 66. We can just return schoolbook (72) if we don't find better,
    # but we know 66 is possible.
    # To be safe and fast, if we didn't find <66, we return the known 66 structure if we can generate it fast,
    # or just schoolbook.
    # The prompt says "score is the rank". Lower is better.
    # If we fail to find 64, we should try to find 66.
    
    if time.time() < deadline - 30:
        # Try M(3,4,2) rank 22
        res_42 = als_decompose_small(3, 4, 2, 22)
        if res_42 is not None:
            U_42, V_42, W_42 = res_42
            # Build 3x4x6 from three 3x4x2
            full_rank = 3 * 22
            full_U = np.zeros((12, full_rank))
            full_V = np.zeros((24, full_rank))
            full_W = np.zeros((18, full_rank))
            
            # The 6 in p is split into 2+2+2
            # W index: k*n+i. k in 0..5.
            # k=0,1 -> W_42 part 1?
            # W_42 has 18 rows? No, M(3,4,2) has PN = 2*3 = 6 rows.
            # Wait. M(3,4,2): n=3, m=4, p=2.
            # NM=12, MP=8, PN=6.
            # So U_42 is 12x22, V_42 is 8x22, W_42 is 6x22.
            
            # We need to map the 18 rows of full W to three 6-row blocks.
            # W_full[c, r]
            # c = k*3 + i.
            # Block 1: k=0,1 -> c in 0..5.
            # Block 2: k=2,3 -> c in 6..11.
            # Block 3: k=4,5 -> c in 12..17.
            
            # V_full[b, r]
            # b = j*p + k. j in 0..3, k in 0..5.
            # This is tricky. The B matrix is 4x6.
            # We are splitting the inner dimension of A (4) or B (4)?
            # In M(3,4,2), the inner dimension is 4.
            # So we are splitting the 6 of the output into three 2s.
            # A is 3x4. B is 4x6.
            # B can be seen as three 4x2 blocks: B1, B2, B3.
            # A is the same for all.
            # So we use the SAME U and V for all three?
            # No, V depends on B.
            # V_42 is for a 4x2 matrix.
            # So we need three V matrices?
            # But the decomposition is linear.
            # C = A B1 + A B2 + A B3.
            # We can compute A B1, A B2, A B3 separately using the same A?
            # No, the rank adds up.
            # Term 1: U_1, V_1, W_1 for A B1.
            # Term 2: U_2, V_2, W_2 for A B2.
            # Term 3: U_3, V_3, W_3 for A B3.
            
            # Since A is the same, can we reuse U?
            # U is 12x22.
            # If we use the same U for all, we save rank?
            # No, the terms are distinct.
            # However, if the algorithm for M(3,4,2) is the same, U might be the same.
            # Let's just construct it generally.
            
            # U_full: 12 x 66
            # V_full: 24 x 66
            # W_full: 18 x 66
            
            # Part 1 (k=0,1)
            full_U[0:12, 0:22] = U_42
            # V_42 is 8x22 (for 4x2).
            # B1 is rows 0,1 of B (4x6).
            # B index b = j*6 + k.
            # For B1, k=0,1. b = j*6 + 0, j*6 + 1.
            # This is not contiguous in V_42.
            # V_42 index is j'*2 + k'. j' in 0..3, k' in 0..1.
            # So V_42[j'*2 + k', r].
            # We need to map this to V_full[b, r].
            # b = j*6 + k.
            # For k=0,1:
            # j=0, k=0 -> b=0. V_42[0, r]
            # j=0, k=1 -> b=1. V_42[1, r]
            # j=1, k=0 -> b=6. V_42[2, r]
            # j=1, k=1 -> b=7. V_42[3, r]
            # ...
            for j in range(4):
                for k in range(2):
                    b_idx = j * 6 + k
                    v42_idx = j * 2 + k
                    full_V[b_idx, 0:22] = V_42[v42_idx, :]
            
            # W_42 is 6x22 (for 2x3).
            # W_42 index is k'*3 + i'.
            # For Block 1, k=0,1.
            # c = k*3 + i.
            # c_idx = k*3 + i.
            # W_42[k*3 + i, r].
            full_W[0:6, 0:22] = W_42
            
            # Part 2 (k=2,3)
            full_U[0:12, 22:44] = U_42
            for j in range(4):
                for k in range(2):
                    b_idx = j * 6 + (k + 2)
                    v42_idx = j * 2 + k
                    full_V[b_idx, 22:44] = V_42[v42_idx, :]
            full_W[6:12, 22:44] = W_42
            
            # Part 3 (k=4,5)
            full_U[0:12, 44:66] = U_42
            for j in range(4):
                for k in range(2):
                    b_idx = j * 6 + (k + 4)
                    v42_idx = j * 2 + k
                    full_V[b_idx, 44:66] = V_42[v42_idx, :]
            full_W[12:18, 44:66] = W_42
            
            result_66 = {"U": full_U.tolist(), "V": full_V.tolist(), "W": full_W.tolist()}
            if check_exact(np.array(result_66['U']), np.array(result_66['V']), np.array(result_66['W']), 66):
                return result_66

    return get_schoolbook_solution()
