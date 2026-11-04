import time
import numpy as np
import itertools

def construct():
    _t0 = time.time()
    deadline = _t0 + 140.0

    n, m, p = 3, 4, 6

    # Possible splits for A rows (size n=3)
    splits_n = [(1, 2), (2, 1)]
    # Possible splits for A cols / B rows (size m=4)
    splits_m = [(1, 3), (2, 2), (3, 1)]
    # Possible splits for B cols (size p=6)
    # We need 3 blocks? Or 2?
    # If A cols split is [2,2], we have 2 k-blocks.
    # If B cols split is [2,2,2], we have 3 j-blocks.
    # The loop iterates over k-blocks and j-blocks.
    # Actually, the structure is:
    # for i_block in range(num_i_blocks):
    #   for k_block in range(num_k_blocks):
    #     for j_block in range(num_j_blocks):
    #       multiply A[i_block, k_block] * B[k_block, j_block]
    
    # We need a function to get the rank of a block multiplication h x w * w x z
    def get_block_rank(h, w, z):
        # h: A rows, w: A cols/B rows, z: B cols
        if h == 1 and w == 1 and z == 1: return 1
        if h == 1 and w == 2 and z == 1: return 2
        if h == 1 and w == 1 and z == 2: return 2
        if h == 1 and w == 2 and z == 2: return 4
        if h == 1 and w == 3 and z == 1: return 3
        if h == 1 and w == 1 and z == 3: return 3
        if h == 1 and w == 3 and z == 2: return 6
        if h == 1 and w == 2 and z == 3: return 6
        if h == 2 and w == 2 and z == 1: return 4
        if h == 1 and w == 2 and z == 2: return 4
        if h == 2 and w == 1 and z == 2: return 4
        if h == 2 and w == 2 and z == 2: return 7
        if h == 2 and w == 3 and z == 2: return 11 # Estimate
        if h == 2 and w == 2 and z == 3: return 11 # Estimate
        if h == 2 and w == 3 and z == 3: return 16 # Estimate
        if h == 3 and w == 2 and z == 2: return 14 # 2x2x2 (7) + 1x2x2 (4) + 1x2x2 (4)? No.
        # 3x2 * 2x2. Split rows 1+2.
        # 1x2 * 2x2 (4) + 2x2 * 2x2 (7) = 11.
        return h * w * z # Fallback schoolbook

    best_U = None
    best_V = None
    best_W = None
    best_rank = 72 # Schoolbook

    # Generate candidate splits
    # We will try all combinations of splits_n, splits_m, and a few splits_p
    splits_p_candidates = [
        (2, 2, 2),
        (1, 2, 3),
        (3, 2, 1),
        (1, 1, 4),
        (4, 1, 1),
        (2, 1, 3),
        (3, 1, 2),
        (1, 3, 2),
        (2, 3, 1),
        (1, 2, 1, 2), # 4 blocks?
        (2, 2, 1, 1),
    ]

    for sn in splits_n:
        for sm in splits_m:
            for sp in splits_p_candidates:
                # Calculate theoretical rank
                # We need to map the splits to block indices
                # A blocks: (sn[0], sm[0]), (sn[0], sm[1]), ...
                # This is complex to calculate exactly without building.
                # Let's just build and verify.
                
                if time.time() > deadline - 10:
                    break
                
                # Build decomposition for this split
                U = []
                V = []
                W = []
                
                # We need to define the block boundaries
                a_row_bounds = [0]
                for size in sn: a_row_bounds.append(a_row_bounds[-1] + size)
                a_col_bounds = [0]
                for size in sm: a_col_bounds.append(a_col_bounds[-1] + size)
                b_row_bounds = a_col_bounds # Must match
                b_col_bounds = [0]
                for size in sp: b_col_bounds.append(b_col_bounds[-1] + size)
                
                # Iterate over blocks
                # i_idx, k_idx, j_idx
                for i_idx in range(len(sn)):
                    for k_idx in range(len(sm)):
                        for j_idx in range(len(sp)):
                            h = sn[i_idx]
                            w = sm[k_idx]
                            z = sp[j_idx]
                            
                            r_start = len(U)
                            # Add block decomposition
                            # We need a helper to add a generic block multiplication
                            # This is hard to do generically with the existing _add_ functions.
                            # We will stick to the specific 2x2 Strassen and 1x2 logic.
                            # So we only allow splits that result in blocks of size 1 or 2.
                            if h > 2 or w > 2 or z > 2:
                                continue
                            
                            # Map to specific helper
                            # We need to know the absolute indices
                            i0 = a_row_bounds[i_idx]
                            k0 = a_col_bounds[k_idx]
                            j0 = b_col_bounds[j_idx]
                            
                            # We can only use _add_strassen if h=2, w=2, z=2
                            # _add_1x2_2x2 if h=1, w=2, z=2
                            # etc.
                            # This is getting messy.
                            
                            pass
                
                # Since generic block building is complex, we will only try splits
                # that are compatible with the existing helpers.
                # Helpers support:
                # - 1x2 * 2x2 -> 1x2 (Rank 4)
                # - 2x2 * 2x2 -> 2x2 (Rank 7)
                # So we need blocks to be 1 or 2.
                # sn must be [1,2] or [2,1] (all <= 2)
                # sm must be [1,3] (3>2, skip), [2,2] (ok), [3,1] (3>2, skip)
                # sp must be all <= 2. So [2,2,2], [1,2,1,2], etc.
                
                if any(x > 2 for x in sn) or any(x > 2 for x in sm) or any(x > 2 for x in sp):
                    continue
                
                # Build
                U_list = []
                V_list = []
                W_list = []
                col = 0
                
                # We need to re-implement the add functions to accept absolute indices
                # and append to lists.
                
                def add_block(i0, k0, j0, h, w, z):
                    nonlocal col
                    # Indices
                    # U rows: i0..i0+h, k0..k0+w
                    # V rows: k0..k0+w, j0..j0+z
                    # W rows: j0..j0+z, i0..i0+h
                    
                    if h == 1 and w == 2 and z == 2:
                        # 1x2 * 2x2
                        # A[0,0], A[0,1]
                        # B[0,0], B[0,1], B[1,0], B[1,1]
                        # C[0,0], C[0,1]
                        
                        # M1: A00 B00 -> C00
                        U_list.append([0]*len(U_list[0]) if U_list else [0]*0) # Placeholder
                        # This list approach is inefficient.
                        pass
                
                # Given the complexity, we will just try the current structure
                # but with different splits of B cols if they are 1 or 2.
                # Current: A rows [1,2], A cols [2,2], B cols [2,2,2].
                # Try: A rows [1,2], A cols [2,2], B cols [1,2,1,2] (4 blocks).
                # This increases the number of blocks, potentially increasing rank.
                # We want to DECREASE rank.
                # Merging blocks (e.g. 2+2 -> 4) allows Strassen? No, 4x4 is 49.
                # 2x2 is 7. 4x4 is 49. 7+7+7+7 = 28. 49 > 28.
                # So splitting is better.
                
                pass

    # Since the search space is constrained by the helpers, 
    # and the current 66 is likely optimal for this family,
    # we will stick to the current decomposition but improve the 
    # "drop term" search by using random restarts.
    
    # Re-use the existing 66-rank builder
    RANK = 66
    U = [[0] * RANK for _ in range(n * m)]
    V = [[0] * RANK for _ in range(m * p)]
    W = [[0] * RANK for _ in range(p * n)]
    col = 0

    def _idx_u(i, k): return i * m + k
    def _idx_v(k, j): return k * p + j
    def _idx_w(j, i): return j * n + i

    def _add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = _idx_u(a_r, a_c)
        A1 = _idx_u(a_r, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C0 = _idx_w(c_c, c_r)
        C1 = _idx_w(c_c + 1, c_r)
        U[A0][col] = 1; V[B00][col] = 1; W[C0][col] = 1; col += 1
        U[A1][col] = 1; V[B10][col] = 1; W[C0][col] = 1; col += 1
        U[A0][col] = 1; V[B01][col] = 1; W[C1][col] = 1; col += 1
        U[A1][col] = 1; V[B11][col] = 1; W[C1][col] = 1; col += 1

    def _add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = _idx_u(a_r, a_c); A01 = _idx_u(a_r, a_c + 1)
        A10 = _idx_u(a_r + 1, a_c); A11 = _idx_u(a_r + 1, a_c + 1)
        B00 = _idx_v(b_r, b_c); B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c); B11 = _idx_v(b_r + 1, b_c + 1)
        C00 = _idx_w(c_c, c_r); C01 = _idx_w(c_c + 1, c_r)
        C10 = _idx_w(c_c, c_r + 1); C11 = _idx_w(c_c + 1, c_r + 1)
        U[A00][col] = 1; U[A11][col] = 1; V[B00][col] = 1; V[B11][col] = 1; W[C00][col] = 1; W[C11][col] = 1; col += 1
        U[A10][col] = 1; U[A11][col] = 1; V[B00][col] = 1; W[C10][col] = 1; W[C11][col] = -1; col += 1
        U[A00][col] = 1; V[B01][col] = 1; V[B11][col] = -1; W[C01][col] = 1; W[C11][col] = 1; col += 1
        U[A11][col] = 1; V[B10][col] = 1; V[B00][col] = -1; W[C00][col] = 1; W[C10][col] = 1; col += 1
        U[A00][col] = 1; U[A01][col] = 1; V[B11][col] = 1; W[C00][col] = -1; W[C01][col] = 1; col += 1
        U[A10][col] = 1; U[A00][col] = -1; V[B00][col] = 1; V[B01][col] = 1; W[C11][col] = 1; col += 1
        U[A01][col] = 1; U[A11][col] = -1; V[B10][col] = 1; V[B11][col] = 1; W[C00][col] = 1; col += 1

    for j in range(3):
        bc = 2 * j
        _add_1x2_2x2(0, 0, 0, bc, 0, 2 * j)
        _add_1x2_2x2(0, 2, 2, bc, 0, 2 * j)
        _add_strassen(1, 0, 0, bc, 1, 2 * j)
        _add_strassen(1, 2, 2, bc, 1, 2 * j)

    # Now, perform random restarts on the drop search
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T_expected = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                T_expected[a, b, c] = 1.0

    def _verify_fast(Ua, Va, Wa):
        T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
        return np.allclose(T, T_expected, atol=1e-9)

    best_rank = 66
    best_U, best_V, best_W = U_np, V_np, W_np

    # Try dropping terms
    current_U, current_V, current_W = U_np, V_np, W_np
    current_rank = 66
    
    for _ in range(50): # 50 restarts
        if time.time() > deadline - 5: break
        # Perturb
        if _ > 0:
            current_U = U_np + np.random.normal(0, 0.1, U_np.shape)
            current_V = V_np + np.random.normal(0, 0.1, V_np.shape)
            current_W = W_np + np.random.normal(0, 0.1, W_np.shape)
            # Re-verify? No, perturbation breaks it.
            # We must start from a valid solution.
            # So we only perturb if we can repair? No.
            # Just try dropping from the original.
            current_U, current_V, current_W = U_np, V_np, W_np
            current_rank = 66
            
        # Try dropping
        for r in range(current_rank):
            if time.time() > deadline - 2: break
            success, Ua_new, Va_new, Wa_new = _try_drop(current_U, current_V, current_W, r)
            if success:
                current_U, current_V, current_W = Ua_new, Va_new, Wa_new
                current_rank -= 1
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_U, best_V, best_W = current_U, current_V, current_W
                r = -1
                if time.time() > deadline - 2: break

    # Return best found
    if best_rank < 66:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
        U_out = [[round(x * 2) / 2 for x in row] for row in U_out]
        V_out = [[round(x * 2) / 2 for x in row] for row in V_out]
        W_out = [[round(x * 2) / 2 for x in row] for row in W_out]
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        # Return original 66
        U_out = U_np.tolist()
        V_out = V_np.tolist()
        W_out = W_np.tolist()
        U_out = [[round(x * 2) / 2 for x in row] for row in U_out]
        V_out = [[round(x * 2) / 2 for x in row] for row in V_out]
        W_out = [[round(x * 2) / 2 for x in row] for row in W_out]
        return {"U": U_out, "V": V_out, "W": W_out}
