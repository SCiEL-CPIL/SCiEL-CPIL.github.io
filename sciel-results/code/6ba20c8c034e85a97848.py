"""Matrix multiplication 2x4x8 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 8]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin

    n, m, p = DIMENSIONS
    A_dim = n * m  # 8
    B_dim = m * p  # 32
    C_dim = p * n  # 16

    # Strassen factors for 2x2x2 multiplication.
    # Inputs: A=[a00,a01,a10,a11], B=[b00,b01,b10,b11]
    # Outputs: C=[c00,c01,c10,c11]
    u_strassen = np.array([
        [1, 0, 0, 1],   # M1: a00+a11
        [0, 0, 1, 1],   # M2: a10+a11
        [1, 0, 0, 0],   # M3: a00
        [0, 0, 0, 1],   # M4: a11
        [1, 1, 0, 0],   # M5: a00+a01
        [-1, 0, 1, 0],  # M6: a10-a00
        [0, 1, 0, -1]   # M7: a01-a11
    ], dtype=int)

    v_strassen = np.array([
        [1, 0, 0, 1],   # M1: b00+b11
        [1, 0, 0, 0],   # M2: b00
        [0, 0, 1, -1],  # M3: b10-b11
        [0, 1, -1, 0],  # M4: b01-b10
        [0, 0, 0, 1],   # M5: b11
        [1, 0, 1, 0],   # M6: b00+b10
        [0, 1, 0, 1]    # M7: b01+b11
    ], dtype=int)

    w_strassen = np.array([
        [1, 0, 0, 1],   # M1: c00+c11
        [0, 0, 1, -1],  # M2: c10-c11
        [0, 1, 0, 1],   # M3: c01+c11
        [1, 0, 1, 0],   # M4: c00+c10
        [-1, 1, 0, 0],  # M5: -c00+c01
        [0, 0, 0, 1],   # M6: c11
        [1, 0, 0, 0]    # M7: c00
    ], dtype=int)

    # Build rank-56 Strassen decomposition
    U_global = np.zeros((A_dim, 56), dtype=int)
    V_global = np.zeros((B_dim, 56), dtype=int)
    W_global = np.zeros((C_dim, 56), dtype=int)

    col_offset = 0

    for g in range(4):
        a_l_idx = [0, 1, 4, 5]
        a_r_idx = [2, 3, 6, 7]
        b_top_idx = [2*g, 2*g + 1, 8 + 2*g, 8 + 2*g + 1]
        b_bot_idx = [16 + 2*g, 16 + 2*g + 1, 24 + 2*g, 24 + 2*g + 1]
        c_block_idx = [2*g*2, 2*g*2 + 1, (2*g+1)*2, (2*g+1)*2 + 1]

        for s in range(7):
            col = col_offset + s
            for local_i, global_i in enumerate(a_l_idx):
                U_global[global_i, col] = u_strassen[s, local_i]
            for local_j, global_j in enumerate(b_top_idx):
                V_global[global_j, col] = v_strassen[s, local_j]
            for local_k, global_k in enumerate(c_block_idx):
                W_global[global_k, col] = w_strassen[s, local_k]
        col_offset += 7

        for s in range(7):
            col = col_offset + s
            for local_i, global_i in enumerate(a_r_idx):
                U_global[global_i, col] = u_strassen[s, local_i]
            for local_j, global_j in enumerate(b_bot_idx):
                V_global[global_j, col] = v_strassen[s, local_j]
            for local_k, global_k in enumerate(c_block_idx):
                W_global[global_k, col] = w_strassen[s, local_k]
        col_offset += 7

    # Verify the decomposition
    T = np.zeros((A_dim, B_dim, C_dim), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1

    T_rec = np.einsum('ar,br,cr->abc', U_global.astype(float),
                       V_global.astype(float), W_global.astype(float))

    if not np.allclose(T, T_rec, atol=1e-10):
        # Fallback: schoolbook rank-64
        rank = 64
        U_s = np.zeros((A_dim, rank), dtype=int)
        V_s = np.zeros((B_dim, rank), dtype=int)
        W_s = np.zeros((C_dim, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m + j, col] = 1
                    V_s[j*p + k, col] = 1
                    W_s[k*n + i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    # Try to reduce rank via linear dependence elimination
    # Check if any set of columns is linearly dependent in the tensor sense
    # We look for rank reductions by checking if the tensor can be decomposed
    # with fewer terms using integer/half-integer coefficients

    # First, check for trivial linear dependencies in the column space
    # The key insight: if we can find a linear combination of columns that
    # gives the same tensor, we can reduce rank

    # Try rank 55: check if we can merge two columns
    # Look for pairs of columns (i, j) where there exists alpha such that
    # U_i + alpha*U_j, V_i + alpha*V_j, W_i + alpha*W_j still decomposes
    # This is equivalent to checking if column i - column j (after scaling)
    # gives zero tensor contribution

    # A more systematic approach: compute the tensor from each pair difference
    # and check if it's zero
    for trial_rank in range(55, 50, -1):
        if time.time() > deadline - 5:
            break

        # Try to find 56 - trial_rank columns to eliminate
        # by finding linear dependencies
        # Simple approach: greedily remove columns that are linear combinations
        # of others in the tensor space

        # Build the tensor representation as a 3D array of shape (A_dim, B_dim, C_dim)
        # Each column r contributes U[:,r] * V[:,r] * W[:,r] (outer product)
        # We want to find the minimum subset that spans the tensor

        # Check rank by building the combined tensor and checking if
        # removing a column still gives a valid decomposition
        U_curr = U_global.copy()
        V_curr = V_global.copy()
        W_curr = W_global.copy()

        removed = set()
        for attempt in range(56 - trial_rank):
            if time.time() > deadline - 3:
                break
            # Try removing each remaining column
            found = False
            for r in range(56):
                if r in removed:
                    continue
                # Temporarily remove column r
                remaining = [x for x in range(56) if x != r and x not in removed]
                # Check if the remaining columns still decompose the tensor
                U_rem = U_global[:, remaining]
                V_rem = V_global[:, remaining]
                W_rem = W_global[:, remaining]
                T_test = np.einsum('ar,br,cr->abc', U_rem.astype(float),
                                    V_rem.astype(float), W_rem.astype(float))
                if np.allclose(T, T_test, atol=1e-10):
                    removed.add(r)
                    found = True
                    break
            if not found:
                break

        if len(removed) >= 56 - trial_rank:
            remaining = [x for x in range(56) if x not in removed]
            U_new = U_global[:, remaining]
            V_new = V_global[:, remaining]
            W_new = W_global[:, remaining]
            # Final verification
            T_final = np.einsum('ar,br,cr->abc', U_new.astype(float),
                                 V_new.astype(float), W_new.astype(float))
            if np.allclose(T, T_final, atol=1e-10):
                return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}

    # Try a different structural approach: use the fact that m=4=2*2, p=8=4*2
    # and try a 2x4x4 Strassen-like decomposition combined with a 2x2x2 one
    # This might give rank < 56 if there's a better factorization

    # Alternative: try Winograd-style or other algebraic identities
    # For now, return the best we have
    return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}
# EVOLVE-BLOCK-END
