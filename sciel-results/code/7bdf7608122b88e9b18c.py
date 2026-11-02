import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 2, 4, 6

    def get_rank_42_solution():
        """Fallback rank-42 solution using Strassen blocks."""
        rank_total = 42
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]

        # Strassen coefficients for 2x2x2
        strassen_U = [
            {0: 1, 3: 1}, {2: 1, 3: 1}, {0: 1}, {3: 1},
            {0: 1, 1: 1}, {0: -1, 2: 1}, {1: 1, 3: -1}
        ]
        strassen_V = [
            {0: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}, {2: 1, 0: -1},
            {3: 1}, {0: 1, 1: 1}, {2: 1, 3: 1}
        ]
        strassen_W = [
            {0: 1, 3: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}, {0: 1, 2: 1},
            {0: -1, 1: 1}, {3: 1}, {0: 1}
        ]

        col_offset = 0
        for j_block in range(3):
            for side in ['L', 'R']:
                if side == 'L':
                    u_col_offset = 0
                    v_row_offset = 0
                else:
                    u_col_offset = 2
                    v_row_offset = 2

                for prod_idx in range(7):
                    col = col_offset + prod_idx

                    for local_idx, sign in strassen_U[prod_idx].items():
                        if local_idx == 0: u_idx = u_col_offset
                        elif local_idx == 1: u_idx = u_col_offset + 1
                        elif local_idx == 2: u_idx = 4 + u_col_offset
                        else: u_idx = 4 + u_col_offset + 1
                        U[u_idx][col] += sign

                    for local_idx, sign in strassen_V[prod_idx].items():
                        if local_idx == 0: v_idx = v_row_offset * 6 + 2 * j_block
                        elif local_idx == 1: v_idx = v_row_offset * 6 + 2 * j_block + 1
                        elif local_idx == 2: v_idx = (v_row_offset + 1) * 6 + 2 * j_block
                        else: v_idx = (v_row_offset + 1) * 6 + 2 * j_block + 1
                        V[v_idx][col] += sign

                    for local_idx, sign in strassen_W[prod_idx].items():
                        if local_idx == 0: w_idx = 2 * (2 * j_block)
                        elif local_idx == 1: w_idx = 2 * (2 * j_block + 1)
                        elif local_idx == 2: w_idx = 2 * (2 * j_block) + 1
                        else: w_idx = 2 * (2 * j_block + 1) + 1
                        W[w_idx][col] += sign

                col_offset += 7

        return {"U": U, "V": V, "W": W}

    def build_tensor_2x4x2():
        """Build the tensor for 2x4 * 4x2 matrix multiplication."""
        T = np.zeros((8, 8, 4))
        for i in range(2):
            for j in range(4):
                for k in range(2):
                    a_idx = i * 4 + j
                    b_idx = j * 2 + k
                    c_idx = k * 2 + i
                    T[a_idx, b_idx, c_idx] = 1.0
        return T

    def find_decomposition(rank_target, max_seeds=30):
        """Find a decomposition of the 2x4x2 tensor at the given rank."""
        T = build_tensor_2x4x2()
        
        def unpack(params):
            U = params[:8*rank_target].reshape(8, rank_target)
            V = params[8*rank_target:8*rank_target+8*rank_target].reshape(8, rank_target)
            W = params[8*rank_target+8*rank_target:].reshape(4, rank_target)
            return U, V, W

        def loss(params):
            U, V, W = unpack(params)
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((T_recon - T)**2)
            return err

        def try_decomposition(U0, V0, W0, max_iter=2000):
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            # First pass: BFGS
            res = minimize(loss, x0, method='BFGS', 
                          options={'maxiter': max_iter, 'gtol': 1e-10})
            
            if res.fun > 1e-8:
                return None
            
            U, V, W = unpack(res.x)
            
            # Second pass: L-BFGS-B with tighter tolerance
            res2 = minimize(loss, res.x, method='L-BFGS-B',
                           options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-12})
            if res2.fun < res.fun:
                U, V, W = unpack(res2.x)
            
            # Try snapping to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
            if np.allclose(T_recon, T, atol=1e-9):
                return U_snap, V_snap, W_snap
            
            # Try snapping to integers
            U_snap_i = np.round(U)
            V_snap_i = np.round(V)
            W_snap_i = np.round(W)
            T_recon_i = np.einsum('ar,br,cr->abc', U_snap_i, V_snap_i, W_snap_i)
            if np.allclose(T_recon_i, T, atol=1e-9):
                return U_snap_i, V_snap_i, W_snap_i
            
            return None

        # Try multiple seeds with different initialization strategies
        for seed in range(max_seeds):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(seed * 1000 + 42)
            
            if seed < 10:
                # Small random initialization
                scale = 0.1
                U0 = np.random.randn(8, rank_target) * scale
                V0 = np.random.randn(8, rank_target) * scale
                W0 = np.random.randn(4, rank_target) * scale
            elif seed < 20:
                # Sparse initialization - only a few non-zero entries
                U0 = np.zeros((8, rank_target))
                V0 = np.zeros((8, rank_target))
                W0 = np.zeros((4, rank_target))
                for r in range(rank_target):
                    # Randomly select a few positions
                    u_rows = np.random.choice(8, size=2, replace=False)
                    v_rows = np.random.choice(8, size=2, replace=False)
                    w_rows = np.random.choice(4, size=1, replace=False)
                    U0[u_rows, r] = np.random.choice([-1, 1], size=2)
                    V0[v_rows, r] = np.random.choice([-1, 1], size=2)
                    W0[w_rows, r] = np.random.choice([-1, 1], size=1)
            else:
                # Strassen-inspired initialization
                U0 = np.zeros((8, rank_target))
                V0 = np.zeros((8, rank_target))
                W0 = np.zeros((4, rank_target))
                # Use Strassen-like patterns
                for r in range(rank_target):
                    base = r % 7
                    if base == 0:
                        U0[0, r] = 1; U0[3, r] = 1
                        V0[0, r] = 1; V0[3, r] = 1
                        W0[0, r] = 1; W0[3, r] = 1
                    elif base == 1:
                        U0[2, r] = 1; U0[3, r] = 1
                        V0[0, r] = 1
                        W0[2, r] = 1; W0[3, r] = -1
                    elif base == 2:
                        U0[0, r] = 1
                        V0[1, r] = 1; V0[3, r] = -1
                        W0[1, r] = 1; W0[3, r] = 1
                    elif base == 3:
                        U0[3, r] = 1
                        V0[2, r] = 1; V0[0, r] = -1
                        W0[2, r] = 1; W0[0, r] = 1
                    elif base == 4:
                        U0[0, r] = 1; U0[1, r] = 1
                        V0[3, r] = 1
                        W0[0, r] = -1; W0[1, r] = 1
                    elif base == 5:
                        U0[2, r] = 1; U0[0, r] = -1
                        V0[0, r] = 1; V0[1, r] = 1
                        W0[3, r] = 1
                    elif base == 6:
                        U0[1, r] = 1; U0[3, r] = -1
                        V0[2, r] = 1; V0[3, r] = 1
                        W0[0, r] = 1
            
            result = try_decomposition(U0, V0, W0)
            if result is not None:
                return result
        
        return None

    def expand_decomposition(U_sub, V_sub, W_sub):
        """Expand a 2x4x2 decomposition to a full 2x4x6 decomposition."""
        rank_sub = U_sub.shape[1]
        total_rank = 3 * rank_sub
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        for col_pair in range(3):
            col_offset = col_pair * rank_sub
            
            for r in range(rank_sub):
                # U: rows 0-3 correspond to A[i,j] for i in {0,1}, j in {0,1,2,3}
                for i in range(2):
                    for j in range(4):
                        u_idx_full = i * 4 + j
                        # Map from 2x4x2 tensor indices
                        # In the sub-tensor, A index is i*4+j, but we only have 4 columns
                        # So j maps to j % 2 for the local 2x2 block
                        local_j = j % 2
                        local_i = i
                        U[u_idx_full][col_offset + r] += U_sub[local_i*2+local_j, r]
                
                # V: rows correspond to B[j,k] for j in {0,1,2,3}, k in {0,1}
                for j in range(4):
                    for k in range(2):
                        v_idx_full = j * 6 + (2 * col_pair + k)
                        local_j = j
                        local_k = k
                        V[v_idx_full][col_offset + r] += V_sub[local_j*2+local_k, r]
                
                # W: rows correspond to C[i,k] for i in {0,1}, k in {0,1}
                for i in range(2):
                    for k in range(2):
                        w_idx_full = (2 * col_pair + k) * 2 + i
                        local_i = i
                        local_k = k
                        W[w_idx_full][col_offset + r] += W_sub[local_i*2+local_k, r]
        
        return {"U": U, "V": V, "W": W}

    # Try to find a lower-rank decomposition
    # The 2x4x2 tensor has rank at most 14 (sum of two 2x2x2 Strassen)
    # Try ranks from 11 to 14
    for rank_target in [11, 12, 13, 14]:
        if time.time() > deadline - 30:
            break
        
        decomp = find_decomposition(rank_target, max_seeds=20)
        if decomp is not None:
            U_sub, V_sub, W_sub = decomp
            result = expand_decomposition(U_sub, V_sub, W_sub)
            return result

    # Fallback to rank-42 solution
    return get_rank_42_solution()
