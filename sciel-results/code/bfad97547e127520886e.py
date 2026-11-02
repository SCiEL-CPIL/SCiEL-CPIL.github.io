import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0

    def strassen_2x2x2_cols(i0, j0, k0, n_dim, m_dim, p_dim):
        """
        Returns a list of 7 tuples (U_dict, V_dict, W_dict) for Strassen 2x2x2
        applied to the block at top-left (i0, j0, k0).
        U index: i * m_dim + j
        V index: j * p_dim + k
        W index: k * n_dim + i
        """
        cols = []
        
        # M1 = (A11+A22)(B11+B22)
        cols.append((
            {i0*m_dim+j0: 1, (i0+1)*m_dim+(j0+1): 1},
            {j0*p_dim+k0: 1, (j0+1)*p_dim+(k0+1): 1},
            {k0*n_dim+i0: 1, (k0+1)*n_dim+(i0+1): 1}
        ))
        
        # M2 = (A21+A22)B11
        cols.append((
            {(i0+1)*m_dim+j0: 1, (i0+1)*m_dim+(j0+1): 1},
            {j0*p_dim+k0: 1},
            {k0*n_dim+(i0+1): 1, (k0+1)*n_dim+(i0+1): -1}
        ))
        
        # M3 = A11(B12-B22)
        cols.append((
            {i0*m_dim+j0: 1},
            {j0*p_dim+(k0+1): 1, (j0+1)*p_dim+(k0+1): -1},
            {(k0+1)*n_dim+i0: 1, (k0+1)*n_dim+(i0+1): 1}
        ))
        
        # M4 = A22(B21-B11)
        cols.append((
            {(i0+1)*m_dim+(j0+1): 1},
            {(j0+1)*p_dim+k0: 1, j0*p_dim+k0: -1},
            {k0*n_dim+i0: 1, k0*n_dim+(i0+1): 1}
        ))
        
        # M5 = (A11+A12)B22
        cols.append((
            {i0*m_dim+j0: 1, i0*m_dim+(j0+1): 1},
            {(j0+1)*p_dim+(k0+1): 1},
            {k0*n_dim+i0: -1, (k0+1)*n_dim+i0: 1}
        ))
        
        # M6 = (A21-A11)(B11+B12)
        cols.append((
            {(i0+1)*m_dim+j0: 1, i0*m_dim+j0: -1},
            {j0*p_dim+k0: 1, j0*p_dim+(k0+1): 1},
            {(k0+1)*n_dim+(i0+1): 1}
        ))
        
        # M7 = (A12-A22)(B21+B22)
        cols.append((
            {i0*m_dim+(j0+1): 1, (i0+1)*m_dim+(j0+1): -1},
            {(j0+1)*p_dim+k0: 1, (j0+1)*p_dim+(k0+1): 1},
            {k0*n_dim+i0: 1}
        ))
        
        return cols

    def decompose(n_curr, m_curr, p_curr, i0, j0, k0, n_dim, m_dim, p_dim):
        """
        Returns a list of (U_dict, V_dict, W_dict) for the block.
        """
        cols = []
        
        def recurse(nc, mc, pc, ii, jj, kk):
            if time.time() > deadline:
                return
            
            # Base case: 1x1x1
            if nc == 1 and mc == 1 and pc == 1:
                u_idx = ii * m_dim + jj
                v_idx = jj * p_dim + kk
                w_idx = kk * n_dim + ii
                cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))
                return

            # Special case: 1 x m x p (rank m*p)
            # But we can do better if m=1 or p=1?
            # If mc == 1: rank is nc * pc? No, rank 1 if nc=1 or pc=1?
            # For 1 x 1 x p: rank 1.
            # For 1 x m x 1: rank 1.
            # For 1 x 1 x 1: rank 1.
            
            # If any dimension is 1, we can often reduce rank.
            # Specifically, if mc == 1, the tensor is T_{i0k} = 1.
            # This can be decomposed with rank 1.
            if mc == 1:
                u_idx = ii * m_dim + jj
                v_idx = jj * p_dim + kk
                # W depends on k and i
                # We need sum_r U_{i0, r} V_{0k, r} W_{ki, r} = 1
                # Use r=1
                u_dict = {}
                v_dict = {}
                w_dict = {}
                for i in range(ii, ii + nc):
                    for k in range(kk, kk + pc):
                        w_idx = k * n_dim + i
                        u_dict[i * m_dim + jj] = 1.0
                        v_dict[jj * p_dim + k] = 1.0
                        w_dict[w_idx] = 1.0
                cols.append((u_dict, v_dict, w_dict))
                return

            # If nc == 1, rank is mc * pc?
            # T_{0jk} = 1.
            # sum_r U_{0j, r} V_{jk, r} W_{k0, r} = 1.
            # This is equivalent to matrix multiplication of 1xm and mxp.
            # Rank is mc * pc.
            if nc == 1:
                for j in range(jj, jj + mc):
                    for k in range(kk, kk + pc):
                        u_idx = ii * m_dim + j
                        v_idx = j * p_dim + k
                        w_idx = k * n_dim + ii
                        cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))
                return

            # If pc == 1, rank is nc * mc?
            # T_{ij0} = 1.
            if pc == 1:
                for i in range(ii, ii + nc):
                    for j in range(jj, jj + mc):
                        u_idx = i * m_dim + j
                        v_idx = j * p_dim + kk
                        w_idx = kk * n_dim + i
                        cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))
                return

            # Try Strassen 2x2x2
            if nc >= 2 and mc >= 2 and pc >= 2:
                cols.extend(strassen_2x2x2_cols(ii, jj, kk, n_dim, m_dim, p_dim))
                
                # Recurse on remainders
                # 1. (nc-2) x mc x pc at (ii+2, jj, kk)
                if nc > 2:
                    recurse(nc - 2, mc, pc, ii + 2, jj, kk)
                
                # 2. 2 x (mc-2) x pc at (ii, jj+2, kk)
                if mc > 2:
                    recurse(2, mc - 2, pc, ii, jj + 2, kk)
                
                # 3. 2 x 2 x (pc-2) at (ii, jj, kk+2)
                if pc > 2:
                    recurse(2, 2, pc - 2, ii, jj, kk + 2)
                
                return

            # Fallback: split one dimension
            # Prefer splitting m if possible to expose 1s?
            # For 3x4x6, splitting m=4 into 3+1 allows 3x3x6 and 3x1x6.
            # 3x1x6 uses the mc==1 case (rank 1).
            # 3x3x6 splits into 3x3x3 + 3x3x3.
            
            if mc > 1:
                # Split m in half or 3+1?
                # Let's try splitting m into 1 and mc-1 if mc is small?
                # Or just half.
                split_m = mc // 2
                if mc == 3 or mc == 4:
                    # Try 1 + (mc-1)
                    recurse(nc, 1, pc, ii, jj, kk)
                    recurse(nc, mc - 1, pc, ii, jj + 1, kk)
                else:
                    recurse(nc, split_m, pc, ii, jj, kk)
                    recurse(nc, mc - split_m, pc, ii, jj + split_m, kk)
            elif nc > 1:
                recurse(1, mc, pc, ii, jj, kk)
                recurse(nc - 1, mc, pc, ii + 1, jj, kk)
            elif pc > 1:
                recurse(nc, mc, 1, ii, jj, kk)
                recurse(nc, mc, pc - 1, ii, jj, kk)
            else:
                u_idx = ii * m_dim + jj
                v_idx = jj * p_dim + kk
                w_idx = kk * n_dim + ii
                cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))

        recurse(n_curr, m_curr, p_curr, i0, j0, k0)
        return cols

    cols = decompose(n, m, p, 0, 0, 0, n, m, p)
    
    rank = len(cols)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(cols):
        for idx, val in u_dict.items():
            U[idx, r] = val
        for idx, val in v_dict.items():
            V[idx, r] = val
        for idx, val in w_dict.items():
            W[idx, r] = val
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
