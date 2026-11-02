import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def min_rank(nc, mc, pc):
        """Compute minimum rank for nc x mc x pc matrix multiplication via DP."""
        if time.time() > deadline:
            return nc * mc * pc
        
        if nc == 1 and mc == 1 and pc == 1:
            return 1
        if nc == 1:
            return mc * pc
        if mc == 1:
            return nc * pc
        if pc == 1:
            return nc * mc
        
        best = nc * mc * pc
        
        # Try all splits
        for i in range(1, nc):
            best = min(best, min_rank(i, mc, pc) + min_rank(nc - i, mc, pc))
        for j in range(1, mc):
            best = min(best, min_rank(nc, j, pc) + min_rank(nc, mc - j, pc))
        for k in range(1, pc):
            best = min(best, min_rank(nc, mc, k) + min_rank(nc, mc, pc - k))
        
        # Strassen 2x2x2
        if nc >= 2 and mc >= 2 and pc >= 2:
            r = 7
            if nc > 2:
                r += min_rank(nc - 2, mc, pc)
            if mc > 2:
                r += min_rank(2, mc - 2, pc)
            if pc > 2:
                r += min_rank(2, 2, pc - 2)
            best = min(best, r)
        
        return best

    def strassen_2x2x2_cols(i0, j0, k0, n_dim, m_dim, p_dim):
        cols = []
        cols.append(({i0*m_dim+j0: 1, (i0+1)*m_dim+(j0+1): 1},
                     {j0*p_dim+k0: 1, (j0+1)*p_dim+(k0+1): 1},
                     {k0*n_dim+i0: 1, (k0+1)*n_dim+(i0+1): 1}))
        cols.append(({(i0+1)*m_dim+j0: 1, (i0+1)*m_dim+(j0+1): 1},
                     {j0*p_dim+k0: 1},
                     {k0*n_dim+(i0+1): 1, (k0+1)*n_dim+(i0+1): -1}))
        cols.append(({i0*m_dim+j0: 1},
                     {j0*p_dim+(k0+1): 1, (j0+1)*p_dim+(k0+1): -1},
                     {(k0+1)*n_dim+i0: 1, (k0+1)*n_dim+(i0+1): 1}))
        cols.append(({(i0+1)*m_dim+(j0+1): 1},
                     {(j0+1)*p_dim+k0: 1, j0*p_dim+k0: -1},
                     {k0*n_dim+i0: 1, k0*n_dim+(i0+1): 1}))
        cols.append(({i0*m_dim+j0: 1, i0*m_dim+(j0+1): 1},
                     {(j0+1)*p_dim+(k0+1): 1},
                     {k0*n_dim+i0: -1, (k0+1)*n_dim+i0: 1}))
        cols.append(({(i0+1)*m_dim+j0: 1, i0*m_dim+j0: -1},
                     {j0*p_dim+k0: 1, j0*p_dim+(k0+1): 1},
                     {(k0+1)*n_dim+(i0+1): 1}))
        cols.append(({i0*m_dim+(j0+1): 1, (i0+1)*m_dim+(j0+1): -1},
                     {(j0+1)*p_dim+k0: 1, (j0+1)*p_dim+(k0+1): 1},
                     {k0*n_dim+i0: 1}))
        return cols

    def build_decomp(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols):
        if time.time() > deadline:
            for di in range(nc):
                for dj in range(mc):
                    for dk in range(pc):
                        cols.append((
                            {(ii+di)*m_dim+(jj+dj): 1.0},
                            {(jj+dj)*p_dim+(kk+dk): 1.0},
                            {(kk+dk)*n_dim+(ii+di): 1.0}
                        ))
            return
        
        if nc == 1 and mc == 1 and pc == 1:
            cols.append((
                {ii*m_dim+jj: 1.0},
                {jj*p_dim+kk: 1.0},
                {kk*n_dim+ii: 1.0}
            ))
            return
        
        if nc == 1:
            for dj in range(mc):
                for dk in range(pc):
                    cols.append((
                        {ii*m_dim+(jj+dj): 1.0},
                        {(jj+dj)*p_dim+(kk+dk): 1.0},
                        {(kk+dk)*n_dim+ii: 1.0}
                    ))
            return
        if mc == 1:
            for di in range(nc):
                for dk in range(pc):
                    cols.append((
                        {(ii+di)*m_dim+jj: 1.0},
                        {jj*p_dim+(kk+dk): 1.0},
                        {(kk+dk)*n_dim+(ii+di): 1.0}
                    ))
            return
        if pc == 1:
            for di in range(nc):
                for dj in range(mc):
                    cols.append((
                        {(ii+di)*m_dim+(jj+dj): 1.0},
                        {(jj+dj)*p_dim+kk: 1.0},
                        {kk*n_dim+(ii+di): 1.0}
                    ))
            return
        
        # Determine best decomposition strategy using min_rank
        best_cost = nc * mc * pc
        best_strategy = None
        
        # Check Strassen
        if nc >= 2 and mc >= 2 and pc >= 2:
            s_cost = 7
            if nc > 2:
                s_cost += min_rank(nc-2, mc, pc)
            if mc > 2:
                s_cost += min_rank(2, mc-2, pc)
            if pc > 2:
                s_cost += min_rank(2, 2, pc-2)
            if s_cost < best_cost:
                best_cost = s_cost
                best_strategy = 'strassen'
        
        # Check splits
        for i in range(1, nc):
            c = min_rank(i, mc, pc) + min_rank(nc-i, mc, pc)
            if c < best_cost:
                best_cost = c
                best_strategy = ('split_n', i)
        for j in range(1, mc):
            c = min_rank(nc, j, pc) + min_rank(nc, mc-j, pc)
            if c < best_cost:
                best_cost = c
                best_strategy = ('split_m', j)
        for k in range(1, pc):
            c = min_rank(nc, mc, k) + min_rank(nc, mc, pc-k)
            if c < best_cost:
                best_cost = c
                best_strategy = ('split_p', k)
        
        if best_strategy == 'strassen':
            cols.extend(strassen_2x2x2_cols(ii, jj, kk, n_dim, m_dim, p_dim))
            if nc > 2:
                build_decomp(nc-2, mc, pc, ii+2, jj, kk, n_dim, m_dim, p_dim, cols)
            if mc > 2:
                build_decomp(2, mc-2, pc, ii, jj+2, kk, n_dim, m_dim, p_dim, cols)
            if pc > 2:
                build_decomp(2, 2, pc-2, ii, jj, kk+2, n_dim, m_dim, p_dim, cols)
        elif isinstance(best_strategy, tuple):
            if best_strategy[0] == 'split_n':
                build_decomp(best_strategy[1], mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
                build_decomp(nc-best_strategy[1], mc, pc, ii+best_strategy[1], jj, kk, n_dim, m_dim, p_dim, cols)
            elif best_strategy[0] == 'split_m':
                build_decomp(nc, best_strategy[1], pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
                build_decomp(nc, mc-best_strategy[1], pc, ii, jj+best_strategy[1], kk, n_dim, m_dim, p_dim, cols)
            elif best_strategy[0] == 'split_p':
                build_decomp(nc, mc, best_strategy[1], ii, jj, kk, n_dim, m_dim, p_dim, cols)
                build_decomp(nc, mc, pc-best_strategy[1], ii, jj, kk+best_strategy[1], n_dim, m_dim, p_dim, cols)
        else:
            # Fallback: schoolbook
            for di in range(nc):
                for dj in range(mc):
                    for dk in range(pc):
                        cols.append((
                            {(ii+di)*m_dim+(jj+dj): 1.0},
                            {(jj+dj)*p_dim+(kk+dk): 1.0},
                            {(kk+dk)*n_dim+(ii+di): 1.0}
                        ))

    cols = []
    build_decomp(n, m, p, 0, 0, 0, n, m, p, cols)
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
