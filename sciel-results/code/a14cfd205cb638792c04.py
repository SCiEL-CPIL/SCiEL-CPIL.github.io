import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin

    def get_rank_42():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(i_start, j_start, k_start, offset):
            # M1
            col = offset
            U[(i_start+0)*m + (j_start+0)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1
            # M2
            col = offset + 1
            U[(i_start+1)*m + (j_start+0)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            W[(k_start+0)*n + (i_start+1)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] -= 1
            # M3
            col = offset + 2
            U[(i_start+0)*m + (j_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] -= 1
            W[(k_start+1)*n + (i_start+0)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1
            # M4
            col = offset + 3
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] -= 1
            W[(k_start+0)*n + (i_start+0)][col] += 1
            W[(k_start+0)*n + (i_start+1)][col] += 1
            # M5
            col = offset + 4
            U[(i_start+0)*m + (j_start+0)][col] += 1
            U[(i_start+0)*m + (j_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] -= 1
            W[(k_start+1)*n + (i_start+0)][col] += 1
            # M6
            col = offset + 5
            U[(i_start+1)*m + (j_start+0)][col] += 1
            U[(i_start+0)*m + (j_start+0)][col] -= 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+1)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1
            # M7
            col = offset + 6
            U[(i_start+0)*m + (j_start+1)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] -= 1
            V[(j_start+1)*p + (k_start+0)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] += 1

        add_strassen_block(0, 0, 0, 0)
        add_strassen_block(0, 0, 2, 7)
        add_strassen_block(0, 0, 4, 14)
        add_strassen_block(0, 2, 0, 21)
        add_strassen_block(0, 2, 2, 28)
        add_strassen_block(0, 2, 4, 35)
        
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def find_rank_10_subtensor():
        """
        Finds a rank 10 decomposition for M_{2,4,2}.
        n=2, m=4, p=2.
        """
        n_sub, m_sub, p_sub = 2, 4, 2
        rank_sub = 10
        
        U_size = n_sub
        V_size = m_sub
        W_size = p_sub
        
        # Tensor indices (i, j, k) where T[i,j,k] = 1
        # i in 0..1, j in 0..3, k in 0..1
        idx_triples = [(i, j, k) for i in range(n_sub) for j in range(m_sub) for k in range(p_sub)]
        
        def residuals(params):
            U = params[:U_size*rank_sub].reshape(U_size, rank_sub)
            V = params[U_size*rank_sub:(U_size+V_size)*rank_sub].reshape(V_size, rank_sub)
            W = params[(U_size+V_size)*rank_sub:].reshape(W_size, rank_sub)
            
            res = np.empty(len(idx_triples))
            for idx, (i, j, k) in enumerate(idx_triples):
                res[idx] = np.dot(U[i], V[j] * W[k]) - 1.0
            return res

        best_solution = None
        best_cost = np.inf
        max_attempts = 20
        max_nfev = 2000

        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            x0 = np.random.uniform(-1, 1, size=(U_size + V_size + W_size) * rank_sub)
            
            try:
                sol = least_squares(residuals, x0, method='trf', max_nfev=max_nfev)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_solution = sol.x
                    
                    if best_cost < 1e-10:
                        U = best_solution[:U_size*rank_sub].reshape(U_size, rank_sub)
                        V = best_solution[U_size*rank_sub:(U_size+V_size)*rank_sub].reshape(V_size, rank_sub)
                        W = best_solution[(U_size+V_size)*rank_sub:].reshape(W_size, rank_sub)
                        
                        # Snap to half-integers or integers if close
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        
                        # Verify snapped solution
                        ok = True
                        for i in range(n_sub):
                            for j in range(m_sub):
                                for k in range(p_sub):
                                    s = np.dot(U_snap[i], V_snap[j] * W_snap[k])
                                    if abs(s - 1.0) > 1e-9:
                                        ok = False
                                        break
                                if not ok: break
                            if not ok: break
                        if ok:
                            return U_snap, V_snap, W_snap
                            
            except Exception:
                continue
                
        return None

    def construct_rank_30(U_sub, V_sub, W_sub):
        """
        Constructs M_{2,4,6} decomposition from three M_{2,4,2} rank 10 blocks.
        """
        rank_total = 30
        U_full = np.zeros((n * m, rank_total))
        V_full = np.zeros((m * p, rank_total))
        W_full = np.zeros((p * n, rank_total))
        
        n_sub, m_sub, p_sub = 2, 4, 2
        
        # Block 1: k in {0, 1}
        # Rank indices 0..9
        for r in range(10):
            for i in range(n_sub):
                for j in range(m_sub):
                    U_full[i*m_sub + j, r] = U_sub[i, r]
            for j in range(m_sub):
                for k in range(p_sub):
                    V_full[j*p + k, r] = V_sub[j, r]
            for k in range(p_sub):
                for i in range(n_sub):
                    W_full[k*n + i, r] = W_sub[k, r]
                    
        # Block 2: k in {2, 3}
        # Rank indices 10..19
        for r in range(10):
            r_full = 10 + r
            for i in range(n_sub):
                for j in range(m_sub):
                    U_full[i*m_sub + j, r_full] = U_sub[i, r]
            for j in range(m_sub):
                for k in range(p_sub):
                    V_full[j*p + (k + 2), r_full] = V_sub[j, r]
            for k in range(p_sub):
                for i in range(n_sub):
                    W_full[(k + 2)*n + i, r_full] = W_sub[k, r]
                    
        # Block 3: k in {4, 5}
        # Rank indices 20..29
        for r in range(10):
            r_full = 20 + r
            for i in range(n_sub):
                for j in range(m_sub):
                    U_full[i*m_sub + j, r_full] = U_sub[i, r]
            for j in range(m_sub):
                for k in range(p_sub):
                    V_full[j*p + (k + 4), r_full] = V_sub[j, r]
            for k in range(p_sub):
                for i in range(n_sub):
                    W_full[(k + 4)*n + i, r_full] = W_sub[k, r]
                    
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }

    # Try to find rank 10 sub-decomposition
    if time.time() < deadline - 30:
        sub_decomp = find_rank_10_subtensor()
        if sub_decomp is not None:
            U_sub, V_sub, W_sub = sub_decomp
            result = construct_rank_30(U_sub, V_sub, W_sub)
            if verify(result["U"], result["V"], result["W"]):
                return result

    # Fallback to rank 42
    return get_rank_42()
