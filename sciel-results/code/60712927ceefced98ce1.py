"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import scipy.optimize

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T[a, b, c]
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # 1. Construct the known rank-42 Strassen decomposition as a safe fallback
    def build_strassen_rank42():
        rank = 42
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Strassen 7 products for 2x2 x 2x2
        # Indices in U: i*m+j (0..3)
        # Indices in V: j*p+k (0..23)
        # Indices in W: k*n+i (0..23)
        
        # Local indices for 2x2 block:
        # u_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # v_local: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # w_local: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)  <-- Note: W[k*n+i], so k=0->0,1; k=1->2,3? 
        # Wait, W index is k*n+i.
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        # So w_local map: (k,i) -> k*2+i.
        
        strassen_products = [
            # M1 = (a00+a11)*(b00+b11) -> c00+=1, c11+=1
            {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
            # M2 = (a10+a11)*b00 -> c10+=1
            {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1)]},
            # M3 = a00*(b01-b11) -> c01+=1
            {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1)]},
            # M4 = a11*(b10-b00) -> c00+=1, c10+=1
            {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
            # M5 = (a00+a01)*b11 -> c00-=1, c01+=1
            {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
            # M6 = (a10-a00)*(b00+b01) -> c11+=1
            {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
            # M7 = (a01-a11)*(b10+b11) -> c00-=1, c11+=1
            {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, -1), (3, 1)]},
        ]
        
        # Verify local Strassen on 2x2x2
        # We will just trust the standard Strassen construction for the pairs.
        # Partition 12 columns into 6 pairs of 2.
        for t in range(6):
            for s in range(7):
                col = 7 * t + s
                for idx, coeff in strassen_products[s]['U']:
                    U[idx, col] = coeff
                for idx, coeff in strassen_products[s]['V']:
                    j = idx // 2
                    k_local = idx % 2
                    k = 2 * t + k_local
                    V[j * p + k, col] = coeff
                for idx, coeff in strassen_products[s]['W']:
                    k_local = idx // 2
                    i = idx % 2
                    k = 2 * t + k_local
                    W[k * n + i, col] = coeff
        
        # Verify
        T_rec = np.zeros((n * m, m * p, p * n))
        for col in range(rank):
            T_rec += np.einsum('i,j,k->ijk', U[:, col], V[:, col], W[:, col])
        err = np.linalg.norm(T_rec - T)
        return err, U, V, W
    
    err42, U42, V42, W42 = build_strassen_rank42()
    
    if err42 < 1e-9:
        best_rank = 42
        best_U, best_V, best_W = U42, V42, W42
    else:
        # Fallback to schoolbook rank 48
        rank_fb = 48
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_fb[a, col] = 1
                    V_fb[b, col] = 1
                    W_fb[c, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # 2. Attempt to find a lower rank decomposition using a different strategy:
    #    Try to construct a rank-30 decomposition by assuming a block structure of 3 groups of 4 columns.
    #    We will use a numerical optimizer to find the coefficients for a rank-10 2x2x4 block,
    #    then replicate it.
    
    def try_find_rank10_2x2x4():
        """
        Attempt to find a rank-10 decomposition for 2x2 x 2x4.
        Uses scipy.optimize.least_squares on a flattened parameter vector.
        We initialize with a structured guess and try to refine.
        """
        if time.time() > deadline - 20:
            return None
        
        n_sub, m_sub, p_sub = 2, 2, 4
        T_sub = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub))
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    a = i * m_sub + j
                    b = j * p_sub + k
                    c = k * n_sub + i
                    T_sub[a, b, c] = 1.0
        
        rank_sub = 10
        # Parameters: U (4*10), V (8*10), W (8*10)
        # Total params: 40 + 80 + 80 = 200
        
        def objective(params):
            U_flat = params[:40].reshape(4, rank_sub)
            V_flat = params[40:120].reshape(8, rank_sub)
            W_flat = params[120:200].reshape(8, rank_sub)
            
            T_rec = np.zeros_like(T_sub)
            for r in range(rank_sub):
                T_rec += np.einsum('i,j,k->ijk', U_flat[:, r], V_flat[:, r], W_flat[:, r])
            
            residual = T_rec - T_sub
            return residual.flatten()
        
        # Initial guess: Start with Strassen-like patterns but allow optimization.
        # We can try to initialize with a known rank-14 solution (2 Strassens) and try to collapse it.
        # Or just random sparse initialization.
        
        best_err = np.inf
        best_params = None
        
        np.random.seed(42)
        
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            
            # Random sparse initialization
            U_init = np.zeros((4, rank_sub))
            V_init = np.zeros((8, rank_sub))
            W_init = np.zeros((8, rank_sub))
            
            for r in range(rank_sub):
                # Each term touches ~2-4 entries
                nz_u = np.random.randint(1, 4)
                nz_v = np.random.randint(1, 5)
                nz_w = np.random.randint(1, 5)
                
                idx_u = np.random.choice(4, size=nz_u, replace=False)
                idx_v = np.random.choice(8, size=nz_v, replace=False)
                idx_w = np.random.choice(8, size=nz_w, replace=False)
                
                U_init[idx_u, r] = np.random.choice([-1, 1], size=nz_u)
                V_init[idx_v, r] = np.random.choice([-1, 1], size=nz_v)
                W_init[idx_w, r] = np.random.choice([-1, 1], size=nz_w)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = scipy.optimize.least_squares(objective, x0, method='trf', max_nfev=10000, ftol=1e-15, xtol=1e-15, gtol=1e-15)
                err = np.linalg.norm(result.fun)
                
                if err < best_err:
                    best_err = err
                    best_params = result.x.copy()
                    
                    # If we found a very small error, try to snap to integers/half-integers
                    if err < 1e-4:
                        U_snap = result.x[:40].reshape(4, rank_sub)
                        V_snap = result.x[40:120].reshape(8, rank_sub)
                        W_snap = result.x[120:200].reshape(8, rank_sub)
                        
                        # Snap to nearest 0.5
                        U_snap = np.round(U_snap * 2) / 2
                        V_snap = np.round(V_snap * 2) / 2
                        W_snap = np.round(W_snap * 2) / 2
                        
                        # Check if snapped version is still valid
                        T_rec_snap = np.zeros_like(T_sub)
                        for r in range(rank_sub):
                            T_rec_snap += np.einsum('i,j,k->ijk', U_snap[:, r], V_snap[:, r], W_snap[:, r])
                        err_snap = np.linalg.norm(T_rec_snap - T_sub)
                        
                        if err_snap < 1e-9:
                            return U_snap, V_snap, W_snap
                            
            except Exception:
                continue
                
        return None

    # Try to find the rank-10 sub-decomposition
    r10_res = try_find_rank10_2x2x4()
    
    if r10_res is not None:
        U_sub, V_sub, W_sub = r10_res
        rank_total = 30
        U_total = np.zeros((n * m, rank_total))
        V_total = np.zeros((m * p, rank_total))
        W_total = np.zeros((p * n, rank_total))
        
        for g in range(3):
            for r in range(10):
                col = 10 * g + r
                # U is the same for all groups (A doesn't change)
                U_total[:, col] = U_sub[:, r]
                # V: j*p+k where k = 4*g + k_local
                for b_idx in range(8):
                    j = b_idx // 4
                    k_local = b_idx % 4
                    k = 4 * g + k_local
                    V_total[j * p + k, col] = V_sub[b_idx, r]
                # W: k*n+i where k = 4*g + k_local
                for c_idx in range(8):
                    k_local = c_idx // 2
                    i = c_idx % 2
                    k = 4 * g + k_local
                    W_total[k * n + i, col] = W_sub[c_idx, r]
        
        # Verify
        T_rec = np.zeros((n * m, m * p, p * n))
        for col in range(rank_total):
            T_rec += np.einsum('i,j,k->ijk', U_total[:, col], V_total[:, col], W_total[:, col])
        err = np.linalg.norm(T_rec - T)
        
        if err < 1e-9:
            best_rank = 30
            best_U, best_V, best_W = U_total, V_total, W_total
    
    # Final check and return
    if best_rank <= 42:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": U42.tolist(),
            "V": V42.tolist(),
            "W": W42.tolist()
        }
# EVOLVE-BLOCK-END
