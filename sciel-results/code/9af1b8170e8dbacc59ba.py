import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm = n * m  # 6
    mp = m * p  # 24
    pn = p * n  # 16

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Known rank 11 decomposition for 2x3x2
    # Strassen for a=0,1 (7 terms) + Schoolbook for a=2 (4 terms)
    def _block_232_rank11():
        # Returns U (2x2 x 11), V (3x2 x 11), W (2x2 x 11)
        # U[i, a, r], V[a, k, r], W[k, i, r]
        # We will return them flattened as per the interface
        # U_flat[2*a+i, r] ? No, interface is U[i*m+a, r]
        # For block 2x3x2, m_block=3, p_block=2, n_block=2
        # U shape (6, 11), V shape (6, 11), W shape (4, 11)
        
        U = np.zeros((6, 11))
        V = np.zeros((6, 11))
        W = np.zeros((4, 11))
        
        # Strassen 2x2x2 for a=0,1
        # Indices: i in 0,1; a in 0,1; k in 0,1
        S_U = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ])
        S_V = np.array([
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ])
        S_W = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, -1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ])
        
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    val = S_U[r][li*2 + la]
                    if val != 0:
                        U[li*3 + la, r] = val # i*3 + a
            for la in range(2):
                for lk in range(2):
                    val = S_V[r][la*2 + lk]
                    if val != 0:
                        V[la*2 + lk, r] = val # a*2 + k
            for lk in range(2):
                for li in range(2):
                    val = S_W[r][lk*2 + li]
                    if val != 0:
                        W[lk*2 + li, r] = val # k*2 + i
        
        # Schoolbook for a=2
        for i in range(2):
            for k in range(2):
                col = 7 + i*2 + k
                U[i*3 + 2, col] = 1
                V[2*2 + k, col] = 1
                W[k*2 + i, col] = 1
                
        return U, V, W

    # Try to find a lower rank decomposition for 2x3x2
    def _search_232_rank(R):
        # Target tensor T[i, a, k] = 1 for i in 0,1; a in 0,1,2; k in 0,1
        # Flatten T to vector of length 12
        T_flat = np.ones(12)
        
        # Variables: U (6, R), V (6, R), W (4, R)
        # Total params: 6R + 6R + 4R = 16R
        # For R=10, 160 params.
        
        import random
        best_res = float('inf')
        best_params = None
        
        # Use a simple gradient descent or scipy minimize
        # Given time constraints, we'll do a few random restarts with scipy
        
        from scipy.optimize import minimize
        
        def objective(params):
            U = params[:6*R].reshape(6, R)
            V = params[6*R:12*R].reshape(6, R)
            W = params[12*R:].reshape(4, R)
            
            # Compute sum of rank-1 terms
            # T_calc[i, a, k] = sum_r U[i,a,r] V[a,k,r] W[k,i,r]
            T_calc = np.zeros(12)
            for i in range(2):
                for a in range(3):
                    for k in range(2):
                        s = 0
                        for r in range(R):
                            s += U[i*3+a, r] * V[a*2+k, r] * W[k*2+i, r]
                        T_calc[i*6 + a*2 + k] = s
                        # Wait, index mapping for T_flat
                        # i in 0,1; a in 0,1,2; k in 0,1
                        # Index = i*6 + a*2 + k?
                        # i=0, a=0, k=0 -> 0
                        # i=0, a=0, k=1 -> 1
                        # i=0, a=1, k=0 -> 2
                        # ...
                        # i=1, a=2, k=1 -> 1*6 + 2*2 + 1 = 11. Correct.
            
            return np.sum((T_calc - T_flat)**2)
        
        # Try a few random initializations
        for _ in range(10):
            if not _elapsed_ok():
                break
            x0 = np.random.randn(16*R) * 0.5
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000})
            if res.fun < best_res:
                best_res = res.fun
                best_params = res.x
        
        if best_res < 1e-6:
            U = best_params[:6*R].reshape(6, R)
            V = best_params[6*R:12*R].reshape(6, R)
            W = best_params[12*R:].reshape(4, R)
            return U, V, W
        return None

    # Build full decomposition from a block
    def _build_from_block(U_block, V_block, W_block, num_blocks=4):
        R_block = U_block.shape[1]
        R_total = R_block * num_blocks
        
        U = np.zeros((6, R_total))
        V = np.zeros((24, R_total))
        W = np.zeros((16, R_total))
        
        for b in range(num_blocks):
            U[:, b*R_block:(b+1)*R_block] = U_block
            
            # V_block has shape (6, R_block) for a in 0,1,2; k in 0,1
            # We need to map this to V_full[a*8+k, r]
            # k in block b corresponds to global k = b*2 + k_local
            for a in range(3):
                for k_local in range(2):
                    k_global = b*2 + k_local
                    V[a*8 + k_global, b*R_block:(b+1)*R_block] = V_block[a*2 + k_local, :]
            
            # W_block has shape (4, R_block) for k in 0,1; i in 0,1
            # W_full[k*n+i, r] -> W_full[k*2+i, r]
            for k_local in range(2):
                k_global = b*2 + k_local
                for i in range(2):
                    W[k_global*2 + i, b*R_block:(b+1)*R_block] = W_block[k_local*2 + i, :]
        
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # 1. Try to find rank 10 block
    if _elapsed_ok():
        res10 = _search_232_rank(10)
        if res10 is not None:
            candidate = _build_from_block(*res10)
            if _validate(candidate):
                return candidate

    # 2. Try to find rank 9 block
    if _elapsed_ok():
        res9 = _search_232_rank(9)
        if res9 is not None:
            candidate = _build_from_block(*res9)
            if _validate(candidate):
                return candidate

    # 3. Fallback to rank 11 block
    U11, V11, W11 = _block_232_rank11()
    candidate = _build_from_block(U11, V11, W11)
    if _validate(candidate):
        return candidate

    # 4. Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
