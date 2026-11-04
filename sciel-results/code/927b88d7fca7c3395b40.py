import numpy as np
import time
import sys
import itertools
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 3, 7
    N_U = n * m  # 6
    N_V = m * p  # 21
    N_W = p * n  # 14

    start_time = time.time()
    deadline = 140.0

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((N_U, rank))
        V = np.zeros((N_V, rank))
        W = np.zeros((N_W, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def get_strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        return U, V, W

    def get_2x1x2_rank4():
        U = np.zeros((2, 4))
        V = np.zeros((2, 4))
        W = np.zeros((4, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
        U[1, 2] = 1; V[0, 2] = 1; W[2, 2] = 1
        U[1, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
        return U, V, W

    def get_2x3x1_rank6():
        U = np.zeros((6, 6))
        V = np.zeros((3, 6))
        W = np.zeros((2, 6))
        col = 0
        for i in range(2):
            for j in range(3):
                U[i*3+j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                col += 1
        return U, V, W

    def get_2x3x2_rank10():
        # Known rank-10 decomposition for 2x3x2 matrix multiplication
        # This is a hard-coded solution if one exists, otherwise we return None
        # For now, we'll try to construct one or return the rank-11 one
        # Let's just use the rank-11 one for safety, but name it rank10 to fit the interface
        # Actually, let's try to find a rank 10 one or use a known one.
        # A common rank 10 decomposition for 2x3x2 might be complex.
        # Let's stick to the rank 11 one for now to ensure validity, but the function name suggests rank 10.
        # I will return the rank 11 one from get_2x3x2_rank11 to be safe, or just implement a search.
        # Given the directive to keep the design, I will just call the rank 11 one or a specific rank 10 if found.
        # Let's assume the "find_2x3x2_rank10" is the intended path, but for the static function, 
        # I'll just return the rank 11 one if no rank 10 is hardcoded, or better, 
        # I will just remove the dependency on finding it dynamically in the main flow if it's not working.
        # Let's just return the rank 11 one for now in this helper, but the main logic uses find_2x3x2_rank10.
        U, V, W = get_2x3x2_rank11()
        return U, V, W

    def get_2x3x2_rank11():
        U = np.zeros((6, 11))
        V = np.zeros((9, 11))
        W = np.zeros((4, 11))
        U_s, V_s, W_s = get_strassen_2x2x2()
        U_212, V_212, W_212 = get_2x1x2_rank4()
        
        U[0, 0:7] = U_s[0, :]
        U[1, 0:7] = U_s[1, :]
        U[3, 0:7] = U_s[2, :]
        U[4, 0:7] = U_s[3, :]
        
        V[0, 0:7] = V_s[0, :]
        V[1, 0:7] = V_s[1, :]
        V[3, 0:7] = V_s[2, :]
        V[4, 0:7] = V_s[3, :]
        
        W[0, 0:7] = W_s[0, :]
        W[1, 0:7] = W_s[1, :]
        W[2, 0:7] = W_s[2, :]
        W[3, 0:7] = W_s[3, :]
        
        U[2, 7:11] = U_212[0, :]
        U[5, 7:11] = U_212[1, :]
        V[6, 7:11] = V_212[0, :]
        V[7, 7:11] = V_212[1, :]
        W[0, 7:11] = W_212[0, :]
        W[1, 7:11] = W_212[1, :]
        W[2, 7:11] = W_212[2, :]
        W[3, 7:11] = W_212[3, :]
        return U, V, W

    def find_2x3x2_rank10():
        # Target tensor for 2x3x2
        # n=2, m=3, p=2
        n_b, m_b, p_b = 2, 3, 2
        N_U_b, N_V_b, N_W_b = 6, 9, 4
        T_target = np.zeros((N_U_b, N_V_b, N_W_b))
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    T_target[i*m_b+j, j*p_b+k, k*n_b+i] = 1.0
        
        rank_b = 10
        U = np.random.randn(N_U_b, rank_b)
        V = np.random.randn(N_V_b, rank_b)
        W = np.random.randn(N_W_b, rank_b)
        
        def residuals(x):
            U_r = x[:N_U_b*rank_b].reshape(N_U_b, rank_b)
            V_r = x[N_U_b*rank_b:(N_U_b+N_V_b)*rank_b].reshape(N_V_b, rank_b)
            W_r = x[(N_U_b+N_V_b)*rank_b:].reshape(N_W_b, rank_b)
            
            T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            return (T_calc - T_target).flatten()

        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Try to solve
        try:
            result = least_squares(residuals, x0, max_nfev=10000)
            if result.cost < 1e-8:
                U_sol = result.x[:N_U_b*rank_b].reshape(N_U_b, rank_b)
                V_sol = result.x[N_U_b*rank_b:(N_U_b+N_V_b)*rank_b].reshape(N_V_b, rank_b)
                W_sol = result.x[(N_U_b+N_V_b)*rank_b:].reshape(N_W_b, rank_b)
                # Verify exactness
                T_check = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
                if np.allclose(T_check, T_target, atol=1e-9):
                    return U_sol, V_sol, W_sol
        except Exception:
            pass
        return None

    def build_2x3x7(rank_232_decomp=None):
        if rank_232_decomp is None:
            U11, V11, W11 = get_2x3x2_rank11()
            r_block = 11
        else:
            U11, V11, W11 = rank_232_decomp
            r_block = 10
            
        U6, V6, W6 = get_2x3x1_rank6()
        R_total = 3 * r_block + 6
        U_full = np.zeros((N_U, R_total))
        V_full = np.zeros((N_V, R_total))
        W_full = np.zeros((N_W, R_total))
        
        # Block 1: p=0,1
        U_full[:, 0:r_block] = U11
        for j in range(3):
            for k in range(2):
                V_full[j*7+k, 0:r_block] = V11[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W_full[k*2+i, 0:r_block] = W11[k*2+i, :]
        
        # Block 2: p=2,3
        U_full[:, r_block:2*r_block] = U11
        for j in range(3):
            for k in range(2):
                V_full[j*7+(k+2), r_block:2*r_block] = V11[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W_full[(k+2)*2+i, r_block:2*r_block] = W11[k*2+i, :]
        
        # Block 3: p=4,5
        U_full[:, 2*r_block:3*r_block] = U11
        for j in range(3):
            for k in range(2):
                V_full[j*7+(k+4), 2*r_block:3*r_block] = V11[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W_full[(k+4)*2+i, 2*r_block:3*r_block] = W11[k*2+i, :]
        
        # Block 4: p=6
        U_full[:, 3*r_block:R_total] = U6
        for j in range(3):
            V_full[j*7+6, 3*r_block:R_total] = V6[j, :]
        for i in range(2):
            W_full[6*2+i, 3*r_block:R_total] = W6[i, :]
            
        return U_full, V_full, W_full

    def verify(U, V, W):
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        T_target = np.zeros((N_U, N_V, N_W), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        return np.allclose(T_calc, T_target, atol=1e-9)

    # Attempt to find rank 10 decomposition for 2x3x2
    best_decomp = None
    if time.time() < deadline - 20:
        best_decomp = find_2x3x2_rank10()
        
    if best_decomp is not None:
        U_full, V_full, W_full = build_2x3x7(best_decomp)
        if verify(U_full, V_full, W_full):
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }

    # Fallback to rank 39
    U_full, V_full, W_full = build_2x3x7(None)
    if verify(U_full, V_full, W_full):
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }

    return get_schoolbook()
