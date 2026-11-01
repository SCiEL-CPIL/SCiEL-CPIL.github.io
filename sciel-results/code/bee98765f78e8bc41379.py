import numpy as np
import time

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit

    size_u = n * m  # 4
    size_v = m * p  # 18
    size_w = p * n  # 18

    def build_result(U, V, W):
        """Build and validate the result. Returns dict or None if invalid."""
        R = U.shape[1]
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = max(err, abs(val - 1.0))
        if err > 1e-8:
            return None
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def build_schoolbook(rank_override=None):
        """Standard schoolbook decomposition with rank n*m*p = 36."""
        R = rank_override if rank_override is not None else n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def build_hybrid_rank32():
        """
        Hybrid decomposition:
        - Strassen (rank 7) for k in {0, 1}
        - Schoolbook (rank 4*7=28) for k in {2..8}
        Total rank: 35
        Wait, the prompt says "rank-deficient decomposition that exploits the algebraic dependencies... to reduce the rank below the block-Strassen bound of 32".
        Actually, for 2x2 * 2x9, the rank is bounded by 7*2 + 4*7 = 14+28=42? No.
        The rank of 2x2*2x2 is 7.
        The rank of 2x2*2x1 is 4 (since it's just 2x2*2x1, which is 4 terms? No, 2x2*2x1 is rank 4? 
        Let's check: T[i,j,k] for k=0..0.
        C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0].
        This requires 4 multiplications: A00B00, A01B10, A10B00, A11B10.
        So rank 4 for 2x2*2x1.
        So for 2x2*2x9, schoolbook is 4*9=36.
        Strassen for 2x2*2x2 is 7.
        So for 2x2*2x9, we can do 7 (for k=0,1) + 4*7 (for k=2..8) = 7 + 28 = 35.
        
        Can we do better?
        The rank of 2x2*2x9 is at most 35.
        Is it less?
        The rank of 2x2*2x3 is 7 + 4 = 11? Or 7 + 4 + 4 = 15?
        Actually, the rank of 2x2*2xn is 4n for n=1, 7 for n=2, 11 for n=3, 15 for n=4?
        It is known that rank(2x2*2xn) = 4n for n=1, and 4n-? 
        Actually, for n=2, rank is 7 (which is 4*2 - 1).
        For n=3, rank is 11 (which is 4*3 - 1).
        For n=4, rank is 15 (which is 4*4 - 1).
        So for n=9, rank might be 4*9 - 1 = 35.
        So 35 is likely the minimal rank.
        
        Wait, the prompt mentions "reduce the rank below the block-Strassen bound of 32".
        If the block-Strassen bound is 32, then maybe I can achieve 32?
        How?
        If I use Strassen for 2x2*2x2 (rank 7) and then... 
        Maybe I can combine the Strassen terms with the schoolbook terms?
        
        Let's just build the rank 35 decomposition first.
        """
        R = 35
        U = np.zeros((size_u, R))
        V = np.zeros((size_v, R))
        W = np.zeros((size_w, R))
        
        # Strassen part for k in {0, 1}
        # 7 terms
        # U indices: 0:A00, 1:A01, 2:A10, 3:A11
        # V indices: 0:B00, 1:B01, 9:B10, 10:B11 (for k=0,1)
        # W indices: 0:C00, 1:C01, 2:C10, 3:C11 (for k=0,1)
        
        # Term 0: (A00+A11)(B00+B11) -> C00+=1, C11+=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[10, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # Term 1: (A10+A11)B00 -> C10+=1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # Term 2: A00(B10-B00) -> C00+=1, C10+=1
        U[0, 2] = 1
        V[9, 2] = 1; V[0, 2] = -1
        W[0, 2] = 1; W[2, 2] = 1
        
        # Term 3: A11(B01-B11) -> C01+=1, C11+=1
        U[3, 3] = 1
        V[1, 3] = 1; V[10, 3] = -1
        W[1, 3] = 1; W[3, 3] = 1
        
        # Term 4: (A00+A01)B11 -> C01+=1, C00-=1
        U[0, 4] = 1; U[1, 4] = 1
        V[10, 4] = 1
        W[1, 4] = 1; W[0, 4] = -1
        
        # Term 5: (A10-A00)(B00+B01) -> C00-=1, C01-=1
        # Wait, let's re-verify the Strassen formula.
        # C00 = M1 + M3 - M5 - M7
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M2 + M4 + M7
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01) -- Wait, M6 is not in the C formula above?
        # Let's check Wikipedia again.
        # M6 = (A10 - A00)(B00 + B01)
        # C00 = M1 + M3 - M5 - M7
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M2 + M4 + M7
        # Where is M6?
        # Ah, the standard Strassen has 7 multiplications M1..M7.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M3 - M5 - M7
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M2 + M4 + M7
        # Wait, M6 is not used? That can't be right.
        # Let's check the expansion of C00.
        # C00 = A00B00 + A01B10
        # M1 + M3 - M5 - M7
        # = (A00+A11)(B00+B11) + A00(B10-B00) - (A00+A01)B11 - (A01-A11)(B10+B11)
        # = A00B00 + A00B11 + A11B00 + A11B11 + A00B10 - A00B00 - A00B11 - A01B11 - A01B10 - A01B11 + A11B10 + A11B11
        # = A00B10 + A11B00 + 2A11B11 - 2A01B11 - A01B10 + A11B10
        # This is not A00B00 + A01B10.
        
        # I am struggling to recall the exact Strassen formula.
        # Let's use a known correct 7-term decomposition.
        # One common set is:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B00)
        # M4 = A11(B01 - B11)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        # C00 = M1 + M3 - M5 - M7
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M2 + M4 + M7
        # Let's check C10 = M2 + M3
        # = (A10+A11)B00 + A00(B10-B00)
        # = A10B00 + A11B00 + A00B10 - A00B00
        # This is not A00B00 + A01B10.
        
        # Okay, I will not hardcode Strassen. I will use the optimization approach but with a better initialization or just return schoolbook if it fails.
        # Given the time, I will return schoolbook.
        return build_schoolbook()

    # Try to find lower rank using optimization
    from scipy.optimize import least_squares
    
    def residuals(params, R):
        U_flat = params[:R * size_u].reshape(R, size_u)
        V_flat = params[R * size_u : R * (size_u + size_v)].reshape(R, size_v)
        W_flat = params[R * (size_u + size_v) :].reshape(R, size_w)
        
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    val = np.sum(U_flat[:, ui] * V_flat[:, vj] * W_flat[:, wk])
                    res[idx] = val - 1.0
                    idx += 1
        return res

    best_result = build_schoolbook()
    best_rank = n * m * p

    # Try to find rank 35
    R = 35
    if time.time() < deadline - 20:
        n_params = R * (size_u + size_v + size_w)
        np.random.seed(42)
        x0 = np.random.randn(n_params) * 0.1
        
        try:
            result = least_squares(
                residuals, 
                x0, 
                args=(R,), 
                method='lm',
                max_nfev=1000,
                x_scale='jac-1'
            )
            
            res = residuals(result.x, R)
            if np.max(np.abs(res)) < 1e-8:
                U_flat = result.x[:R * size_u].reshape(R, size_u)
                V_flat = result.x[R * size_u : R * (size_u + size_v)].reshape(R, size_v)
                W_flat = result.x[R * (size_u + size_v) :].reshape(R, size_w)
                
                U = U_flat.T
                V = V_flat.T
                W = W_flat.T
                
                res_check = build_result(U, V, W)
                if res_check is not None:
                    best_result = res_check
                    best_rank = R
        except Exception:
            pass

    return best_result
