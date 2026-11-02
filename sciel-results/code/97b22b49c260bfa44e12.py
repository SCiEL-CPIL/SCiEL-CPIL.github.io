import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    timeout = 110
    
    n, m, p = DIMENSIONS
    
    # Target: Rank 18 via 3 blocks of Rank 6 (2x3x3)
    # Fallback: Rank 54 (Schoolbook)
    
    def find_2x3x3_decomp(max_attempts=20):
        # T[i,j,k] = 1 for all i in 0..1, j in 0..2, k in 0..2
        # We want U (2x6), V (3x6), W (3x6) such that sum_r U[:,r] V[:,r] W[:,r] = T
        # Flatten T to vector of length 18
        T_flat = np.ones(18)
        
        def residual(params):
            # params: 6 * (2+3+3) = 36
            U = params[0:12].reshape(2, 6)
            V = params[12:30].reshape(3, 6)
            W = params[30:36].reshape(3, 6) # Wait, W is 3x6, so 18 elements. 30+18=48.
            # Correct sizes:
            # U: 2*6=12
            # V: 3*6=18
            # W: 3*6=18
            # Total 48.
            # My slice above was wrong.
            
            U = params[0:12].reshape(2, 6)
            V = params[12:30].reshape(3, 6)
            W = params[30:48].reshape(3, 6)
            
            res = np.zeros(18)
            idx = 0
            for i in range(2):
                for j in range(3):
                    for k in range(3):
                        val = 0
                        for r in range(6):
                            val += U[i, r] * V[j, r] * W[k, r]
                        res[idx] = val - 1
                        idx += 1
            return res
            
        best_rank = 54 # Fallback
        best_U = None
        best_V = None
        best_W = None
        
        for attempt in range(max_attempts):
            if time.time() - start_time > timeout - 10:
                break
            
            # Random init
            x0 = np.random.randn(48)
            
            try:
                sol = opt.least_squares(residual, x0, method='lm', max_nfev=1000)
                if sol.cost < 1e-10:
                    U = sol.x[0:12].reshape(2, 6)
                    V = sol.x[12:30].reshape(3, 6)
                    W = sol.x[30:48].reshape(3, 6)
                    
                    # Check for integer/half-integer
                    # Scale by 2 to check integers
                    U2 = U * 2
                    V2 = V * 2
                    W2 = W * 2
                    
                    if np.allclose(U2, np.round(U2), atol=1e-5) and \
                       np.allclose(V2, np.round(V2), atol=1e-5) and \
                       np.allclose(W2, np.round(W2), atol=1e-5):
                       
                        # Round to half-integers
                        U_rounded = np.round(U2) / 2.0
                        V_rounded = np.round(V2) / 2.0
                        W_rounded = np.round(W2) / 2.0
                        
                        # Verify exactness
                        res_check = residual(np.concatenate([
                            U_rounded.flatten(), 
                            V_rounded.flatten(), 
                            W_rounded.flatten()
                        ]))
                        if np.max(np.abs(res_check)) < 1e-9:
                            return U_rounded, V_rounded, W_rounded
            except Exception:
                continue
                
        return None, None, None

    U_sub, V_sub, W_sub = find_2x3x3_decomp()
    
    if U_sub is not None:
        # Construct full rank 18 decomposition
        rank = 18
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # The 2x3x3 decomposition covers one 3x3 block of B and C.
        # We have 3 such blocks.
        # Block 0: B cols 0-2, C cols 0-2
        # Block 1: B cols 3-5, C cols 3-5
        # Block 2: B cols 6-8, C cols 6-8
        
        for block in range(3):
            r_offset = block * 6
            # U is same for all blocks (depends on A rows/cols)
            # A is 2x3. U_sub is 2x6.
            # U in full problem is 6x18.
            # Row i of U_full corresponds to A_{i, j}?
            # In 2x3x3, U is 2x6. Rows correspond to i=0,1.
            # In full 2x3x9, U is (2*3)x18 = 6x18.
            # The input A is flattened to 6.
            # Index in U_full: i*m + j.
            # For a fixed block, the multiplication uses A_{i, 0..2} and B_{0..2, block*3..block*3+2}.
            # Wait, the tensor decomposition for 2x3x3 assumes A is 2x3 and B is 3x3.
            # So U_sub[i, r] corresponds to A_{i, j}?
            # In the 2x3x3 tensor, the indices are i (row of A), j (col of A/row of B), k (col of B).
            # The decomposition is sum_r U_{i,r} V_{j,r} W_{k,r}.
            # Here U depends on i. V depends on j. W depends on k.
            # In the full problem:
            # A is 2x3. Flattened index: a_idx = i*3 + j.
            # B is 3x9. Flattened index: b_idx = j*9 + k.
            # C is 2x9. Flattened index: c_idx = k*2 + i? No, W[k*n+i, r].
            # W index: k*2 + i.
            
            # For block b (0,1,2):
            # k ranges from b*3 to b*3+2.
            # j ranges from 0 to 2.
            # i ranges from 0 to 1.
            
            # U_full[a_idx, r] should be U_sub[i, r_local]
            # a_idx = i*3 + j.
            # But U_sub only depends on i.
            # This implies that for a fixed i, U_sub[i, r] is the same for all j?
            # No, in the tensor T_{ijk}, U is associated with i.
            # So U_sub[i, r] is the coefficient for the input A_{ij}?
            # No, the tensor decomposition is $\sum U_{i,r} V_{j,r} W_{k,r} A_{ij} B_{jk}$.
            # So U_{i,r} multiplies A_{ij}? No, U_{i,r} is part of the term.
            # The term is $U_{i,r} V_{j,r} W_{k,r}$.
            # This term contributes to $C_{ik}$ via $A_{ij} B_{jk}$?
            # Actually, the decomposition is of the *tensor* $M_{ijk}$.
            # $M_{ijk} = \sum_r U_{i,r} V_{j,r} W_{k,r}$.
            # Then $C_{ik} = \sum_{j} A_{ij} B_{jk} M_{ijk}$? No.
            # $C_{ik} = \sum_{j} A_{ij} B_{jk}$.
            # The algorithm computes $C_{ik}$ as $\sum_r P_r Q_r R_r$.
            # $P_r = \sum_i A_{ij} U_{i,r}$?
            
            # Let's stick to the matrix form given in the prompt:
            # U: (n*m, rank) -> 6x18
            # V: (m*p, rank) -> 27x18
            # W: (p*n, rank) -> 18x18
            
            # The decomposition is valid if:
            # For all i,j,k: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 (if we are computing the identity?)
            # No, the tensor entry is 1.
            
            # So for block b:
            # k in [b*3, b*3+2]
            # j in [0, 2]
            # i in [0, 1]
            
            for r_local in range(6):
                r_global = r_offset + r_local
                for i in range(2):
                    for j in range(3):
                        # U index: i*3 + j
                        U[i*3 + j, r_global] = U_sub[i, r_local]
                        
                        # V index: j*9 + k
                        # V_sub depends on j.
                        # V_sub[j, r_local]
                        for k in range(3):
                            k_global = b*3 + k
                            V[j*9 + k_global, r_global] = V_sub[j, r_local]
                            
                            # W index: k_global*2 + i
                            W[k_global*2 + i, r_global] = W_sub[k, r_local]
                            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to Schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
