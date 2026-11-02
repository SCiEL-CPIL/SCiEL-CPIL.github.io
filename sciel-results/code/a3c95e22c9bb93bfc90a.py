import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    
    def verify_solution(U, V, W):
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-9): return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-9): return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-9): return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[u, r] * V[v, r] * W[w, r]
                    if not np.isclose(s, 1.0, atol=1e-9):
                        return False
        return True

    def build_strassen_2x2x2():
        # Strassen's algorithm for 2x2 * 2x2
        # Inputs: A (2x2), B (2x2)
        # A = [[a11, a12], [a21, a22]]
        # B = [[b11, b12], [b21, b22]]
        
        # M1 = (a11 + a22)(b11 + b22)
        # M2 = (a21 + a22)b11
        # M3 = a11(b12 - b22)
        # M4 = a22(b21 - b11)
        # M5 = (a11 + a12)b22
        # M6 = (a21 - a11)(b11 + b12)
        # M7 = (a12 - a22)(b21 + b22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map these to the tensor indices for the 2x2x2 subproblem.
        # Let's define the local indices for the 2x2x2 subproblem.
        # Local i, j, k in {0, 1}
        # U_local[i*2+j, r]
        # V_local[j*2+k, r]
        # W_local[k*2+i, r]
        
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # M1: (a11+a22)(b11+b22)
        # U: a11 (0,0) -> 1, a22 (1,1) -> 1
        # V: b11 (0,0) -> 1, b22 (1,1) -> 1
        # W: C11 gets +1, C22 gets +1
        # W indices: k*n+i. 
        # C11 is i=0, k=0 -> w=0. C22 is i=1, k=1 -> w=3.
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1
        
        # M2: (a21+a22)b11
        # U: a21 (1,0) -> 1, a22 (1,1) -> 1
        # V: b11 (0,0) -> 1
        # W: C21 gets +1. i=1, k=0 -> w=1.
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[1, 1] = 1
        
        # M3: a11(b12-b22)
        # U: a11 (0,0) -> 1
        # V: b12 (0,1) -> 1, b22 (1,1) -> -1
        # W: C12 gets +1. i=0, k=1 -> w=2.
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        W_s[2, 2] = 1
        
        # M4: a22(b21-b11)
        # U: a22 (1,1) -> 1
        # V: b21 (1,0) -> 1, b11 (0,0) -> -1
        # W: C11 gets -1 (wait, C11 = M1+M4-M5+M7, so M4 is +1 in formula? No, C11=M1+M4... yes +1).
        # Wait, let's re-verify Strassen formulas.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # Yes, M4 is +1 in C11 and +1 in C21.
        # C21 is i=1, k=0 -> w=1.
        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = 1; W_s[1, 3] = 1
        
        # M5: (a11+a12)b22
        # U: a11 (0,0) -> 1, a12 (0,1) -> 1
        # V: b22 (1,1) -> 1
        # W: C12 gets +1 (w=2), C11 gets -1 (w=0).
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[2, 4] = 1; W_s[0, 4] = -1
        
        # M6: (a21-a11)(b11+b12)
        # U: a21 (1,0) -> 1, a11 (0,0) -> -1
        # V: b11 (0,0) -> 1, b12 (0,1) -> 1
        # W: C22 gets +1 (w=3).
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[3, 5] = 1
        
        # M7: (a12-a22)(b21+b22)
        # U: a12 (0,1) -> 1, a22 (1,1) -> -1
        # V: b21 (1,0) -> 1, b22 (1,1) -> 1
        # W: C11 gets +1 (w=0).
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1
        
        return U_s, V_s, W_s

    def build_1x3x3():
        # 1x3 * 3x3
        # A is 1x3: [a1, a2, a3]
        # B is 3x3:
        # [[b11, b12, b13],
        #  [b21, b22, b23],
        #  [b31, b32, b33]]
        # C is 1x3: [c1, c2, c3]
        # c1 = a1*b11 + a2*b21 + a3*b31
        # c2 = a1*b12 + a2*b22 + a3*b32
        # c3 = a1*b13 + a2*b23 + a3*b33
        
        # Rank 3 decomposition (schoolbook for this slice)
        U_133 = np.zeros((3, 3)) # n=1, m=3 -> 1*3=3 rows
        V_133 = np.zeros((9, 3)) # m=3, p=3 -> 3*3=9 rows
        W_133 = np.zeros((3, 3)) # p=3, n=1 -> 3*1=3 rows
        
        # Term 1: a1 * col1 of B
        # U: a1 (0,0) -> 1
        # V: b11 (0,0), b21 (1,0), b31 (2,0) -> 1
        # W: c1 (0,0) -> 1
        U_133[0, 0] = 1
        V_133[0, 0] = 1
        V_133[3, 0] = 1
        V_133[6, 0] = 1
        W_133[0, 0] = 1
        
        # Term 2: a2 * col2 of B
        # U: a2 (0,1) -> 1
        # V: b12 (0,1), b22 (1,1), b32 (2,1) -> 1
        # W: c2 (1,0) -> 1
        U_133[1, 1] = 1
        V_133[1, 1] = 1
        V_133[4, 1] = 1
        V_133[7, 1] = 1
        W_133[1, 1] = 1
        
        # Term 3: a3 * col3 of B
        # U: a3 (0,2) -> 1
        # V: b13 (0,2), b23 (1,2), b33 (2,2) -> 1
        # W: c3 (2,0) -> 1
        U_133[2, 2] = 1
        V_133[2, 2] = 1
        V_133[5, 2] = 1
        V_133[8, 2] = 1
        W_133[2, 2] = 1
        
        return U_133, V_133, W_133

    # Construct the combined rank-10 solution
    # The full problem is 2x3 * 3x3
    # We split the first dimension (n=2) into two parts? No, we split the tensor.
    # The tensor is T[i, j, k].
    # We can view the 2x3x3 tensor as a direct sum of two tensors?
    # Actually, the "sum of powers" or block decomposition usually applies when dimensions allow it.
    # Here, we can split the output rows or input columns.
    
    # Let's try splitting the first dimension of A (n=2) into two 1x3 blocks?
    # If we split A into A1 (1x3) and A2 (1x3), then C = A1*B + A2*B.
    # This is not a direct sum of tensors in the decomposition sense unless we treat them as separate multiplications.
    # However, matrix multiplication is linear.
    # T = T1 + T2 where T1 corresponds to i=0 and T2 to i=1?
    # T1[i=0, j, k] = A[0, j] B[j, k]
    # T2[i=1, j, k] = A[1, j] B[j, k]
    # These are two 1x3x3 tensors.
    # Rank of 1x3x3 is 3.
    # So Rank(T) <= Rank(T1) + Rank(T2) = 3 + 3 = 6?
    # Wait, is the rank of 1x3x3 really 3?
    # Yes, it's just the dot product of a row and columns. It's a sum of 3 rank-1 tensors.
    # Can it be lower?
    # Lower bound for 1xm x mxp is max(m, p) if n=1?
    # Actually, for 1xm x mxp, the tensor is sum_{j} e_j^A \otimes e_j^B \otimes e_j^C? No.
    # T[i,j,k] = delta_{i,0} A_{0j} B_{jk}.
    # This is the tensor for multiplying a 1x3 matrix by a 3x3 matrix.
    # This is equivalent to 3 independent dot products of length 3? No.
    # C_k = sum_j A_j B_jk.
    # This is a tensor of size 3 x 9 x 3.
    # The rank is 3. (Since it's essentially 3 parallel 1x3x3? No, it's one 1x3x3).
    # Wait, 1x3 * 3x3 is a single matrix multiplication.
    # The tensor is sum_{j=1}^3 e_j^{(A)} \otimes (column j of B) \otimes e_j^{(C)}? No.
    # T_{j, (j', k), k'} = delta_{j, j'} delta_{k, k'}? No.
    # T_{j, j'k, k'} = 1 if j=j' and k=k'? No, that's identity.
    # T_{j, j'k, k'} = 1 if j=j' and k=k' is for 1x1x1.
    # For 1xm x mxp:
    # T_{j, j'k, k'} = 1 if j=j' and k=k' is wrong.
    # T_{j, j'k, k'} = 1 if j=j' and k=k' ... wait.
    # C_k = sum_j A_j B_jk.
    # T_{j, j'k, k'} = 1 if j=j' and k=k'?
    # If j=j', then A_j B_jk.
    # If we sum over j, we get C_k.
    # So the tensor is sum_{j=1}^m sum_{k=1}^p e_j^{(A)} \otimes e_{jk}^{(B)} \otimes e_k^{(C)}.
    # This is a sum of m*p rank-1 tensors.
    # So the rank is m*p = 3*3 = 9?
    # No, we can share the A_j part?
    # No, the rank-1 term is e_j \otimes e_{jk} \otimes e_k.
    # We cannot combine e_{11} \otimes e_1 and e_{12} \otimes e_2 because the B indices are different.
    # So the rank of 1x3x3 is 9?
    # Let's check the lower bound.
    # For 1xm x mxp, the rank is m*p.
    # So T1 (i=0) has rank 9. T2 (i=1) has rank 9.
    # Total rank 18. This is just schoolbook.
    
    # So the "direct sum" of two 1x3x3 tensors gives rank 18.
    # The Strassen idea was to use 2x2x2 (rank 7) + 1x3x3 (rank ?).
    # But 2x2x2 + 1x3x3 doesn't cover 2x3x3 directly.
    
    # Let's stick to the schoolbook rank 18 for now, as finding a lower rank is the hard part.
    # The current program uses ALS to try to find lower ranks.
    # I will restore the ALS logic but ensure it runs efficiently.
    
    # Actually, the directive says "Use a structured algebraic construction... verifying that the resulting rank-10 decomposition satisfies the tensor identity".
    # This implies there IS a rank-10 construction.
    # Let's try to find it or verify if it's possible.
    # If I can't find it, I should fall back to the best known (18) or the ALS result.
    
    # Let's re-read the inspiration. "decompose the 2x3x3 tensor into a direct sum of a 2x2x2 tensor (rank 7 via Strassen) and a 1x3x3 tensor (rank 3)".
    # If 1x3x3 has rank 3, then total rank is 10.
    # Why would 1x3x3 have rank 3?
    # Maybe the 1x3x3 tensor here is not the full multiplication, but a specific slice?
    # Or maybe my rank calculation for 1x3x3 is wrong.
    # Let's check the rank of 1xm x mxp.
    # It is known that the rank of 1xm x mxp is m*p.
    # So rank(1x3x3) = 9.
    # So 7 + 9 = 16.
    # This is better than 18!
    
    # Let's try to construct rank 16.
    # Split 2x3x3 into 2x2x3 and 1x3x3? No.
    # Split into 2x2x2 (part of it) and the rest?
    
    # Let's try splitting the columns of B.
    # B is 3x3. Split into B1 (3x2) and B2 (3x1).
    # A is 2x3.
    # C = A * B1 + A * B2.
    # A * B1 is 2x2. Rank of 2x3x2?
    # A * B2 is 2x1. Rank of 2x3x1?
    
    # Rank of 2x3x1:
    # C_i = sum_j A_ij B_j1.
    # This is 2 independent dot products of length 3.
    # Rank is 3? Or 2*3=6?
    # T_{ij, j'k, k'i} ...
    # For 2x3x1:
    # T_{i, j, (j, k=0), (k=0, i)}
    # T_{ij, jk, ki} = 1 if j=j' and k=0 and i=i'?
    # C_i = sum_j A_{ij} B_{j0}.
    # Tensor is sum_{i,j} e_{ij}^A \otimes e_{j0}^B \otimes e_{0i}^C.
    # This is a sum of 6 rank-1 tensors.
    # Can we do better?
    # Lower bound max(n*m, m*p, n*p) = max(6, 3, 2) = 6.
    # So rank is 6.
    
    # Rank of 2x3x2:
    # Lower bound max(6, 6, 4) = 6.
    # Is it 6?
    # If we can find rank 6 for 2x3x2, then total rank is 6 + 6 = 12.
    
    # Let's try to find a rank 6 decomposition for 2x3x2.
    # Or maybe 2x2x2 (rank 7) is not optimal for 2x3x2.
    
    # Let's just run the ALS from the original program, which is a general search.
    # I will keep the original program's logic as it is a valid search strategy.
    # The "structured" part might be a red herring or I'm missing the specific construction.
    # Given the time limit, I will revert to the robust ALS search.

    # Re-inserting the original code logic for safety.
    
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    U_indices = []
    for i in range(n):
        for j in range(m):
            u = i * m + j
            eqs = []
            for k in range(p):
                v = j * p + k
                w = k * n + i
                eqs.append((v, w))
            U_indices.append((i, j, u, eqs))
            
    V_indices = []
    for j in range(m):
        for k in range(p):
            v = j * p + k
            eqs = []
            for i in range(n):
                u = i * m + j
                w = k * n + i
                eqs.append((u, w))
            V_indices.append((j, k, v, eqs))
            
    W_indices = []
    for k in range(p):
        for i in range(n):
            w = k * n + i
            eqs = []
            for j in range(m):
                u = i * m + j
                v = j * p + k
                eqs.append((u, v))
            W_indices.append((k, i, w, eqs))

    def als_decompose(U_init, V_init, W_init, R, max_iters=200, tol=1e-10):
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        prev_err = np.inf
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            for i, j, u, eqs in U_indices:
                B = np.zeros((len(eqs), R))
                for idx, (v, w) in enumerate(eqs):
                    B[idx, :] = V[v, :] * W[w, :]
                try:
                    U_u, _, _, _ = np.linalg.lstsq(B, np.ones(len(eqs)), rcond=None)
                    U[u, :] = U_u
                except:
                    pass
                    
            for j, k, v, eqs in V_indices:
                C = np.zeros((len(eqs), R))
                for idx, (u, w) in enumerate(eqs):
                    C[idx, :] = U[u, :] * W[w, :]
                try:
                    V_v, _, _, _ = np.linalg.lstsq(C, np.ones(len(eqs)), rcond=None)
                    V[v, :] = V_v
                except:
                    pass
                    
            for k, i, w, eqs in W_indices:
                D = np.zeros((len(eqs), R))
                for idx, (u, v) in enumerate(eqs):
                    D[idx, :] = U[u, :] * V[v, :]
                try:
                    W_w, _, _, _ = np.linalg.lstsq(D, np.ones(len(eqs)), rcond=None)
                    W[w, :] = W_w
                except:
                    pass
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = i * m + j
                        v = j * p + k
                        w = k * n + i
                        s = 0.0
                        for r in range(R):
                            s += U[u, r] * V[v, r] * W[w, r]
                        err += (s - 1.0) ** 2
                        
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        U, V, W = best_U, best_V, best_W
        
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        if verify_solution(U_round, V_round, W_round):
            return U_round, V_round, W_round
            
        return None

    def try_rank(R, U_ref, V_ref, W_ref, max_restarts=10, time_limit=5):
        R_ref = U_ref.shape[1]
        if R > R_ref:
            return None
        
        start_time = time.time()
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            P = np.random.randn(R_ref, R)
            Q, _ = np.linalg.qr(P)
            P_proj = Q[:, :R]
            
            U_init = U_ref @ P_proj
            V_init = V_ref @ P_proj
            W_init = W_ref @ P_proj
            
            res = als_decompose(U_init, V_init, W_init, R, max_iters=100)
            if res is not None:
                return res
                
        return None

    start_time = time.time()
    deadline = 120
    
    best_result = (U_school, V_school, W_school)
    
    for R in range(17, 0, -1):
        if time.time() > deadline:
            break
            
        max_restarts = 20 if R < 15 else 10
        time_limit_per_rank = 10 if R < 15 else 5
        
        res = try_rank(R, U_school, V_school, W_school, max_restarts=max_restarts, time_limit=time_limit_per_rank)
        if res is not None:
            U, V, W = res
            best_result = (U, V, W)
            
    return {
        "U": best_result[0].tolist(),
        "V": best_result[1].tolist(),
        "W": best_result[2].tolist()
    }
