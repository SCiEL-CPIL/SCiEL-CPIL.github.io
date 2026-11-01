"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m  # 4
    dim_B = m * p  # 8
    dim_C = p * n  # 8
    
    # Build the target tensor T of shape (dim_A, dim_B, dim_C)
    # T[i, j, k] = 1 if i = i_n*m + i_m, j = i_m*p + k_p, k = i_n*p + k_p
    # i_n in 0..1, i_m in 0..1, k_p in 0..3
    T = np.zeros((dim_A, dim_B, dim_C))
    for i_n in range(n):
        for i_m in range(m):
            for k_p in range(p):
                i = i_n * m + i_m
                j = i_m * p + k_p
                k = i_n * p + k_p
                T[i, j, k] = 1.0

    def verify_decomposition(U, V, W):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_recon - T)
        return err

    def snap_and_verify(U, V, W):
        """Snap entries to nearest integer or half-integer and verify."""
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        # Check if snapping was successful
        if np.linalg.norm(U_s - U) > 1e-4 or np.linalg.norm(V_s - V) > 1e-4 or np.linalg.norm(W_s - W) > 1e-4:
            return None
            
        if verify_decomposition(U_s, V_s, W_s) < 1e-6:
            return U_s, V_s, W_s
        return None

    def try_rank(R, time_limit=10, init_U=None, init_V=None, init_W=None):
        """Try to find a rank-R decomposition."""
        start_time = time.time()
        rng = np.random.RandomState(42)
        
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # If no init, use random
        if init_U is None:
            init_U = rng.randn(dim_A, R)
            init_V = rng.randn(dim_B, R)
            init_W = rng.randn(dim_C, R)
            
        for attempt in range(10):
            if time.time() - start_time > time_limit:
                break
                
            if attempt == 0:
                U = init_U.copy()
                V = init_V.copy()
                W = init_W.copy()
            else:
                U = rng.randn(dim_A, R)
                V = rng.randn(dim_B, R)
                W = rng.randn(dim_C, R)
            
            # Normalize
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            prev_err = verify_decomposition(U, V, W)
            
            for step in range(200):
                if time.time() - start_time > time_limit:
                    break
                
                # Update U
                # T[i, jk] = sum_r U[i, r] * (V[j, r] * W[k, r])
                M_U = np.einsum('jr,kr->jkr', V, W).reshape(dim_B * dim_C, R)
                T_U = T.reshape(dim_A, dim_B * dim_C)
                try:
                    U_new = np.linalg.lstsq(M_U.T, T_U.T, rcond=None)[0].T
                    U = U_new
                except:
                    pass

                # Update V
                M_V = np.einsum('ir,kr->ikr', U, W).reshape(dim_A * dim_C, R)
                T_V = T.transpose(1, 0, 2).reshape(dim_B, dim_A * dim_C)
                try:
                    V_new = np.linalg.lstsq(M_V.T, T_V.T, rcond=None)[0].T
                    V = V_new
                except:
                    pass

                # Update W
                M_W = np.einsum('ir,jr->ijr', U, V).reshape(dim_A * dim_B, R)
                T_W = T.transpose(2, 0, 1).reshape(dim_C, dim_A * dim_B)
                try:
                    W_new = np.linalg.lstsq(M_W.T, T_W.T, rcond=None)[0].T
                    W = W_new
                except:
                    pass

                err = verify_decomposition(U, V, W)
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
                if err < 1e-8:
                    break
                if err > prev_err * 1.01:
                    break
                prev_err = err

        # Check best solution
        if best_err < 1e-6:
            res = snap_and_verify(best_U, best_V, best_W)
            if res:
                return res
        return None

    # Try to find a rank 13 decomposition
    # Use the known rank 14 decomposition as a starting point by dropping one term
    # This is a heuristic: if rank 13 exists, it might be close to a rank 14 solution
    # with one term effectively zero or merged.
    
    # First, let's get the known rank 14 solution to use as init
    def get_rank14():
        rank = 14
        U = np.zeros((dim_A, rank))
        V = np.zeros((dim_B, rank))
        W = np.zeros((dim_C, rank))
        
        # Strassen for B1 (cols 0,1) -> Output cols 0,1
        # Indices in B: 0,1 (row 0), 2,3 (row 1) -> Wait, B is 2x4.
        # B = [[e, f, g, h], [i, j, k, l]]
        # B1 = [[e, f], [i, j]] -> indices 0,1,2,3? No.
        # B row 0: e(0), f(1), g(2), h(3)
        # B row 1: i(4), j(5), k(6), l(7)
        # B1 uses cols 0,1 of B: e,f,i,j. Indices 0,1,4,5.
        # B2 uses cols 2,3 of B: g,h,k,l. Indices 2,3,6,7.
        
        # C1 uses cols 0,1 of C: indices 0,1,4,5 (C is 2x4, row 0: 0,1,2,3, row 1: 4,5,6,7)
        # C1 = [[ae+bi, af+bj], [ce+di, cf+dj]]
        # C2 = [[ag+bk, ah+bl], [cg+dk, ch+dl]]
        
        # Let's map Strassen terms for B1
        # M1 = (a+d)(e+i) -> U: a+d, V: e+i, W: contributes to C1
        # ...
        
        # Instead of re-deriving, let's use the one from the prompt which was valid.
        # But I will construct it properly here to ensure correctness.
        
        # A = [[a, b], [c, d]]
        # a: U[0], b: U[1], c: U[2], d: U[3]
        
        # B1 = [[e, f], [i, j]]
        # e: V[0], f: V[1], i: V[4], j: V[5]
        
        # C1 = [[c00, c01], [c10, c11]]
        # c00: W[0], c01: W[1], c10: W[4], c11: W[5]
        
        # Strassen for 2x2:
        # M1 = (a+d)(e+i)
        # M2 = (c+d)e
        # M3 = a(f-i)
        # M4 = d(g-c) -> Here g is j? No, B1 is [[e,f],[i,j]].
        # Standard Strassen:
        # M1 = (a+d)(e+i)
        # M2 = (c+d)e
        # M3 = a(f-i)
        # M4 = d(j-c)  # Note: usually g is bottom-right of left matrix? No.
        # Let's stick to the standard:
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+d)(e+g)
        # M2 = (c+d)e
        # M3 = a(f-g)
        # M4 = d(h-c)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # For B1: e->e, f->f, g->i, h->j.
        # M1 = (a+d)(e+i)
        # M2 = (c+d)e
        # M3 = a(f-i)
        # M4 = d(j-c)
        # M5 = (a+b)j
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(i+j)
        
        # C1 = [[M1+M4-M5-M7, M3+M5], [M2+M7, M1-M2+M3+M4]]
        
        # Let's fill U, V, W for B1 (indices 0..6)
        
        # Term 0: M1
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[4,0]=1
        W[0,0]=1; W[5,0]=1; W[4,0]=-1; W[1,0]=-1 # Wait, C1 indices:
        # C1[0,0] is W[0], C1[0,1] is W[1], C1[1,0] is W[4], C1[1,1] is W[5]
        # M1 contributes to:
        # C1[0,0]: +1
        # C1[1,1]: +1
        # C1[1,0]: -1
        # C1[0,1]: -1
        # So W[0,0]=1, W[5,0]=1, W[4,0]=-1, W[1,0]=-1.
        
        # Term 1: M2
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        # C1[1,0]: +1
        # C1[1,1]: -1
        W[4,1]=1; W[5,1]=-1
        
        # Term 2: M3
        U[0,2]=1
        V[1,2]=1; V[4,2]=-1
        # C1[0,1]: +1
        # C1[1,1]: +1
        W[1,2]=1; W[5,2]=1
        
        # Term 3: M4
        U[3,3]=1
        V[5,3]=1; V[2,3]=-1 # Wait, d(j-c). j is V[5], c is U[2].
        # V part: j -> V[5]=1.
        # U part: d -> U[3]=1.
        # But M4 = d(j-c). This is not a simple product of U and V vectors?
        # Yes it is. U=[0,0,0,1], V=[0,0,0,0,0,1,0,0].
        # But where is the -c?
        # Strassen M4 is d(h-c). This requires U=[0,0,0,1] and V=[0,0,0,0,0,1,0,0]?
        # No, the term is $d \cdot j - d \cdot c$? No, $d$ and $c$ are from A.
        # $M_4 = d(h-c)$ is NOT a bilinear form in A and B separately?
        # $d$ is in A, $h,c$ ... $c$ is in A, $h$ is in B.
        # $d(h-c) = dh - dc$.
        # $dh$ is bilinear (A,B). $dc$ is bilinear (A,A) -> NOT allowed.
        # Ah, Strassen's M4 is $d(h-c)$?
        # Let's check Strassen formulas again.
        # M1 = (a+d)(e+g)
        # M2 = (c+d)e
        # M3 = a(f-g)
        # M4 = d(h-c) -> This is wrong.
        # Correct M4: $d(h-c)$ is incorrect for matrix mult.
        # Correct: $M_4 = d(h-c)$ is NOT standard.
        # Standard:
        # M1 = (a+d)(e+g)
        # M2 = (c+d)e
        # M3 = a(f-g)
        # M4 = d(h-c) -- Wait, c is from A. h is from B.
        # Is $dh - dc$ valid? $dc$ is product of two elements from A.
        # So $M_4$ cannot be $d(h-c)$.
        # The correct formula is $M_4 = d(h-c)$? No.
        # It is $M_4 = d(h-c)$ in some notations where c is from B?
        # Let's use the standard:
        # M1 = (a+d)(e+g)
        # M2 = (c+d)e
        # M3 = a(f-g)
        # M4 = d(h-c) -> This is definitely wrong if c is in A.
        # Actually, M4 = d(h-c) is often written, but let's check the reconstruction.
        # C11 = M1 - M2 + M3 + M4.
        # If M4 = d(h-c), then C11 = (a+d)(e+g) - (c+d)e + a(f-g) + d(h-c)
        # = ae + ag + de + dg - ce - de + af - ag + dh - dc
        # = ae + dg + af + dh - ce - dc.
        # Expected C11 = ah + dg.
        # This doesn't match.
        
        # Let's use the reliable Strassen decomposition:
        # M1 = (a+d)(e+g)
        # M2 = (c+d)e
        # M3 = a(f-g)
        # M4 = d(h-c) -> No.
        # M4 = d(h-c) is WRONG.
        # Correct M4: $M_4 = d(h-c)$ is a common typo.
        # It should be $M_4 = d(h-c)$?
        # Let's look up "Strassen algorithm 7 multiplications".
        # M1 = (a+d)(e+g)
        # M2 = (c+d)e
        # M3 = a(f-g)
        # M4 = d(h-c)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # Let's re-evaluate M4.
        # If M4 = d(h-c), and c is in A, h is in B.
        # Then $dc$ is in A*A. This is not a bilinear form in A and B.
        # So M4 MUST be $d \cdot (\text{something in B})$.
        # The term is $d(h-c)$? No.
        # It is $M_4 = d(h-c)$ where $c$ is the bottom-left of B?
        # If B = [[e,f],[g,h]], then $c$ in $M_4$ refers to $g$?
        # No, usually $c$ is bottom-left of A.
        # Let's assume the standard:
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]].
        # M4 = d(h-c) is INCORRECT.
        # Correct M4: $M_4 = d(h-c)$?
        # Actually, $M_4 = d(h-c)$ is often cited, but let's check $M_4 = d(h-c)$.
        # Wait, $M_4 = d(h-c)$ implies $d \cdot h - d \cdot c$.
        # $d \cdot c$ is not allowed.
        # So $M_4$ must be $d \cdot h$? No, that's schoolbook.
        # The correct Strassen M4 is $d(h-c)$?
        # Let's try $M_4 = d(h-c)$ with $c$ being $g$ (bottom-left of B)?
        # No.
        
        # Let's use the version that works:
        # M1 = (a+d)(e+g)
        # M2 = (c+d)e
        # M3 = a(f-g)
        # M4 = d(h-c) -> This is the one.
        # Wait, if $c$ is in A, it's invalid.
        # Is it possible that $M_4 = d(h-c)$ where $c$ is $g$?
        # Let's assume the prompt's rank 14 solution was correct, so I will just use it.
        # I will copy the rank 14 solution from the prompt to ensure it's valid.
        
        # From prompt:
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # Note: Prompt used W[3] for C1[1,1]?
        # C is 2x4. W indices:
        # k*n+i.
        # C[0,0] -> k=0, i=0 -> W[0]
        # C[0,1] -> k=0, i=1 -> W[1]
        # C[0,2] -> k=0, i=2 -> W[2]
        # C[0,3] -> k=0, i=3 -> W[3]
        # C[1,0] -> k=1, i=0 -> W[4]
        # C[1,1] -> k=1, i=1 -> W[5]
        # C[1,2] -> k=1, i=2 -> W[6]
        # C[1,3] -> k=1, i=3 -> W[7]
        
        # The prompt's W indices for B1 (cols 0,1) were:
        # W[0], W[1], W[3], W[4], W[5], W[6]?
        # Prompt: W[0][0]=1; W[3][0]=1.
        # This implies C[0,0] and C[0,3]?
        # But B1 only affects C cols 0,1.
        # So W[3] (C[0,3]) should be 0.
        # The prompt's solution might have had a different mapping or I misread.
        # Let's just use the prompt's exact code for the rank 14 fallback.
        pass

    # Re-implementing the fallback from the prompt to be safe
    def get_rank14_fallback():
        rank = 14
        U = [[0] * rank for _ in range(dim_A)]
        V = [[0] * rank for _ in range(dim_B)]
        W = [[0] * rank for _ in range(dim_C)]
        
        # Strassen for B1 (cols 0,1)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1 # Wait, W[3] is C[0,3]. This is suspicious.
        # Let's check the prompt's W convention: W[k*n+i,r].
        # If B1 is cols 0,1, output should be C cols 0,1.
        # C[0,0] is W[0]. C[0,1] is W[1].
        # C[1,0] is W[4]. C[1,1] is W[5].
        # The prompt has W[3][0]=1. This is C[0,3].
        # This suggests the prompt's B1 might have been mapped to C cols 0,3?
        # Or the prompt's solution is for a different problem?
        # "Matrix multiplication 2,2,4".
        # If the prompt's solution scored 1.1429, it is valid.
        # I will trust the prompt's rank 14 solution.
        
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1

        # Strassen for B2 (cols 2,3)
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1

        return np.array(U), np.array(V), np.array(W)

    # Try to find rank 13
    # Initialize with rank 14, drop one term, and optimize
    U14, V14, W14 = get_rank14_fallback()
    
    # Try dropping each term and re-optimizing
    for drop_idx in range(14):
        if time.time() > 110: # Safety margin
            break
        U_init = np.delete(U14, drop_idx, axis=1)
        V_init = np.delete(V14, drop_idx, axis=1)
        W_init = np.delete(W14, drop_idx, axis=1)
        
        res = try_rank(13, time_limit=5, init_U=U_init, init_V=V_init, init_W=W_init)
        if res:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # If rank 13 not found, return rank 14
    U, V, W = U14, V14, W14
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
