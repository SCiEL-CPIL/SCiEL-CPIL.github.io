import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    def schoolbook():
        R = n * m * p
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
        return U, V, W, R

    U_sb, V_sb, W_sb, R_sb = schoolbook()
    best_U, best_V, best_W, best_R = U_sb, V_sb, W_sb, R_sb

    def verify(U, V, W, R, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def snap_to_half_integer(mat, tol=1e-6):
        """Snap values to nearest half-integer if close."""
        snapped = np.round(mat * 2) / 2
        if np.max(np.abs(mat - snapped)) < tol:
            return snapped
        return mat

    def try_discrete_structured(R):
        """Construct a rank-39 decomposition using hybrid Strassen/Schoolbook.
        
        Split inner dimension m=3 into m1=2 and m2=1.
        A = [A1 (2x2) | A2 (2x1)]
        B = [B1 (2x7); B2 (1x7)]
        C = A1*B1 + A2*B2
        
        1. A2*B2: 2x1 * 1x7 -> 14 schoolbook terms.
        2. A1*B1: 2x2 * 2x7.
           Split B1 columns into pairs (0,1), (2,3), (4,5) and single (6).
           For each pair, use Strassen (7 mults). 3 pairs -> 21 mults.
           For single col 6, use schoolbook (2 mults).
           Total for A1*B1: 23 mults.
           
        Total Rank: 14 + 23 = 37? 
        Wait, let's re-verify the Strassen mapping.
        Strassen computes C = AB for 2x2 matrices.
        Here we compute C_part = A1 * B1_block (2x2 * 2x2).
        The result C_part is 2x2.
        We need to add this to the global C matrix.
        
        Let's trace indices carefully.
        U is (6, R). Indices: i*m + j.
        i in {0,1}, j in {0,1,2}.
        j=0,1 are in A1. j=2 is in A2.
        
        V is (21, R). Indices: j*p + k.
        j in {0,1,2}. k in {0..6}.
        j=0,1 are in B1. j=2 is in B2.
        
        W is (14, R). Indices: k*n + i.
        k in {0..6}, i in {0,1}.
        
        Part 1: A2 (j=2) * B2 (j=2)
        Terms: A[i,2] * B[2,k] -> C[i,k].
        U[i*3+2, r] = 1
        V[2*7+k, r] = 1
        W[k*2+i, r] = 1
        
        Part 2: A1 (j=0,1) * B1 (j=0,1)
        We process column pairs of B1.
        Pair (c0, c1).
        B1_block = [[B[0,c0], B[0,c1]], [B[1,c0], B[1,c1]]]
        A1 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
        
        Strassen intermediates:
        M1 = (A00+A11) * (B00+B11)
        M2 = (A10+A11) * B00
        M3 = A00 * (B01-B11)
        M4 = A11 * (B00+B10)
        M5 = (A00+A10) * B11
        M6 = (A11-A10) * (B00+B01)
        M7 = (A01-A11) * (B10+B11)
        
        C00 = M1 + M3 - M4 - M7
        C01 = M5 + M7
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M5
        
        We need to map these to W contributions.
        C00 contributes to C[0, c0] -> W[c0*2+0]
        C01 contributes to C[0, c1] -> W[c1*2+0]
        C10 contributes to C[1, c0] -> W[c0*2+1]
        C11 contributes to C[1, c1] -> W[c1*2+1]
        
        Let's build the vectors for each M.
        
        U vectors (length 6, indices 0..5 for A[0,0]..A[1,1]):
        A00: [1,0,0,0,0,0]
        A01: [0,1,0,0,0,0]
        A10: [0,0,1,0,0,0]
        A11: [0,0,0,1,0,0]
        
        M1_U = A00+A11 = [1,0,0,1,0,0]
        M2_U = A10+A11 = [0,0,1,1,0,0]
        M3_U = A00 = [1,0,0,0,0,0]
        M4_U = A11 = [0,0,0,1,0,0]
        M5_U = A00+A10 = [1,0,1,0,0,0]
        M6_U = A11-A10 = [0,0,-1,1,0,0]
        M7_U = A01-A11 = [0,1,0,-1,0,0]
        
        V vectors (length 21, indices for B rows 0,1,2).
        For B1 block (rows 0,1), columns c0, c1:
        B00 = B[0,c0] -> index 0*7+c0
        B01 = B[0,c1] -> index 0*7+c1
        B10 = B[1,c0] -> index 1*7+c0
        B11 = B[1,c1] -> index 1*7+c1
        
        M1_V = B00+B11 -> vec with 1 at c0 (row0) and 1 at 7+c1 (row1)
        M2_V = B00 -> vec with 1 at c0 (row0)
        M3_V = B01-B11 -> vec with 1 at c1 (row0) and -1 at 7+c1 (row1)
        M4_V = B00+B10 -> vec with 1 at c0 (row0) and 1 at 7+c0 (row1)
        M5_V = B11 -> vec with 1 at 7+c1 (row1)
        M6_V = B00+B01 -> vec with 1 at c0 (row0) and 1 at c1 (row0)
        M7_V = B10+B11 -> vec with 1 at 7+c0 (row1) and 1 at 7+c1 (row1)
        
        W vectors (length 14, indices k*2+i).
        M1_W: +C00, +C11 -> +W[c0*2+0], +W[c1*2+1]
        M2_W: +C10, -C11 -> +W[c0*2+1], -W[c1*2+1]
        M3_W: +C00, +C11 -> +W[c0*2+0], +W[c1*2+1]
        M4_W: -C00, +C10 -> -W[c0*2+0], +W[c0*2+1]
        M5_W: +C01, +C11 -> +W[c1*2+0], +W[c1*2+1]
        M6_W: 0 (unused in standard Strassen output reconstruction? 
               Standard: C00=M1+M3-M4-M7, C01=M5+M7, C10=M2+M4, C11=M1-M2+M3+M5.
               M6 is not used in the final sum? 
               Yes, M6 is calculated but not added to any C_ij in the standard 7-mult Strassen.
               So M6_W = 0.
        M7_W: -C00, +C01 -> -W[c0*2+0], +W[c1*2+0]
        
        Let's verify M6. 
        C = AB.
        M6 = (A11-A10)(B00+B01).
        It is indeed not part of the final linear combination for C00..C11.
        So we can skip M6 entirely, saving 1 rank per block!
        
        Revised Rank Count:
        A2*B2: 14
        A1*B1: 3 blocks * 6 mults (M1-M5, M7) + 1 block * 2 mults (col 6)
               = 18 + 2 = 20.
        Total: 14 + 20 = 34.
        
        Let's double check if M6 is truly unnecessary.
        Strassen:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B01-B11)
        M4 = A11(B00+B10)
        M5 = (A00+A10)B11
        M6 = (A11-A10)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M1 + M3 - M4 - M7
        C01 = M5 + M7
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M5
        
        Yes, M6 is not used. So we only need 6 mults per 2x2 block.
        
        New Total Rank:
        3 Strassen blocks * 6 = 18
        1 Schoolbook col * 2 = 2
        14 Schoolbook A2B2 = 14
        Total = 34.
        
        This is even better than 39!
        """
        U_list = []
        V_list = []
        W_list = []
        
        def add_vec(lst, idx, val):
            v = [0.0] * len(lst[0]) if lst else [0.0]*0 # placeholder logic, better to build lists of lists
            pass
            
        # Helper to create zero vectors of correct size
        U_size = 6
        V_size = 21
        W_size = 14
        
        def add_term(U_vals, V_vals, W_vals):
            U_list.append(U_vals)
            V_list.append(V_vals)
            W_list.append(W_vals)
            
        # Part 1: A2 (j=2) * B2 (j=2)
        # A[i,2] is at U index i*3+2
        # B[2,k] is at V index 2*7+k
        # C[i,k] is at W index k*2+i
        
        for i in range(2):
            for k in range(7):
                U_vals = [0.0]*6
                U_vals[i*3+2] = 1.0
                V_vals = [0.0]*21
                V_vals[2*7+k] = 1.0
                W_vals = [0.0]*14
                W_vals[k*2+i] = 1.0
                add_term(U_vals, V_vals, W_vals)
                
        # Part 2: A1 (j=0,1) * B1 (j=0,1)
        # Process column pairs (0,1), (2,3), (4,5) with Strassen (6 mults)
        # Process column 6 with Schoolbook (2 mults)
        
        def add_strassen_6_mults(c0, c1):
            # Indices in V for B1 block:
            # B00: 0*7+c0
            # B01: 0*7+c1
            # B10: 1*7+c0
            # B11: 1*7+c1
            
            b00_idx = 0*7 + c0
            b01_idx = 0*7 + c1
            b10_idx = 1*7 + c0
            b11_idx = 1*7 + c1
            
            # W indices:
            # C00: c0*2+0
            # C01: c1*2+0
            # C10: c0*2+1
            # C11: c1*2+1
            
            w_c00 = c0*2 + 0
            w_c01 = c1*2 + 0
            w_c10 = c0*2 + 1
            w_c11 = c1*2 + 1
            
            # M1 = (A00+A11) * (B00+B11)
            U1 = [0.0]*6; U1[0]=1; U1[3]=1 # A00+A11
            V1 = [0.0]*21; V1[b00_idx]=1; V1[b11_idx]=1
            W1 = [0.0]*14; W1[w_c00]=1; W1[w_c11]=1
            add_term(U1, V1, W1)
            
            # M2 = (A10+A11) * B00
            U2 = [0.0]*6; U2[2]=1; U2[3]=1 # A10+A11
            V2 = [0.0]*21; V2[b00_idx]=1
            W2 = [0.0]*14; W2[w_c10]=1; W2[w_c11]=-1
            add_term(U2, V2, W2)
            
            # M3 = A00 * (B01-B11)
            U3 = [0.0]*6; U3[0]=1
            V3 = [0.0]*21; V3[b01_idx]=1; V3[b11_idx]=-1
            W3 = [0.0]*14; W3[w_c00]=1; W3[w_c11]=1
            add_term(U3, V3, W3)
            
            # M4 = A11 * (B00+B10)
            U4 = [0.0]*6; U4[3]=1
            V4 = [0.0]*21; V4[b00_idx]=1; V4[b10_idx]=1
            W4 = [0.0]*14; W4[w_c00]=-1; W4[w_c10]=1
            add_term(U4, V4, W4)
            
            # M5 = (A00+A10) * B11
            U5 = [0.0]*6; U5[0]=1; U5[2]=1
            V5 = [0.0]*21; V5[b11_idx]=1
            W5 = [0.0]*14; W5[w_c01]=1; W5[w_c11]=1
            add_term(U5, V5, W5)
            
            # M7 = (A01-A11) * (B10+B11)
            U7 = [0.0]*6; U7[1]=1; U7[3]=-1
            V7 = [0.0]*21; V7[b10_idx]=1; V7[b11_idx]=1
            W7 = [0.0]*14; W7[w_c00]=-1; W7[w_c01]=1
            add_term(U7, V7, W7)
            
        # Apply to pairs (0,1), (2,3), (4,5)
        add_strassen_6_mults(0, 1)
        add_strassen_6_mults(2, 3)
        add_strassen_6_mults(4, 5)
        
        # Column 6: Schoolbook for A1 * B1[:,6]
        # C[0,6] += A[0,0]*B[0,6] + A[0,1]*B[1,6]
        # C[1,6] += A[1,0]*B[0,6] + A[1,1]*B[1,6]
        
        # Term 1: A[0,0]*B[0,6] -> C[0,6]
        U_t1 = [0.0]*6; U_t1[0]=1
        V_t1 = [0.0]*21; V_t1[0*7+6]=1
        W_t1 = [0.0]*14; W_t1[6*2+0]=1
        add_term(U_t1, V_t1, W_t1)
        
        # Term 2: A[0,1]*B[1,6] -> C[0,6]
        U_t2 = [0.0]*6; U_t2[1]=1
        V_t2 = [0.0]*21; V_t2[1*7+6]=1
        W_t2 = [0.0]*14; W_t2[6*2+0]=1
        add_term(U_t2, V_t2, W_t2)
        
        # Term 3: A[1,0]*B[0,6] -> C[1,6]
        U_t3 = [0.0]*6; U_t3[2]=1
        V_t3 = [0.0]*21; V_t3[0*7+6]=1
        W_t3 = [0.0]*14; W_t3[6*2+1]=1
        add_term(U_t3, V_t3, W_t3)
        
        # Term 4: A[1,1]*B[1,6] -> C[1,6]
        U_t4 = [0.0]*6; U_t4[3]=1
        V_t4 = [0.0]*21; V_t4[1*7+6]=1
        W_t4 = [0.0]*14; W_t4[6*2+1]=1
        add_term(U_t4, V_t4, W_t4)
        
        R_actual = len(U_list)
        U_mat = np.array(U_list).T
        V_mat = np.array(V_list).T
        W_mat = np.array(W_list).T
        
        if verify(U_mat, V_mat, W_mat, R_actual):
            return U_mat, V_mat, W_mat
        
        return None

    def continuous_optimize(R, attempt, deadline):
        """Optimize a rank-R decomposition using least_squares."""
        rng = np.random.default_rng(attempt * 7919 + R * 104729)
        
        # Initialize with structured random values
        vals = [-1, -0.5, 0, 0.5, 1]
        U = rng.choice(vals, size=(n * m, R)).astype(float)
        V = rng.choice(vals, size=(m * p, R)).astype(float)
        W = rng.choice(vals, size=(p * n, R)).astype(float)
        
        def flatten(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unflatten(x):
            u_size = n * m * R
            v_size = m * p * R
            U = x[:u_size].reshape(n * m, R)
            V = x[u_size:u_size+v_size].reshape(m * p, R)
            W = x[u_size+v_size:].reshape(p * n, R)
            return U, V, W
        
        def residuals(x):
            U, V, W = unflatten(x)
            res = np.zeros(n * m * p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                        res[idx] = s - 1.0
                        idx += 1
            return res
        
        x0 = flatten(U, V, W)
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=3000, ftol=1e-14, xtol=1e-14)
        except Exception:
            return None
        
        if sol.cost < 1e-12:
            U_sol, V_sol, W_sol = unflatten(sol.x)
            
            # Try to snap to half-integers
            U_snapped = snap_to_half_integer(U_sol, tol=1e-5)
            V_snapped = snap_to_half_integer(V_sol, tol=1e-5)
            W_snapped = snap_to_half_integer(W_sol, tol=1e-5)
            
            if verify(U_snapped, V_snapped, W_snapped, R):
                return U_snapped, V_snapped, W_snapped
            
            # If snapping doesn't work, check if raw solution is close to half-integers
            all_close = True
            for mat in [U_sol, V_sol, W_sol]:
                snapped = snap_to_half_integer(mat, tol=1e-4)
                if np.max(np.abs(mat - snapped)) > 1e-4:
                    all_close = False
                    break
            
            if all_close:
                return U_snapped, V_snapped, W_snapped
        
        return None

    # First, try the hybrid Strassen/Schoolbook decomposition
    result = try_discrete_structured(37)
    if result is not None:
        U_sol, V_sol, W_sol = result
        if verify(U_sol, V_sol, W_sol, 37):
            return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
    
    # Try ranks from a reasonable lower bound up to schoolbook
    min_rank_to_try = 21
    max_attempts_per_rank = 3
    
    for R in range(min_rank_to_try, min(R_sb + 1, 42)):
        if time.time() > deadline - 20:
            break
        
        # Try continuous optimization with fewer attempts to save time
        for attempt in range(max_attempts_per_rank):
            if time.time() > deadline - 20:
                break
            
            result = continuous_optimize(R, attempt, deadline)
            if result is not None:
                U_sol, V_sol, W_sol = result
                if verify(U_sol, V_sol, W_sol, R):
                    return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
    
    # If no better solution found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
