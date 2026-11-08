import numpy as np
import time

_START_TIME = time.time()
DEADLINE_SECONDS = 130

def _elapsed():
    return time.time() - _START_TIME

def _time_remaining():
    return DEADLINE_SECONDS - _elapsed()

def construct():
    n, m, p = 2, 4, 7
    nA = n * m   # 8
    nB = m * p   # 28
    nW = p * n   # 14

    def build_rank48():
        """
        Rank-48 decomposition using rank-10 for 2x2x3 blocks.
        
        2x2x3: A=[[a,b],[c,d]], B=[[e,f,g],[h,i,j]], C=[[C00,C01,C02],[C10,C11,C12]]
        
        Rank-10 construction:
        P1 = (a+d)(e+h)     -> contributes to C00, C11
        P2 = (c+d)e         -> contributes to C10, C11
        P3 = a(f-h)         -> contributes to C01, C11
        P4 = d(i-e)         -> contributes to C00, C10
        P5 = (a+b)h         -> contributes to C00, C01
        P6 = (c-a)(e+f)     -> contributes to C11
        P7 = (b-d)(h+i)     -> contributes to C00
        P8 = (a+c)g         -> contributes to C02, C12
        P9 = (b+d)j         -> contributes to C02, C12
        P10 = (b-c)g        -> contributes to C02, C12
        
        Let me verify:
        C00 = a*e + b*h
        C01 = a*f + b*i
        C02 = a*g + b*j
        C10 = c*e + d*h
        C11 = c*f + d*i
        C12 = c*g + d*j
        
        From P1-P7 (Strassen on first 2 cols):
        C00 = P1 + P4 - P5 + P7
        C01 = P1 + P3
        C10 = P2 + P6
        C11 = P1 - P2 + P3 + P5
        
        For column 2:
        C02 = a*g + b*j
        C12 = c*g + d*j
        
        P8 = (a+c)g = a*g + c*g
        P9 = (b+d)j = b*j + d*j
        P10 = (b-c)g = b*g - c*g
        
        C02 = a*g + b*j = P8 - c*g + P9 - d*j = P8 + P9 - (c*g + d*j) = P8 + P9 - C12
        C12 = c*g + d*j
        
        Hmm, this is circular. Let me try different P8, P9, P10:
        
        P8 = a*g
        P9 = b*j
        P10 = (c+d)g = c*g + d*g
        
        C02 = P8 + P9
        C12 = c*g + d*j = P10 - d*g + d*j ... still have d*j
        
        Let me try:
        P8 = (a+b)g = a*g + b*g
        P9 = (c+d)j = c*j + d*j
        P10 = (b-d)g = b*g - d*g
        
        C02 = a*g + b*j = P8 - b*g + b*j = P8 - (P10 + d*g) + b*j ... messy
        
        Alternative approach: use the known fact that rank(2x2x3) = 10 and find it numerically.
        """
        
        # Let me just use the 7+4=11 approach for now, giving rank 50
        # But try to find a better construction
        
        # Actually, let me try a different decomposition:
        # For 2x2x3, we can use:
        # P1 = a*e, P2 = a*f, P3 = a*g, P4 = b*h, P5 = b*i, P6 = b*j,
        # P7 = c*e, P8 = c*f, P9 = c*g, P10 = d*h, P11 = d*i, P12 = d*j
        # That's rank 12 (schoolbook). Not good.
        
        # Let me try to find rank-10 numerically
        # Set up the system and solve
        
        def find_rank10_2x2x3():
            """Find rank-10 decomposition for 2x2x3 matrix multiplication."""
            # We need U (4x10), V (6x10), W (6x10) such that
            # sum_l U[i*2+j, l] * V[j*3+k, l] * W[k*2+i, l] = 1 for all (i,j,k)
            
            # This is 12 equations in 160 unknowns. Very underdetermined.
            # Let's try to find a solution with integer/half-integer entries.
            
            # Strategy: Fix W to be a 6x10 matrix, then solve for U, V.
            # Or use random initialization and verify.
            
            np.random.seed(123)
            for attempt in range(5000):
                # Random initialization with small integers
                U = np.random.randint(-2, 3, (4, 10)).astype(float)
                V = np.random.randint(-2, 3, (6, 10)).astype(float)
                W = np.random.randint(-2, 3, (6, 10)).astype(float)
                
                # Check if this works
                ok = True
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            val = 0.0
                            for l in range(10):
                                val += U[i*2+j, l] * V[j*3+k, l] * W[k*2+i, l]
                            if abs(val - 1.0) > 1e-9:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                
                if ok:
                    return U.tolist(), V.tolist(), W.tolist()
            
            return None
        
        # Try to find rank-10 decomposition
        rank10_result = find_rank10_2x2x3()
        
        if rank10_result is not None:
            U10, V10, W10 = rank10_result
            U = [[0] * 48 for _ in range(nA)]
            V = [[0] * 48 for _ in range(nB)]
            W = [[0] * 48 for _ in range(nW)]
            col = 0
            
            # Add rank-10 blocks for each 2x2x3 sub-block
            # For j in [0,1] and [2,3], and k in [0,1,2], [3,4,5], [6]
            # Actually, 2x2x7 = 2x2x3 + 2x2x3 + 2x2x1
            # 2x2x1 has rank 4 (schoolbook)
            
            def add_2x2x3_block(j_start, k_start):
                nonlocal col
                # Map local indices to global
                for l in range(10):
                    for i in range(2):
                        U[i*m + j_start][col] += U10[i*2, l]
                        U[i*m + j_start + 1][col] += U10[i*2+1, l]
                    for j in range(2):
                        for k in range(3):
                            V[(j_start + j)*p + k_start + k][col] += V10[j*3 + k, l]
                    for k in range(3):
                        for i in range(2):
                            W[(k_start + k)*n + i][col] += W10[k*2 + i, l]
                col += 10
            
            def add_2x2x1_block(j_start, k_start):
                nonlocal col
                # Schoolbook for 2x2x1
                a_idx = 0*m + j_start
                b_idx = 0*m + j_start + 1
                c_idx = 1*m + j_start
                d_idx = 1*m + j_start + 1
                e_idx = j_start*p + k_start
                g_idx = (j_start+1)*p + k_start
                c00 = k_start*n + 0
                c10 = k_start*n + 1
                
                U[a_idx][col] = 1; V[e_idx][col] = 1; W[c00][col] = 1; col += 1
                U[b_idx][col] = 1; V[g_idx][col] = 1; W[c00][col] = 1; col += 1
                U[c_idx][col] = 1; V[e_idx][col] = 1; W[c10][col] = 1; col += 1
                U[d_idx][col] = 1; V[g_idx][col] = 1; W[c10][col] = 1; col += 1
            
            # For j_start in [0, 2]:
            for j_start in [0, 2]:
                # 2x2x7 = 2x2x3 (k=0,1,2) + 2x2x3 (k=3,4,5) + 2x2x1 (k=6)
                add_2x2x3_block(j_start, 0)
                add_2x2x3_block(j_start, 3)
                add_2x2x1_block(j_start, 6)
            
            return U[:col], V[:col], W[:col], col
        else:
            # Fallback to rank-50 with Strassen + schoolbook
            U = [[0] * 50 for _ in range(nA)]
            V = [[0] * 50 for _ in range(nB)]
            W = [[0] * 50 for _ in range(nW)]
            col = 0
            
            def add_strassen(i0, j0, k0):
                nonlocal col
                a_idx = i0 * m + j0
                b_idx = i0 * m + (j0 + 1)
                c_idx = (i0 + 1) * m + j0
                d_idx = (i0 + 1) * m + (j0 + 1)
                e_idx = j0 * p + k0
                f_idx = j0 * p + (k0 + 1)
                g_idx = (j0 + 1) * p + k0
                h_idx = (j0 + 1) * p + (k0 + 1)
                c00 = k0 * n + i0
                c01 = (k0 + 1) * n + i0
                c10 = k0 * n + (i0 + 1)
                c11 = (k0 + 1) * n + (i0 + 1)
                
                # M1 = (a+d)(e+h)
                U[a_idx][col] += 1; U[d_idx][col] += 1
                V[e_idx][col] += 1; V[h_idx][col] += 1
                W[c00][col] += 1; W[c11][col] += 1
                col += 1
                
                # M2 = (c+d)e
                U[c_idx][col] += 1; U[d_idx][col] += 1
                V[e_idx][col] += 1
                W[c10][col] += 1; W[c11][col] -= 1
                col += 1
                
                # M3 = a(f-h)
                U[a_idx][col] += 1
                V[f_idx][col] += 1; V[h_idx][col] -= 1
                W[c01][col] += 1; W[c11][col] += 1
                col += 1
                
                # M4 = d(g-e)
                U[d_idx][col] += 1
                V[g_idx][col] += 1; V[e_idx][col] -= 1
                W[c00][col] += 1; W[c10][col] += 1
                col += 1
                
                # M5 = (a+b)h
                U[a_idx][col] += 1; U[b_idx][col] += 1
                V[h_idx][col] += 1
                W[c00][col] -= 1; W[c01][col] += 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[c_idx][col] += 1; U[a_idx][col] -= 1
                V[e_idx][col] += 1; V[f_idx][col] += 1
                W[c11][col] += 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[b_idx][col] += 1; U[d_idx][col] -= 1
                V[g_idx][col] += 1; V[h_idx][col] += 1
                W[c00][col] += 1
                col += 1
            
            def add_schoolbook_221(i0, j0, k0):
                nonlocal col
                a_idx = i0 * m + j0
                b_idx = i0 * m + (j0 + 1)
                c_idx = (i0 + 1) * m + j0
                d_idx = (i0 + 1) * m + (j0 + 1)
                e_idx = j0 * p + k0
                g_idx = (j0 + 1) * p + k0
                c00 = k0 * n + i0
                c10 = k0 * n + (i0 + 1)
                
                U[a_idx][col] += 1; V[e_idx][col] += 1; W[c00][col] += 1; col += 1
                U[b_idx][col] += 1; V[g_idx][col] += 1; W[c00][col] += 1; col += 1
                U[c_idx][col] += 1; V[e_idx][col] += 1; W[c10][col] += 1; col += 1
                U[d_idx][col] += 1; V[g_idx][col] += 1; W[c10][col] += 1; col += 1
            
            for j_start in [0, 2]:
                for k_start in [0, 2, 4]:
                    add_strassen(0, j_start, k_start)
                add_schoolbook_221(0, j_start, 6)
            
            return U[:col], V[:col], W[:col], col

    def verify(U, V, W, tol=1e-9):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    ai = i * m + j
                    bi = j * p + k
                    wi = k * n + i
                    for l in range(r):
                        val += U[ai][l] * V[bi][l] * W[wi][l]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Build the decomposition
    U, V, W, rank = build_rank48()
    
    # Verify
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 56
    U = [[0] * (nA * nW) for _ in range(nA)]
    V = [[0] * (nA * nW) for _ in range(nB)]
    W = [[0] * (nA * nW) for _ in range(nW)]
    col = 0
    for j in range(m):
        for i in range(n):
            for k in range(p):
                ai = i * m + j
                bi = j * p + k
                wi = k * n + i
                U[ai][col] = 1
                V[bi][col] = 1
                W[wi][col] = 1
                col += 1
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
