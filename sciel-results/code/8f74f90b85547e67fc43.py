"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    # Use 3x3 block structure where each 2x2 block is handled by Strassen (rank 7)
    # Block indices: I in 0..2 (rows of A), K in 0..2 (cols of A / rows of B), L in 0..2 (cols of B)
    # Each block is 2x2.
    
    # Strassen's 7 multiplication formulas for 2x2 matrices:
    # Given A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Strassen term r has:
    # - A-coefficients: which elements of A (0-3) contribute with what coefficient
    # - B-coefficients: which elements of B (0-3) contribute with what coefficient
    # - C-coefficients: which elements of C (0-3) it contributes to, with what coefficient
    
    # Let's encode each Strassen term as:
    # (a_coeffs, b_coeffs, c_coeffs)
    # where each is a dict mapping element index to coefficient
    
    # Element indices for 2x2: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    
    strassen_terms = []
    
    # M1 = (a+d)(e+h)
    # A: a(0)=1, d(3)=1
    # B: e(0)=1, h(3)=1
    # C: C11(0)=1, C22(3)=1
    strassen_terms.append(({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}))
    
    # M2 = (c+d)e
    # A: c(2)=1, d(3)=1
    # B: e(0)=1
    # C: C21(2)=1
    strassen_terms.append(({2:1, 3:1}, {0:1}, {2:1}))
    
    # M3 = a(f-h)
    # A: a(0)=1
    # B: f(1)=1, h(3)=-1
    # C: C12(1)=1, C22(3)=1
    strassen_terms.append(({0:1}, {1:1, 3:-1}, {1:1, 3:1}))
    
    # M4 = d(g-e)
    # A: d(3)=1
    # B: g(2)=1, e(0)=-1
    # C: C11(0)=1, C21(2)=1
    strassen_terms.append(({3:1}, {2:1, 0:-1}, {0:1, 2:1}))
    
    # M5 = (a+b)h
    # A: a(0)=1, b(1)=1
    # B: h(3)=1
    # C: C11(0)=-1, C12(1)=1
    strassen_terms.append(({0:1, 1:1}, {3:1}, {0:-1, 1:1}))
    
    # M6 = (c-a)(e+f)
    # A: c(2)=1, a(0)=-1
    # B: e(0)=1, f(1)=1
    # C: C22(3)=1
    strassen_terms.append(({2:1, 0:-1}, {0:1, 1:1}, {3:1}))
    
    # M7 = (b-d)(g+h)
    # A: b(1)=1, d(3)=-1
    # B: g(2)=1, h(3)=1
    # C: C11(0)=1
    strassen_terms.append(({1:1, 3:-1}, {2:1, 3:1}, {0:1}))
    
    rank = 0
    U = []
    V = []
    W = []
    
    # We'll build U, V, W as lists of lists (each row is a list of rank-length entries)
    # U has n*m = 36 rows
    # V has m*p = 36 rows  
    # W has p*n = 36 rows
    
    U = [[0] * 0 for _ in range(n * m)]  # placeholder
    V = [[0] * 0 for _ in range(m * p)]
    W = [[0] * 0 for _ in range(p * n)]
    
    # First pass: count total rank
    total_rank = 0
    for I in range(3):
        for K in range(3):
            for L in range(3):
                for r in range(7):
                    total_rank += 1
    
    # Now allocate
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for I in range(3):
        for K in range(3):
            for L in range(3):
                for r in range(7):
                    a_coeffs, b_coeffs, c_coeffs = strassen_terms[r]
                    
                    # Fill U: for each A element (local index a_idx), set U[I*4+a_idx][col] = coeff
                    # Local element indices: 0=I*2+0, 1=I*2+1, 2=I*2+2 (i.e., (I+1)*2), 3=I*2+3
                    # Wait, let me be careful with global indices.
                    # Block A[I,K] occupies rows 2*I to 2*I+1, cols 2*K to 2*K+1
                    # Local (0,0) -> global (2*I, 2*K)
                    # Local (0,1) -> global (2*I, 2*K+1)
                    # Local (1,0) -> global (2*I+1, 2*K)
                    # Local (1,1) -> global (2*I+1, 2*K+1)
                    
                    for a_idx, a_coeff in a_coeffs.items():
                        row = (2*I + a_idx // 2) * m + (2*K + a_idx % 2)
                        U[row][col] = a_coeff
                    
                    # Fill V: for each B element (local index b_idx)
                    # Block B[K,L] occupies rows 2*K to 2*K+1, cols 2*L to 2*L+1
                    # Local (0,0) -> global (2*K, 2*L)
                    # Local (0,1) -> global (2*K, 2*L+1)
                    # Local (1,0) -> global (2*K+1, 2*L)
                    # Local (1,1) -> global (2*K+1, 2*L+1)
                    
                    for b_idx, b_coeff in b_coeffs.items():
                        row = (2*K + b_idx // 2) * p + (2*L + b_idx % 2)
                        V[row][col] = b_coeff
                    
                    # Fill W: for each C element (local index c_idx)
                    # W uses trace convention: W[k*n+i, r] where k is col of B, i is row of A
                    # Block C[I,L] occupies rows 2*I to 2*I+1, cols 2*L to 2*L+1
                    # Local (0,0) -> i=2*I, k=2*L
                    # Local (0,1) -> i=2*I, k=2*L+1
                    # Local (1,0) -> i=2*I+1, k=2*L
                    # Local (1,1) -> i=2*I+1, k=2*L+1
                    # W row index = k*n + i
                    
                    for c_idx, c_coeff in c_coeffs.items():
                        i_local = c_idx // 2
                        k_local = c_idx % 2
                        i_global = 2*I + i_local
                        k_global = 2*L + k_local
                        row = k_global * n + i_global
                        W[row][col] = c_coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
