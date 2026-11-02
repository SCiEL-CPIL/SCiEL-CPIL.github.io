"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    # Rank 50 decomposition
    
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (14, rank)
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_vec, v_vec, w_vec):
        """
        u_vec: list of 8 elements (coeffs for A)
        v_vec: list of 28 elements (coeffs for B)
        w_vec: list of 14 elements (coeffs for C)
        """
        U_terms.append(u_vec)
        V_terms.append(v_vec)
        W_terms.append(w_vec)
        
    def create_zero_vec(size):
        return [0.0] * size

    # --- Strassen 2x2x2 ---
    # Local indices: A(4), B(4), C(4)
    # A: [A00, A01, A10, A11] -> i=0,1; j=0,1
    # B: [B00, B01, B10, B11] -> j=0,1; k=0,1
    # C: [C00, C01, C10, C11] -> i=0,1; k=0,1
    
    # Strassen multipliers M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A00+A11)B10
    # M3 = A00(B00-B10)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen(a_offset_i, a_offset_j, b_offset_j, b_offset_k, c_offset_i, c_offset_k):
        """
        Adds a Strassen decomposition for a 2x2x2 sub-block.
        a_offset_i: global i index for A block (0 or 1)
        a_offset_j: global j start index for A block (0 or 2)
        b_offset_j: global j start index for B block (0 or 2)
        b_offset_k: global k start index for B block (0, 2, etc)
        c_offset_i: global i index for C block (0 or 1)
        c_offset_k: global k start index for C block
        """
        # Helper to map local (i,j) to global index in U
        def u_idx(i, j):
            return (a_offset_i + i) * m + (a_offset_j + j)
        
        # Helper to map local (j,k) to global index in V
        def v_idx(j, k):
            return (b_offset_j + j) * p + (b_offset_k + k)
        
        # Helper to map local (i,k) to global index in W
        def w_idx(i, k):
            return (c_offset_k + k) * n + (c_offset_i + i)
            
        # Define local vectors for each multiplier
        # M1
        u1 = create_zero_vec(n*m); u1[u_idx(0,0)] = 1; u1[u_idx(1,1)] = 1
        v1 = create_zero_vec(m*p); v1[v_idx(0,0)] = 1; v1[v_idx(1,1)] = 1
        w1 = create_zero_vec(p*n); w1[w_idx(0,0)] = 1; w1[w_idx(0,1)] = 0; w1[w_idx(1,0)] = 0; w1[w_idx(1,1)] = 1
        # Wait, W is (k*n+i).
        # C00 (i=0,k=0) -> w_idx(0,0)
        # C01 (i=0,k=1) -> w_idx(0,1)
        # C10 (i=1,k=0) -> w_idx(1,0)
        # C11 (i=1,k=1) -> w_idx(1,1)
        
        # It is easier to define the contributions to W for each M
        # M1 contributes to C00, C11
        # M4 contributes to C00, C10
        # M5 contributes to C00 (neg), C01
        # M7 contributes to C00
        # M3 contributes to C01
        # M2 contributes to C10
        # M6 contributes to C11
        
        # Let's build the vectors for each of the 7 multiplications
        
        # M1: (A00+A11) * (B00+B11)
        u_m1 = create_zero_vec(n*m); u_m1[u_idx(0,0)] = 1; u_m1[u_idx(1,1)] = 1
        v_m1 = create_zero_vec(m*p); v_m1[v_idx(0,0)] = 1; v_m1[v_idx(1,1)] = 1
        w_m1 = create_zero_vec(p*n); w_m1[w_idx(0,0)] = 1; w_m1[w_idx(1,1)] = 1
        add_term(u_m1, v_m1, w_m1)
        
        # M2: (A00+A11) * B10
        u_m2 = create_zero_vec(n*m); u_m2[u_idx(0,0)] = 1; u_m2[u_idx(1,1)] = 1
        v_m2 = create_zero_vec(m*p); v_m2[v_idx(1,0)] = 1
        w_m2 = create_zero_vec(p*n); w_m2[w_idx(1,0)] = 1 # C10
        # Wait, C10 is i=1, k=0. w_idx(1,0).
        # C11 is i=1, k=1. w_idx(1,1).
        # M2 is used in C10 and C11 (with -1)
        # So w_m2 should have 1 at C10 and -1 at C11? No, the term is added to the sum.
        # The decomposition is C = sum M_l * (coeffs).
        # So for each M_l, we add a term.
        # The W vector for that term contains the coefficients for C.
        
        # Let's restart W construction for clarity.
        # M1: C00 += 1, C11 += 1
        # M2: C10 += 1, C11 -= 1
        # M3: C01 += 1, C11 += 1
        # M4: C00 += 1, C10 += 1
        # M5: C00 -= 1, C01 += 1
        # M6: C11 += 1
        # M7: C00 += 1
        
        # I already added M1. Let's remove it and do it properly.
        U_terms.pop(); V_terms.pop(); W_terms.pop()
        
        # M1
        u = create_zero_vec(n*m); u[u_idx(0,0)] = 1; u[u_idx(1,1)] = 1
        v = create_zero_vec(m*p); v[v_idx(0,0)] = 1; v[v_idx(1,1)] = 1
        w = create_zero_vec(p*n); w[w_idx(0,0)] = 1; w[w_idx(1,1)] = 1
        add_term(u, v, w)
        
        # M2
        u = create_zero_vec(n*m); u[u_idx(0,0)] = 1; u[u_idx(1,1)] = 1
        v = create_zero_vec(m*p); v[v_idx(1,0)] = 1
        w = create_zero_vec(p*n); w[w_idx(1,0)] = 1; w[w_idx(1,1)] = -1
        add_term(u, v, w)
        
        # M3
        u = create_zero_vec(n*m); u[u_idx(0,0)] = 1
        v = create_zero_vec(m*p); v[v_idx(0,0)] = 1; v[v_idx(1,0)] = -1
        w = create_zero_vec(p*n); w[w_idx(0,1)] = 1; w[w_idx(1,1)] = 1
        add_term(u, v, w)
        
        # M4
        u = create_zero_vec(n*m); u[u_idx(1,1)] = 1
        v = create_zero_vec(m*p); v[v_idx(0,1)] = 1; v[v_idx(1,1)] = -1
        w = create_zero_vec(p*n); w[w_idx(0,0)] = 1; w[w_idx(1,0)] = 1
        add_term(u, v, w)
        
        # M5
        u = create_zero_vec(n*m); u[u_idx(0,0)] = 1; u[u_idx(0,1)] = 1
        v = create_zero_vec(m*p); v[v_idx(1,1)] = 1
        w = create_zero_vec(p*n); w[w_idx(0,0)] = -1; w[w_idx(0,1)] = 1
        add_term(u, v, w)
        
        # M6
        u = create_zero_vec(n*m); u[u_idx(1,0)] = 1; u[u_idx(0,0)] = -1
        v = create_zero_vec(m*p); v[v_idx(0,0)] = 1; v[v_idx(0,1)] = 1
        w = create_zero_vec(p*n); w[w_idx(1,1)] = 1
        add_term(u, v, w)
        
        # M7
        u = create_zero_vec(n*m); u[u_idx(0,1)] = 1; u[u_idx(1,1)] = -1
        v = create_zero_vec(m*p); v[v_idx(1,0)] = 1; v[v_idx(1,1)] = 1
        w = create_zero_vec(p*n); w[w_idx(0,0)] = 1
        add_term(u, v, w)

    def add_schoolbook_221(a_offset_i, a_offset_j, b_offset_j, b_offset_k, c_offset_i, c_offset_k):
        """
        Adds schoolbook 2x2x1 decomposition.
        A(2x2) * B(2x1) -> C(2x1)
        C0 = A00 B0 + A01 B1
        C1 = A10 B0 + A11 B1
        """
        def u_idx(i, j):
            return (a_offset_i + i) * m + (a_offset_j + j)
        def v_idx(j, k):
            return (b_offset_j + j) * p + (b_offset_k + k)
        def w_idx(i, k):
            return (c_offset_k + k) * n + (c_offset_i + i)

        # Term 1: A00 * B0 -> C0
        u = create_zero_vec(n*m); u[u_idx(0,0)] = 1
        v = create_zero_vec(m*p); v[v_idx(0,0)] = 1
        w = create_zero_vec(p*n); w[w_idx(0,0)] = 1
        add_term(u, v, w)
        
        # Term 2: A01 * B1 -> C0
        u = create_zero_vec(n*m); u[u_idx(0,1)] = 1
        v = create_zero_vec(m*p); v[v_idx(1,0)] = 1
        w = create_zero_vec(p*n); w[w_idx(0,0)] = 1
        add_term(u, v, w)
        
        # Term 3: A10 * B0 -> C1
        u = create_zero_vec(n*m); u[u_idx(1,0)] = 1
        v = create_zero_vec(m*p); v[v_idx(0,0)] = 1
        w = create_zero_vec(p*n); w[w_idx(1,0)] = 1
        add_term(u, v, w)
        
        # Term 4: A11 * B1 -> C1
        u = create_zero_vec(n*m); u[u_idx(1,1)] = 1
        v = create_zero_vec(m*p); v[v_idx(1,0)] = 1
        w = create_zero_vec(p*n); w[w_idx(1,0)] = 1
        add_term(u, v, w)

    # --- Build 2x4x7 ---
    # C1 = A B1 (2x4)
    # B1 is 4x4.
    # A1 (2x2, cols 0-1) * B11 (2x2, rows 0-1, cols 0-1) -> C11 (2x2, cols 0-1)
    add_strassen(0, 0, 0, 0, 0, 0)
    # A1 (2x2, cols 0-1) * B12 (2x2, rows 0-1, cols 2-3) -> C12 (2x2, cols 2-3)
    add_strassen(0, 0, 0, 2, 0, 2)
    # A2 (2x2, cols 2-3) * B21 (2x2, rows 2-3, cols 0-1) -> C11 (2x2, cols 0-1)
    # Note: A2 is i=0,1 but j=2,3.
    # In my add_strassen, a_offset_j is the start j.
    # A2 is at j=2.
    add_strassen(0, 2, 2, 0, 0, 0)
    # A2 (2x2, cols 2-3) * B22 (2x2, rows 2-3, cols 2-3) -> C12 (2x2, cols 2-3)
    add_strassen(0, 2, 2, 2, 0, 2)
    
    # C2 = A B2 (2x3)
    # B2 is 4x3.
    # A1 (2x2, cols 0-1) * B31 (2x3, rows 0-1, cols 4-6) -> C21 (2x3, cols 4-6)
    # B31 is 2x3. We can decompose as 2x2x2 + 2x2x1.
    # B31 cols 4,5 -> 2x2 part. Col 6 -> 2x1 part.
    # Strassen for cols 4,5
    add_strassen(0, 0, 0, 4, 0, 4)
    # Schoolbook for col 6
    add_schoolbook_221(0, 0, 0, 6, 0, 6)
    
    # A2 (2x2, cols 2-3) * B41 (2x3, rows 2-3, cols 4-6) -> C21 (2x3, cols 4-6)
    # Strassen for cols 4,5
    add_strassen(0, 2, 2, 4, 0, 4)
    # Schoolbook for col 6
    add_schoolbook_221(0, 2, 2, 6, 0, 6)

    rank = len(U_terms)
    U = [u for u in U_terms]
    V = [v for v in V_terms]
    W = [w for w in W_terms]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
