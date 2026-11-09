"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Uses recursive block decomposition with Strassen for 2x2x2 blocks.
    Target rank: ~60 (vs schoolbook 84)
    """
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # We'll build the decomposition as a list of (u, v, w) vectors.
    # u has length n*m, v has length m*p, w has length p*n.
    # The tensor is sum_r u_r ⊗ v_r ⊗ w_r
    
    terms = []
    
    def add_term(u, v, w):
        """Add a term (u, v, w) to the decomposition."""
        terms.append((np.array(u, dtype=float), np.array(v, dtype=float), np.array(w, dtype=float)))
    
    # Helper: schoolbook for n x m * m x p, with offsets
    def schoolbook_block(n_b, m_b, p_b, i_off, j_off, k_off):
        """
        Add schoolbook terms for a block at position (i_off, j_off, k_off)
        with dimensions n_b x m_b * m_b x p_b.
        """
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    u = np.zeros(n*m)
                    v = np.zeros(m*p)
                    w = np.zeros(p*n)
                    u[(i_off+i)*m + (j_off+j)] = 1
                    v[(j_off+j)*p + (k_off+k)] = 1
                    w[(k_off+k)*n + (i_off+i)] = 1
                    add_term(u, v, w)
    
    # Helper: Strassen for 2x2x2 block
    def strassen_block(i_off, j_off, k_off):
        """
        Add Strassen terms for a 2x2 x 2x2 block at position (i_off, j_off, k_off).
        A block is 2x2, B block is 2x2, C block is 2x2.
        """
        # A indices: (i_off, j_off), (i_off, j_off+1), (i_off+1, j_off), (i_off+1, j_off+1)
        # B indices: (j_off, k_off), (j_off, k_off+1), (j_off+1, k_off), (j_off+1, k_off+1)
        # C indices: (i_off, k_off), (i_off, k_off+1), (i_off+1, k_off), (i_off+1, k_off+1)
        
        # Strassen's 7 products
        # M1 = (a+d)(e+h)
        u1 = np.zeros(n*m); v1 = np.zeros(m*p); w1 = np.zeros(p*n)
        u1[i_off*m + j_off] = 1; u1[(i_off+1)*m + (j_off+1)] = 1
        v1[j_off*p + k_off] = 1; v1[(j_off+1)*p + (k_off+1)] = 1
        w1[k_off*n + i_off] = 1; w1[(k_off+1)*n + (i_off+1)] = 1
        add_term(u1, v1, w1)
        
        # M2 = (c+d)e
        u2 = np.zeros(n*m); v2 = np.zeros(m*p); w2 = np.zeros(p*n)
        u2[(i_off+1)*m + j_off] = 1; u2[(i_off+1)*m + (j_off+1)] = 1
        v2[j_off*p + k_off] = 1
        w2[k_off*n + (i_off+1)] = 1
        add_term(u2, v2, w2)
        
        # M3 = a(f-h)
        u3 = np.zeros(n*m); v3 = np.zeros(m*p); w3 = np.zeros(p*n)
        u3[i_off*m + j_off] = 1
        v3[j_off*p + (k_off+1)] = 1; v3[(j_off+1)*p + k_off] = -1
        w3[(k_off+1)*n + i_off] = 1; w3[(k_off+1)*n + (i_off+1)] = 1
        add_term(u3, v3, w3)
        
        # M4 = d(g-e)
        u4 = np.zeros(n*m); v4 = np.zeros(m*p); w4 = np.zeros(p*n)
        u4[(i_off+1)*m + (j_off+1)] = 1
        v4[j_off*p + k_off] = -1; v4[(j_off+1)*p + k_off] = 1
        w4[k_off*n + i_off] = 1; w4[k_off*n + (i_off+1)] = 1
        add_term(u4, v4, w4)
        
        # M5 = (a+b)h
        u5 = np.zeros(n*m); v5 = np.zeros(m*p); w5 = np.zeros(p*n)
        u5[i_off*m + j_off] = 1; u5[i_off*m + (j_off+1)] = 1
        v5[(j_off+1)*p + (k_off+1)] = 1
        w5[k_off*n + i_off] = -1; w5[(k_off+1)*n + i_off] = 1
        add_term(u5, v5, w5)
        
        # M6 = (c-d)(g+h)
        u6 = np.zeros(n*m); v6 = np.zeros(m*p); w6 = np.zeros(p*n)
        u6[(i_off+1)*m + j_off] = 1; u6[(i_off+1)*m + (j_off+1)] = -1
        v6[(j_off+1)*p + k_off] = 1; v6[(j_off+1)*p + (k_off+1)] = 1
        w6[(k_off+1)*n + (i_off+1)] = 1
        add_term(u6, v6, w6)
        
        # M7 = (b-d)(f+g)
        u7 = np.zeros(n*m); v7 = np.zeros(m*p); w7 = np.zeros(p*n)
        u7[i_off*m + (j_off+1)] = 1; u7[(i_off+1)*m + (j_off+1)] = -1
        v7[j_off*p + (k_off+1)] = 1; v7[(j_off+1)*p + k_off] = 1
        w7[k_off*n + i_off] = 1
        add_term(u7, v7, w7)
    
    # Now decompose 3x4 x 4x7
    # Split m=4 into 2+2
    # A = [A1 (3x2) A2 (3x2)], B = [B1 (2x7); B2 (2x7)]
    # C = A1*B1 + A2*B2
    
    # For each of A1*B1 and A2*B2 (both 3x2 x 2x7):
    # Split n=3 into 2+1, p=7 into 4+3
    # A_sub (3x2) = [A_sub1 (2x2); A_sub2 (1x2)]
    # B_sub (2x7) = [B_sub1 (2x4) B_sub2 (2x3)]
    # A_sub*B_sub = [A_sub1*B_sub1 + A_sub2*B_sub1, A_sub1*B_sub2 + A_sub2*B_sub2]
    
    # For A1*B1 (i_off=0, j_off=0, k_off=0):
    # A_sub1 (2x2) at (0,0), A_sub2 (1x2) at (2,0)
    # B_sub1 (2x4) at (0,0), B_sub2 (2x3) at (0,4)
    
    # A_sub1*B_sub1: 2x2 x 2x4
    # Split p=4 into 2+2
    # B_sub1 = [B_sub1_1 (2x2) B_sub1_2 (2x2)]
    # A_sub1*B_sub1 = [A_sub1*B_sub1_1, A_sub1*B_sub1_2]
    # Each is 2x2 x 2x2, use Strassen
    
    # A_sub1*B_sub1_1: Strassen at (0,0,0)
    strassen_block(0, 0, 0)
    # A_sub1*B_sub1_2: Strassen at (0,0,2)
    strassen_block(0, 0, 2)
    
    # A_sub2*B_sub1: 1x2 x 2x4, schoolbook
    schoolbook_block(1, 2, 4, 2, 0, 0)
    
    # A_sub1*B_sub2: 2x2 x 2x3
    # Split p=3 into 2+1
    # B_sub2 = [B_sub2_1 (2x2) B_sub2_2 (2x1)]
    # A_sub1*B_sub2 = [A_sub1*B_sub2_1, A_sub1*B_sub2_2]
    # A_sub1*B_sub2_1: Strassen at (0,0,4)
    strassen_block(0, 0, 4)
    # A_sub1*B_sub2_2: 2x2 x 2x1, schoolbook
    schoolbook_block(2, 2, 1, 0, 0, 6)
    
    # A_sub2*B_sub2: 1x2 x 2x3, schoolbook
    schoolbook_block(1, 2, 3, 2, 0, 4)
    
    # For A2*B2 (i_off=0, j_off=2, k_off=0):
    # Same structure, but j offset by 2
    # A_sub1 (2x2) at (0,2), A_sub2 (1x2) at (2,2)
    # B_sub1 (2x4) at (2,0), B_sub2 (2x3) at (2,4)
    
    # A_sub1*B_sub1: 2x2 x 2x4
    # A_sub1*B_sub1_1: Strassen at (0,2,0)
    strassen_block(0, 2, 0)
    # A_sub1*B_sub1_2: Strassen at (0,2,2)
    strassen_block(0, 2, 2)
    
    # A_sub2*B_sub1: 1x2 x 2x4, schoolbook
    schoolbook_block(1, 2, 4, 2, 2, 0)
    
    # A_sub1*B_sub2: 2x2 x 2x3
    # A_sub1*B_sub2_1: Strassen at (0,2,4)
    strassen_block(0, 2, 4)
    # A_sub1*B_sub2_2: 2x2 x 2x1, schoolbook
    schoolbook_block(2, 2, 1, 0, 2, 6)
    
    # A_sub2*B_sub2: 1x2 x 2x3, schoolbook
    schoolbook_block(1, 2, 3, 2, 2, 4)
    
    # Now combine all terms into U, V, W matrices
    rank = len(terms)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
    
    # Convert to lists of lists with integer/half-integer entries
    U_list = [[int(x) if x == int(x) else x for x in row] for row in U.tolist()]
    V_list = [[int(x) if x == int(x) else x for x in row] for row in V.tolist()]
    W_list = [[int(x) if x == int(x) else x for x in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
