"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x2 * 2x7 matrix multiplication.
    Split the 7 columns into 2+2+2+1.
    Use Strassen (rank 7) for each 2x2 * 2x2 block.
    Use schoolbook (rank 2) for the 2x2 * 2x1 block.
    Total rank: 7*3 + 2 = 23.
    """
    n, m, p = DIMENSIONS  # 2, 2, 7
    
    U = []
    V = []
    W = []
    
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
    
    def strassen_block(cs):
        """Add 7 Strassen terms for 2x2 * 2x2 block at columns cs, cs+1."""
        # M1 = (a11+a22)(b11+b22)
        u = [1, 0, 0, 1]
        v = [0]*14; v[cs] = 1; v[7+cs+1] = 1
        w = [0]*14; w[cs*2] = 1; w[(cs+1)*2+1] = 1
        add_term(u, v, w)
        
        # M2 = (a21+a22)b11
        u = [0, 0, 1, 1]
        v = [0]*14; v[cs] = 1
        w = [0]*14; w[cs*2+1] = 1; w[(cs+1)*2+1] = -1
        add_term(u, v, w)
        
        # M3 = a11(b21-b22)
        u = [1, 0, 0, 0]
        v = [0]*14; v[7+cs] = 1; v[7+cs+1] = -1
        w = [0]*14; w[(cs+1)*2] = 1; w[(cs+1)*2+1] = 1
        add_term(u, v, w)
        
        # M4 = a22(b12-b11)
        u = [0, 0, 0, 1]
        v = [0]*14; v[cs+1] = 1; v[cs] = -1
        w = [0]*14; w[cs*2] = 1; w[cs*2+1] = 1
        add_term(u, v, w)
        
        # M5 = (a11+a12)b22
        u = [1, 1, 0, 0]
        v = [0]*14; v[7+cs+1] = 1
        w = [0]*14; w[cs*2] = -1; w[(cs+1)*2] = 1
        add_term(u, v, w)
        
        # M6 = (a21-a11)(b11+b12)
        u = [0, 0, 1, -1]
        v = [0]*14; v[cs] = 1; v[cs+1] = 1
        w = [0]*14; w[(cs+1)*2+1] = 1
        add_term(u, v, w)
        
        # M7 = (a12-a22)(b21+b22)
        u = [0, 1, 0, -1]
        v = [0]*14; v[7+cs] = 1; v[7+cs+1] = 1
        w = [0]*14; w[cs*2] = 1
        add_term(u, v, w)
    
    def schoolbook_2x2x1(cs):
        """Add 2 schoolbook terms for 2x2 * 2x1 block at column cs."""
        for i in range(2):
            for k in range(2):
                u = [0]*4
                u[i*2+k] = 1
                v = [0]*14
                v[k*7+cs] = 1
                w = [0]*14
                w[cs*2+i] = 1
                add_term(u, v, w)
    
    # Apply to columns 0-1, 2-3, 4-5, and 6
    strassen_block(0)
    strassen_block(2)
    strassen_block(4)
    schoolbook_2x2x1(6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
