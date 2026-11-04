"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Matrix multiplication tensor decomposition for 3x5x7."""
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Splits:
    # n: [0,1] (2), [2] (1)
    # m: [0,1] (2), [2,3] (2), [4] (1)
    # p: [0,1,2] (3), [3,4,5,6] (4)
    
    n_splits = [([0,1], 2), ([2], 1)]
    m_splits = [([0,1], 2), ([2,3], 2), ([4], 1)]
    p_splits = [([0,1,2], 3), ([3,4,5,6], 4)]
    
    terms = []
    
    def add_term(u_coeff, v_coeff, w_coeff):
        terms.append((u_coeff, v_coeff, w_coeff))
        
    def strassen_222_terms(n_rows, m_cols, p_rows):
        """Add Strassen terms for 2x2x2 block defined by indices."""
        # U, V, W for local 2x2x2
        # p1 = (a+d)(e+h)
        # p2 = (c+d)e
        # p3 = a(f-h)
        # p4 = d(g-e)
        # p5 = (a+b)h
        # p6 = (c-d)(f+g)
        # p7 = (b-d)(e+h)
        
        # c=0: p1
        # A: (0,0), (1,1). B: (0,0), (1,1). C: (0,0), (1,1).
        u = {(n_rows[0], m_cols[0]): 1, (n_rows[1], m_cols[1]): 1}
        v = {(m_cols[0], p_rows[0]): 1, (m_cols[1], p_rows[1]): 1}
        w = {(p_rows[0], n_rows[0]): 1, (p_rows[1], n_rows[1]): 1}
        add_term(u, v, w)
        
        # c=1: p2
        # A: (1,0), (1,1). B: (0,0). C: (1,0).
        u = {(n_rows[1], m_cols[0]): 1, (n_rows[1], m_cols[1]): 1}
        v = {(m_cols[0], p_rows[0]): 1}
        w = {(p_rows[0], n_rows[1]): 1}
        add_term(u, v, w)
        
        # c=2: p3
        # A: (0,0). B: (0,1), (1,1). C: (0,1), (1,1).
        u = {(n_rows[0], m_cols[0]): 1}
        v = {(m_cols[0], p_rows[1]): 1, (m_cols[1], p_rows[1]): -1}
        w = {(p_rows[1], n_rows[0]): 1, (p_rows[1], n_rows[1]): 1}
        add_term(u, v, w)
        
        # c=3: p4
        # A: (1,1). B: (1,0), (0,0). C: (0,0), (1,0).
        u = {(n_rows[1], m_cols[1]): 1}
        v = {(m_cols[1], p_rows[0]): 1, (m_cols[0], p_rows[0]): -1}
        w = {(p_rows[0], n_rows[0]): -1, (p_rows[0], n_rows[1]): 1}
        add_term(u, v, w)
        
        # c=4: p5
        # A: (0,0), (0,1). B: (1,1). C: (0,0), (0,1).
        u = {(n_rows[0], m_cols[0]): 1, (n_rows[0], m_cols[1]): 1}
        v = {(m_cols[1], p_rows[1]): 1}
        w = {(p_rows[0], n_rows[0]): 1, (p_rows[1], n_rows[0]): 1}
        add_term(u, v, w)
        
        # c=5: p6
        # A: (1,0), (1,1). B: (0,1), (1,0). C: (1,1).
        u = {(n_rows[1], m_cols[0]): 1, (n_rows[1], m_cols[1]): -1}
        v = {(m_cols[0], p_rows[1]): 1, (m_cols[1], p_rows[0]): 1}
        w = {(p_rows[1], n_rows[1]): 1}
        add_term(u, v, w)
        
        # c=6: p7
        # A: (0,1), (1,1). B: (0,0), (1,1). C: (0,0).
        u = {(n_rows[0], m_cols[1]): 1, (n_rows[1], m_cols[1]): -1}
        v = {(m_cols[0], p_rows[0]): 1, (m_cols[1], p_rows[1]): 1}
        w = {(p_rows[0], n_rows[0]): -1}
        add_term(u, v, w)

    def schoolbook_terms(n_rows, m_cols, p_rows):
        """Add schoolbook terms for a block."""
        for i in n_rows:
            for j in m_cols:
                for k in p_rows:
                    u = {(i,j): 1}
                    v = {(j,k): 1}
                    w = {(k,i): 1}
                    add_term(u, v, w)

    for n_idx, (n_rows, n_size) in enumerate(n_splits):
        for m_idx, (m_cols, m_size) in enumerate(m_splits):
            for p_idx, (p_rows, p_size) in enumerate(p_splits):
                if n_size == 2 and m_size == 2:
                    if p_size == 3:
                        # Split p into 2+1
                        p_sub1 = p_rows[:2]
                        p_sub2 = p_rows[2:]
                        strassen_222_terms(n_rows, m_cols, p_sub1)
                        schoolbook_terms(n_rows, m_cols, p_sub2)
                    elif p_size == 4:
                        # Split p into 2+2
                        p_sub1 = p_rows[:2]
                        p_sub2 = p_rows[2:4]
                        strassen_222_terms(n_rows, m_cols, p_sub1)
                        strassen_222_terms(n_rows, m_cols, p_sub2)
                else:
                    schoolbook_terms(n_rows, m_cols, p_rows)

    rank = len(terms)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for c, (u_coeff, v_coeff, w_coeff) in enumerate(terms):
        for (i,j), coeff in u_coeff.items():
            U[i*m+j, c] = coeff
        for (j,k), coeff in v_coeff.items():
            V[j*p+k, c] = coeff
        for (k,i), coeff in w_coeff.items():
            W[k*n+i, c] = coeff
            
    # Verify
    if time.time() < deadline:
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-9:
            # Fallback to schoolbook
            rank = n * m * p
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
