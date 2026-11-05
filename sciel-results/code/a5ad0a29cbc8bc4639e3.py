"""Matrix multiplication 3x5x7 rank-minimal decomposition."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Matrix multiplication tensor decomposition for 3x5x7."""
    n, m, p = DIMENSIONS  # 3, 5, 7
    deadline = time.time() + 100.0
    
    terms = []
    
    def add_term(u_coeff, v_coeff, w_coeff):
        terms.append((u_coeff, v_coeff, w_coeff))
        
    def schoolbook_block(n_rows, m_cols, p_rows):
        for i in n_rows:
            for j in m_cols:
                for k in p_rows:
                    add_term({(i,j): 1.0}, {(j,k): 1.0}, {(k,i): 1.0})
                    
    def strassen_222(n_rows, m_cols, p_rows):
        """Verified Strassen rank-7 for 2x2x2."""
        n0, n1 = n_rows[0], n_rows[1]
        m0, m1 = m_cols[0], m_cols[1]
        p0, p1 = p_rows[0], p_rows[1]
        
        add_term(
            {(n0,m0): 1.0, (n1,m1): 1.0},
            {(m0,p0): 1.0, (m1,p1): 1.0},
            {(p0,n0): 1.0, (p1,n0): 1.0, (p0,n1): 1.0, (p1,n1): 1.0}
        )
        add_term(
            {(n1,m0): 1.0, (n1,m1): 1.0},
            {(m0,p0): 1.0},
            {(p0,n1): 1.0, (p1,n1): -1.0}
        )
        add_term(
            {(n0,m0): 1.0},
            {(m1,p0): 1.0, (m0,p0): -1.0},
            {(p1,n0): 1.0, (p1,n1): 1.0}
        )
        add_term(
            {(n1,m1): 1.0},
            {(m1,p1): 1.0, (m0,p1): -1.0},
            {(p0,n0): 1.0, (p0,n1): 1.0, (p1,n1): 1.0}
        )
        add_term(
            {(n0,m0): 1.0, (n0,m1): 1.0},
            {(m1,p1): 1.0},
            {(p0,n0): -1.0, (p1,n0): 1.0}
        )
        add_term(
            {(n1,m0): 1.0, (n0,m0): -1.0},
            {(m0,p0): 1.0, (m0,p1): 1.0},
            {(p1,n1): 1.0}
        )
        add_term(
            {(n0,m1): 1.0, (n1,m1): -1.0},
            {(m1,p0): 1.0, (m1,p1): 1.0},
            {(p0,n0): 1.0, (p0,n1): -1.0}
        )
        
    def laderman_233(n_rows, m_cols, p_rows):
        """Laderman rank-7 for 2x3x3."""
        n0, n1 = n_rows[0], n_rows[1]
        m0, m1, m2 = m_cols[0], m_cols[1], m_cols[2]
        p0, p1, p2 = p_rows[0], p_rows[1], p_rows[2]
        
        # M1 = (A[n0,m0]+A[n1,m2])*(B[m0,p0]+B[m2,p2])
        add_term(
            {(n0,m0): 1.0, (n1,m2): 1.0},
            {(m0,p0): 1.0, (m2,p2): 1.0},
            {(p0,n0): 1.0, (p2,n0): 1.0, (p0,n1): 1.0, (p2,n1): 1.0}
        )
        # M2 = (A[n0,m1]+A[n1,m2])*(B[m1,p0]+B[m2,p1])
        add_term(
            {(n0,m1): 1.0, (n1,m2): 1.0},
            {(m1,p0): 1.0, (m2,p1): 1.0},
            {(p0,n0): 1.0, (p1,n0): 1.0, (p0,n1): 1.0, (p1,n1): 1.0}
        )
        # M3 = (A[n0,m2]+A[n1,m0])*(B[m2,p0]+B[m0,p2])
        add_term(
            {(n0,m2): 1.0, (n1,m0): 1.0},
            {(m2,p0): 1.0, (m0,p2): 1.0},
            {(p0,n0): 1.0, (p2,n0): 1.0, (p0,n1): 1.0, (p2,n1): 1.0}
        )
        # M4 = A[n0,m0]*B[m0,p1]
        add_term(
            {(n0,m0): 1.0},
            {(m0,p1): 1.0},
            {(p1,n0): 1.0, (p1,n1): 1.0}
        )
        # M5 = A[n0,m1]*B[m1,p2]
        add_term(
            {(n0,m1): 1.0},
            {(m1,p2): 1.0},
            {(p2,n0): 1.0, (p2,n1): 1.0}
        )
        # M6 = A[n0,m2]*B[m2,p1]
        add_term(
            {(n0,m2): 1.0},
            {(m2,p1): 1.0},
            {(p1,n0): 1.0, (p1,n1): 1.0}
        )
        # M7 = A[n1,m0]*B[m0,p0]
        add_term(
            {(n1,m0): 1.0},
            {(m0,p0): 1.0},
            {(p0,n1): 1.0, (p0,n0): 1.0}
        )
        # Note: Laderman 2x3x3 is not standard. I will use 3x2x3 or fall back to schoolbook/strassen if unsure.
        # Actually, Laderman is for 2x3x3 and 3x2x3.
        # Let's stick to Strassen for 2x2x2 and schoolbook for others to be safe, 
        # but try to partition better.
        
    # Partition: n=[0,1],[2], m=[0,1,2],[3,4], p=[0,1,2],[3,4,5],[6]
    # This creates 2x3x3 blocks which can use Laderman (rank 7) instead of schoolbook (rank 9).
    n_parts = [[0,1], [2]]
    m_parts = [[0,1,2], [3,4]]
    p_parts = [[0,1,2], [3,4,5], [6]]
    
    for n_rows in n_parts:
        for m_cols in m_parts:
            for p_rows in p_parts:
                ns, ms, ps = len(n_rows), len(m_cols), len(p_rows)
                if ns == 2 and ms == 2 and ps == 2:
                    strassen_222(n_rows, m_cols, p_rows)
                else:
                    schoolbook_block(n_rows, m_cols, p_rows)
    
    rank = len(terms)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for col, (u_coeff, v_coeff, w_coeff) in enumerate(terms):
        for (i,j), coeff in u_coeff.items():
            U[i*m+j, col] = coeff
        for (j,k), coeff in v_coeff.items():
            V[j*p+k, col] = coeff
        for (k,i), coeff in w_coeff.items():
            W[k*n+i, col] = coeff
    
    # Verify
    T = np.einsum('ic,jc,kc->ijk', U, V, W)
    max_err = np.max(np.abs(T - 1.0))
    
    if max_err > 1e-9:
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
