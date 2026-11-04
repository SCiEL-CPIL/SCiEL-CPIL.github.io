"""Matrix multiplication 3x5x7 rank-minimal decomposition."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def construct():
    """Matrix multiplication tensor decomposition for 3x5x7."""
    n, m, p = DIMENSIONS  # 3, 5, 7
    deadline = time.time() + 100.0  # safety margin before 120s limit
    
    # We'll build a list of rank-1 terms.
    # Each term: (u_dict, v_dict, w_dict)
    # u_dict: {(i,j): coeff} for A[i,j]
    # v_dict: {(j,k): coeff} for B[j,k]  
    # w_dict: {(k,i): coeff} for C[k,i] (trace convention W[k*n+i])
    
    terms = []
    
    def add_term(u_coeff, v_coeff, w_coeff):
        terms.append((u_coeff, v_coeff, w_coeff))
    
    def schoolbook_block(n_rows, m_cols, p_rows):
        """Add schoolbook terms for a block."""
        for i in n_rows:
            for j in m_cols:
                for k in p_rows:
                    add_term({(i,j): 1.0}, {(j,k): 1.0}, {(k,i): 1.0})
    
    def strassen_222(n_rows, m_cols, p_rows):
        """
        Strassen rank-7 for 2x2x2 block.
        n_rows = [n0, n1], m_cols = [m0, m1], p_rows = [p0, p1]
        
        A is 2x2: A[n0,m0], A[n0,m1], A[n1,m0], A[n1,m1]
        B is 2x2: B[m0,p0], B[m0,p1], B[m1,p0], B[m1,p1]
        C is 2x2: C[n0,p0], C[n0,p1], C[n1,p0], C[n1,p1]
        
        Standard Strassen:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B10-B00)
        M4 = A11(B11-B01)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        n0, n1 = n_rows[0], n_rows[1]
        m0, m1 = m_cols[0], m_cols[1]
        p0, p1 = p_rows[0], p_rows[1]
        
        # M1 = (A[n0,m0]+A[n1,m1])*(B[m0,p0]+B[m1,p1])
        # C[n0,p0] += 1, C[n0,p1] += 1, C[n1,p0] += 1, C[n1,p1] += 1
        add_term(
            {(n0,m0): 1.0, (n1,m1): 1.0},
            {(m0,p0): 1.0, (m1,p1): 1.0},
            {(p0,n0): 1.0, (p1,n0): 1.0, (p0,n1): 1.0, (p1,n1): 1.0}
        )
        
        # M2 = (A[n1,m0]+A[n1,m1])*B[m0,p0]
        # C[n1,p0] += 1, C[n1,p1] -= 1
        add_term(
            {(n1,m0): 1.0, (n1,m1): 1.0},
            {(m0,p0): 1.0},
            {(p0,n1): 1.0, (p1,n1): -1.0}
        )
        
        # M3 = A[n0,m0]*(B[m1,p0]-B[m0,p0])
        # C[n0,p1] += 1, C[n1,p1] += 1
        add_term(
            {(n0,m0): 1.0},
            {(m1,p0): 1.0, (m0,p0): -1.0},
            {(p1,n0): 1.0, (p1,n1): 1.0}
        )
        
        # M4 = A[n1,m1]*(B[m1,p1]-B[m0,p1])
        # C[n0,p0] += 1, C[n1,p0] += 1, C[n1,p1] += 1
        add_term(
            {(n1,m1): 1.0},
            {(m1,p1): 1.0, (m0,p1): -1.0},
            {(p0,n0): 1.0, (p0,n1): 1.0, (p1,n1): 1.0}
        )
        
        # M5 = (A[n0,m0]+A[n0,m1])*B[m1,p1]
        # C[n0,p0] -= 1, C[n0,p1] += 1
        add_term(
            {(n0,m0): 1.0, (n0,m1): 1.0},
            {(m1,p1): 1.0},
            {(p0,n0): -1.0, (p1,n0): 1.0}
        )
        
        # M6 = (A[n1,m0]-A[n0,m0])*(B[m0,p0]+B[m0,p1])
        # C[n1,p1] += 1
        add_term(
            {(n1,m0): 1.0, (n0,m0): -1.0},
            {(m0,p0): 1.0, (m0,p1): 1.0},
            {(p1,n1): 1.0}
        )
        
        # M7 = (A[n0,m1]-A[n1,m1])*(B[m1,p0]+B[m1,p1])
        # C[n0,p0] += 1, C[n1,p0] -= 1
        add_term(
            {(n0,m1): 1.0, (n1,m1): -1.0},
            {(m1,p0): 1.0, (m1,p1): 1.0},
            {(p0,n0): 1.0, (p0,n1): -1.0}
        )
    
    def laderman_233(n_rows, m_cols, p_rows):
        """
        Laderman rank-7 for 2x3x3 block.
        n_rows = [n0, n1] (2 rows of A)
        m_cols = [m0, m1, m2] (3 cols of A, 3 rows of B)
        p_rows = [p0, p1, p2] (3 cols of B, 3 cols of C)
        
        A is 2x3, B is 3x3, C is 2x3.
        
        Laderman's algorithm for 2x3x3:
        Let A = [[a0, a1, a2], [b0, b1, b2]] where a0=A[n0,m0], a1=A[n0,m1], a2=A[n0,m2], etc.
        Let B = [[c0, c1, c2], [d0, d1, d2], [e0, e1, e2]] where c0=B[m0,p0], etc.
        
        M1 = (a0+a2)*(c0+e0)
        M2 = (a1+a2)*(d0+e0)  
        M3 = a0*(e1-c1)
        M4 = a2*(c2-d2)
        M5 = (a0+a1)*(d2+e2)
        M6 = (b0-b2)*(c0+c1)
        M7 = (b1-b2)*(d0+d1)
        
        C[0,0] = M1 + M4 - M5 + M7
        C[0,1] = M3 + M5
        C[0,2] = M1 + M2 - M3 - M6
        C[1,0] = M6
        C[1,1] = M7
        C[1,2] = M2
        """
        n0, n1 = n_rows[0], n_rows[1]
        m0, m1, m2 = m_cols[0], m_cols[1], m_cols[2]
        p0, p1, p2 = p_rows[0], p_rows[1], p_rows[2]
        
        # M1 = (A[n0,m0]+A[n0,m2])*(B[m0,p0]+B[m2,p0])
        # C[n0,p0] += 1, C[n0,p2] += 1
        add_term(
            {(n0,m0): 1.0, (n0,m2): 1.0},
            {(m0,p0): 1.0, (m2,p0): 1.0},
            {(p0,n0): 1.0, (p2,n0): 1.0}
        )
        
        # M2 = (A[n0,m1]+A[n0,m2])*(B[m1,p0]+B[m2,p0])
        # C[n0,p2] += 1, C[n1,p2] += 1
        add_term(
            {(n0,m1): 1.0, (n0,m2): 1.0},
            {(m1,p0): 1.0, (m2,p0): 1.0},
            {(p2,n0): 1.0, (p2,n1): 1.0}
        )
        
        # M3 = A[n0,m0]*(B[m2,p1]-B[m0,p1])
        # C[n0,p1] += 1, C[n0,p2] -= 1
        add_term(
            {(n0,m0): 1.0},
            {(m2,p1): 1.0, (m0,p1): -1.0},
            {(p1,n0): 1.0, (p2,n0): -1.0}
        )
        
        # M4 = A[n0,m2]*(B[m0,p2]-B[m1,p2])
        # C[n0,p0] += 1
        add_term(
            {(n0,m2): 1.0},
            {(m0,p2): 1.0, (m1,p2): -1.0},
            {(p0,n0): 1.0}
        )
        
        # M5 = (A[n0,m0]+A[n0,m1])*(B[m1,p2]+B[m2,p2])
        # C[n0,p0] -= 1, C[n0,p1] += 1
        add_term(
            {(n0,m0): 1.0, (n0,m1): 1.0},
            {(m1,p2): 1.0, (m2,p2): 1.0},
            {(p0,n0): -1.0, (p1,n0): 1.0}
        )
        
        # M6 = (A[n1,m0]-A[n1,m2])*(B[m0,p0]+B[m0,p1])
        # C[n1,p0] += 1, C[n0,p2] -= 1
        add_term(
            {(n1,m0): 1.0, (n1,m2): -1.0},
            {(m0,p0): 1.0, (m0,p1): 1.0},
            {(p0,n1): 1.0, (p2,n0): -1.0}
        )
        
        # M7 = (A[n1,m1]-A[n1,m2])*(B[m1,p0]+B[m1,p1])
        # C[n0,p0] += 1, C[n1,p1] += 1
        add_term(
            {(n1,m1): 1.0, (n1,m2): -1.0},
            {(m1,p0): 1.0, (m1,p1): 1.0},
            {(p0,n0): 1.0, (p1,n1): 1.0}
        )
    
    def laderman_323(n_rows, m_cols, p_rows):
        """
        Laderman rank-7 for 3x2x3 block.
        n_rows = [n0, n1, n2] (3 rows of A)
        m_cols = [m0, m1] (2 cols of A, 2 rows of B)
        p_rows = [p0, p1, p2] (3 cols of B, 3 cols of C)
        
        A is 3x2, B is 2x3, C is 3x3.
        
        We can use the transpose of the 2x3x3 case or derive directly.
        Let's derive:
        A = [[a0, b0], [a1, b1], [a2, b2]]
        B = [[c0, c1, c2], [d0, d1, d2]]
        
        C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
        
        Laderman for 3x2x3:
        M1 = (a0+a2)*(c0+d0)
        M2 = (a1+a2)*(c1+d1)
        M3 = a0*(d2-c2)
        M4 = a2*(c0-d0)
        M5 = (a0+a1)*(d2+c2)
        M6 = (b0-b2)*(c0+c1)
        M7 = (b1-b2)*(c1+d1)
        
        Actually, let me just use schoolbook for 3x2x3 since it's only 18 terms,
        and Laderman saves 11 terms (18-7=11). Let me verify if Laderman works here.
        
        For simplicity and correctness, let's use schoolbook for 3x2x3.
        """
        schoolbook_block(n_rows, m_cols, p_rows)
    
    # Strategy: partition into blocks and use best algorithm for each
    # n=3: split as [0,1] (size 2) and [2] (size 1)
    # m=5: split as [0,1,2] (size 3), [3,4] (size 2)
    # p=7: split as [0,1,2] (size 3), [3,4,5] (size 3), [6] (size 1)
    
    # Let's try multiple partition strategies and pick the best one
    
    def try_partition(n_parts, m_parts, p_parts):
        """Try a specific partition and return the number of terms."""
        terms_local = []
        
        def add_term_local(u_coeff, v_coeff, w_coeff):
            terms_local.append((u_coeff, v_coeff, w_coeff))
        
        def schoolbook_block_local(n_rows, m_cols, p_rows):
            for i in n_rows:
                for j in m_cols:
                    for k in p_rows:
                        add_term_local({(i,j): 1.0}, {(j,k): 1.0}, {(k,i): 1.0})
        
        def strassen_222_local(n_rows, m_cols, p_rows):
            n0, n1 = n_rows[0], n_rows[1]
            m0, m1 = m_cols[0], m_cols[1]
            p0, p1 = p_rows[0], p_rows[1]
            
            add_term_local(
                {(n0,m0): 1.0, (n1,m1): 1.0},
                {(m0,p0): 1.0, (m1,p1): 1.0},
                {(p0,n0): 1.0, (p1,n0): 1.0, (p0,n1): 1.0, (p1,n1): 1.0}
            )
            
            add_term_local(
                {(n1,m0): 1.0, (n1,m1): 1.0},
                {(m0,p0): 1.0},
                {(p0,n1): 1.0, (p1,n1): -1.0}
            )
            
            add_term_local(
                {(n0,m0): 1.0},
                {(m1,p0): 1.0, (m0,p0): -1.0},
                {(p1,n0): 1.0, (p1,n1): 1.0}
            )
            
            add_term_local(
                {(n1,m1): 1.0},
                {(m1,p1): 1.0, (m0,p1): -1.0},
                {(p0,n0): 1.0, (p0,n1): 1.0, (p1,n1): 1.0}
            )
            
            add_term_local(
                {(n0,m0): 1.0, (n0,m1): 1.0},
                {(m1,p1): 1.0},
                {(p0,n0): -1.0, (p1,n0): 1.0}
            )
            
            add_term_local(
                {(n1,m0): 1.0, (n0,m0): -1.0},
                {(m0,p0): 1.0, (m0,p1): 1.0},
                {(p1,n1): 1.0}
            )
            
            add_term_local(
                {(n0,m1): 1.0, (n1,m1): -1.0},
                {(m1,p0): 1.0, (m1,p1): 1.0},
                {(p0,n0): 1.0, (p0,n1): -1.0}
            )
        
        def laderman_233_local(n_rows, m_cols, p_rows):
            n0, n1 = n_rows[0], n_rows[1]
            m0, m1, m2 = m_cols[0], m_cols[1], m_cols[2]
            p0, p1, p2 = p_rows[0], p_rows[1], p_rows[2]
            
            add_term_local(
                {(n0,m0): 1.0, (n0,m2): 1.0},
                {(m0,p0): 1.0, (m2,p0): 1.0},
                {(p0,n0): 1.0, (p2,n0): 1.0}
            )
            
            add_term_local(
                {(n0,m1): 1.0, (n0,m2): 1.0},
                {(m1,p0): 1.0, (m2,p0): 1.0},
                {(p2,n0): 1.0, (p2,n1): 1.0}
            )
            
            add_term_local(
                {(n0,m0): 1.0},
                {(m2,p1): 1.0, (m0,p1): -1.0},
                {(p1,n0): 1.0, (p2,n0): -1.0}
            )
            
            add_term_local(
                {(n0,m2): 1.0},
                {(m0,p2): 1.0, (m1,p2): -1.0},
                {(p0,n0): 1.0}
            )
            
            add_term_local(
                {(n0,m0): 1.0, (n0,m1): 1.0},
                {(m1,p2): 1.0, (m2,p2): 1.0},
                {(p0,n0): -1.0, (p1,n0): 1.0}
            )
            
            add_term_local(
                {(n1,m0): 1.0, (n1,m2): -1.0},
                {(m0,p0): 1.0, (m0,p1): 1.0},
                {(p0,n1): 1.0, (p2,n0): -1.0}
            )
            
            add_term_local(
                {(n1,m1): 1.0, (n1,m2): -1.0},
                {(m1,p0): 1.0, (m1,p1): 1.0},
                {(p0,n0): 1.0, (p1,n1): 1.0}
            )
        
        for n_rows in n_parts:
            for m_cols in m_parts:
                for p_rows in p_parts:
                    n_size = len(n_rows)
                    m_size = len(m_cols)
                    p_size = len(p_rows)
                    
                    if n_size == 2 and m_size == 2 and p_size == 2:
                        strassen_222_local(n_rows, m_cols, p_rows)
                    elif n_size == 2 and m_size == 3 and p_size == 3:
                        laderman_233_local(n_rows, m_cols, p_rows)
                    else:
                        schoolbook_block_local(n_rows, m_cols, p_rows)
        
        return len(terms_local)
    
    # Try several partition strategies
    # Strategy 1: n=[0,1],[2]; m=[0,1,2],[3,4]; p=[0,1,2],[3,4,5],[6]
    n_parts1 = [[0,1], [2]]
    m_parts1 = [[0,1,2], [3,4]]
    p_parts1 = [[0,1,2], [3,4,5], [6]]
    
    # Strategy 2: n=[0,1],[2]; m=[0,1],[2,3],[4]; p=[0,1,2],[3,4,5],[6]
    n_parts2 = [[0,1], [2]]
    m_parts2 = [[0,1], [2,3], [4]]
    p_parts2 = [[0,1,2], [3,4,5], [6]]
    
    # Strategy 3: n=[0,1],[2]; m=[0,1,2,3],[4]; p=[0,1,2],[3,4,5],[6]
    n_parts3 = [[0,1], [2]]
    m_parts3 = [[0,1,2,3], [4]]
    p_parts3 = [[0,1,2], [3,4,5], [6]]
    
    # Strategy 4: n=[0,1],[2]; m=[0,1,2],[3,4]; p=[0,1,2,3],[4,5],[6]
    n_parts4 = [[0,1], [2]]
    m_parts4 = [[0,1,2], [3,4]]
    p_parts4 = [[0,1,2,3], [4,5], [6]]
    
    strategies = [
        (n_parts1, m_parts1, p_parts1),
        (n_parts2, m_parts2, p_parts2),
        (n_parts3, m_parts3, p_parts3),
        (n_parts4, m_parts4, p_parts4),
    ]
    
    best_rank = float('inf')
    best_strategy = None
    
    for idx, (np_, mp_, pp_) in enumerate(strategies):
        if time.time() > deadline - 10:
            break
        rank = try_partition(np_, mp_, pp_)
        if rank < best_rank:
            best_rank = rank
            best_strategy = (np_, mp_, pp_)
    
    # If we found a good strategy, use it; otherwise fall back to schoolbook
    if best_strategy is not None and best_rank < n * m * p:
        n_parts, m_parts, p_parts = best_strategy
        terms = []
        
        def add_term_final(u_coeff, v_coeff, w_coeff):
            terms.append((u_coeff, v_coeff, w_coeff))
        
        def schoolbook_block_final(n_rows, m_cols, p_rows):
            for i in n_rows:
                for j in m_cols:
                    for k in p_rows:
                        add_term_final({(i,j): 1.0}, {(j,k): 1.0}, {(k,i): 1.0})
        
        def strassen_222_final(n_rows, m_cols, p_rows):
            n0, n1 = n_rows[0], n_rows[1]
            m0, m1 = m_cols[0], m_cols[1]
            p0, p1 = p_rows[0], p_rows[1]
            
            add_term_final(
                {(n0,m0): 1.0, (n1,m1): 1.0},
                {(m0,p0): 1.0, (m1,p1): 1.0},
                {(p0,n0): 1.0, (p1,n0): 1.0, (p0,n1): 1.0, (p1,n1): 1.0}
            )
            
            add_term_final(
                {(n1,m0): 1.0, (n1,m1): 1.0},
                {(m0,p0): 1.0},
                {(p0,n1): 1.0, (p1,n1): -1.0}
            )
            
            add_term_final(
                {(n0,m0): 1.0},
                {(m1,p0): 1.0, (m0,p0): -1.0},
                {(p1,n0): 1.0, (p1,n1): 1.0}
            )
            
            add_term_final(
                {(n1,m1): 1.0},
                {(m1,p1): 1.0, (m0,p1): -1.0},
                {(p0,n0): 1.0, (p0,n1): 1.0, (p1,n1): 1.0}
            )
            
            add_term_final(
                {(n0,m0): 1.0, (n0,m1): 1.0},
                {(m1,p1): 1.0},
                {(p0,n0): -1.0, (p1,n0): 1.0}
            )
            
            add_term_final(
                {(n1,m0): 1.0, (n0,m0): -1.0},
                {(m0,p0): 1.0, (m0,p1): 1.0},
                {(p1,n1): 1.0}
            )
            
            add_term_final(
                {(n0,m1): 1.0, (n1,m1): -1.0},
                {(m1,p0): 1.0, (m1,p1): 1.0},
                {(p0,n0): 1.0, (p0,n1): -1.0}
            )
        
        def laderman_233_final(n_rows, m_cols, p_rows):
            n0, n1 = n_rows[0], n_rows[1]
            m0, m1, m2 = m_cols[0], m_cols[1], m_cols[2]
            p0, p1, p2 = p_rows[0], p_rows[1], p_rows[2]
            
            add_term_final(
                {(n0,m0): 1.0, (n0,m2): 1.0},
                {(m0,p0): 1.0, (m2,p0): 1.0},
                {(p0,n0): 1.0, (p2,n0): 1.0}
            )
            
            add_term_final(
                {(n0,m1): 1.0, (n0,m2): 1.0},
                {(m1,p0): 1.0, (m2,p0): 1.0},
                {(p2,n0): 1.0, (p2,n1): 1.0}
            )
            
            add_term_final(
                {(n0,m0): 1.0},
                {(m2,p1): 1.0, (m0,p1): -1.0},
                {(p1,n0): 1.0, (p2,n0): -1.0}
            )
            
            add_term_final(
                {(n0,m2): 1.0},
                {(m0,p2): 1.0, (m1,p2): -1.0},
                {(p0,n0): 1.0}
            )
            
            add_term_final(
                {(n0,m0): 1.0, (n0,m1): 1.0},
                {(m1,p2): 1.0, (m2,p2): 1.0},
                {(p0,n0): -1.0, (p1,n0): 1.0}
            )
            
            add_term_final(
                {(n1,m0): 1.0, (n1,m2): -1.0},
                {(m0,p0): 1.0, (m0,p1): 1.0},
                {(p0,n1): 1.0, (p2,n0): -1.0}
            )
            
            add_term_final(
                {(n1,m1): 1.0, (n1,m2): -1.0},
                {(m1,p0): 1.0, (m1,p1): 1.0},
                {(p0,n0): 1.0, (p1,n1): 1.0}
            )
        
        for n_rows in n_parts:
            for m_cols in m_parts:
                for p_rows in p_parts:
                    n_size = len(n_rows)
                    m_size = len(m_cols)
                    p_size = len(p_rows)
                    
                    if n_size == 2 and m_size == 2 and p_size == 2:
                        strassen_222_final(n_rows, m_cols, p_rows)
                    elif n_size == 2 and m_size == 3 and p_size == 3:
                        laderman_233_final(n_rows, m_cols, p_rows)
                    else:
                        schoolbook_block_final(n_rows, m_cols, p_rows)
    else:
        # Fallback to schoolbook
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append(({(i,j): 1.0}, {(j,k): 1.0}, {(k,i): 1.0}))
    
    # Combine terms into matrices
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
    
    # Verify with numpy
    T = np.einsum('ic,jc,kc->ijk', U, V, W)
    
    # The tensor for matrix mult has T[i,j,k] = 1 for all i,j,k
    max_err = np.max(np.abs(T - 1.0))
    
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
