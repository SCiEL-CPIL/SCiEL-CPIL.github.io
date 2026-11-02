import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = 2, 5, 5
    # Partition m and p into blocks of size 2,2,1
    m_blocks = [(0, 1), (2, 3), (4, 5)]  # block starts and ends
    p_blocks = [(0, 1), (2, 3), (4, 5)]
    
    # We'll build the decomposition using Strassen for 2x2x2 blocks
    # and schoolbook for the rest.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_strassen_2x2x2(m_start, p_start, i_offset):
        """Add a Strassen 2x2x2 decomposition for blocks at m_start and p_start.
        m_start, p_start: starting indices in m and p dimensions
        i_offset: not used in standard Strassen, but kept for interface
        """
        # Indices in m dimension: m_start, m_start+1
        # Indices in p dimension: p_start, p_start+1
        # In i dimension: 0, 1 (both rows of the 2x5 matrix)
        
        # Strassen formulas for C = A * B where A is 2x2, B is 2x2:
        # A = [[A00, A01], [A10, A11]] (rows i=0,1; cols m_start, m_start+1)
        # B = [[B00, B01], [B10, B11]] (rows m_start, m_start+1; cols p_start, p_start+1)
        # C = [[C00, C01], [C10, C11]] (rows i=0,1; cols p_start, p_start+1)
        
        # The 7 Strassen multiplications:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # For each multiplication Mr, we need:
        # U entries: linear combination of A elements
        # V entries: linear combination of B elements
        # W entries: which output elements receive this term
        
        # Let's define the 7 terms:
        # Term 0: M1 = (A00+A11)(B00+B11)
        # Term 1: M2 = (A10+A11)B00
        # Term 2: M3 = A00(B10-B11)
        # Term 3: M4 = A11(B01-B10)
        # Term 4: M5 = (A00+A01)B11
        # Term 5: M6 = (A10-A00)(B00+B10)
        # Term 6: M7 = (A01-A11)(B01+B11)
        
        # For each term, we need to specify:
        # - Which (i, m) entries in U are non-zero
        # - Which (m, p) entries in V are non-zero
        # - Which (p, i) entries in W are non-zero
        
        # The output is C[i, p] for i in {0,1}, p in {p_start, p_start+1}
        
        # Let's build this carefully.
        
        # Term M1 = (A00+A11)(B00+B11)
        # A00 = A[0, m_start], A11 = A[1, m_start+1]
        # B00 = B[m_start, p_start], B11 = B[m_start+1, p_start+1]
        # M1 contributes to C00 and C11
        # C00 = C[0, p_start], C11 = C[1, p_start+1]
        
        # For M1:
        # U: A00 + A11 -> U[0*m_start + 0, r] += 1, U[1*m_start + 1, r] += 1
        # Wait, U is indexed by (i, m) where i is the row of the first matrix
        # U[i*m + m_idx, r]
        
        # Actually, let me re-index. U has shape (n*m, rank) = (10, rank)
        # U[i*m + j, r] where i in 0..n-1, j in 0..m-1
        
        # V has shape (m*p, rank) = (25, rank)
        # V[j*p + k, r] where j in 0..m-1, k in 0..p-1
        
        # W has shape (p*n, rank) = (10, rank)
        # W[k*n + i, r] where k in 0..p-1, i in 0..n-1
        
        # For M1 = (A[0, m_start] + A[1, m_start+1]) * (B[m_start, p_start] + B[m_start+1, p_start+1])
        # U contributions: U[0*m + m_start, r] += 1, U[1*m + (m_start+1), r] += 1
        # V contributions: V[m_start*p + p_start, r] += 1, V[(m_start+1)*p + (p_start+1), r] += 1
        # W contributions: 
        #   M1 contributes +1 to C[0, p_start] -> W[p_start*n + 0, r] += 1
        #   M1 contributes +1 to C[1, p_start+1] -> W[(p_start+1)*n + 1, r] += 1
        
        def add_term(u_indices, v_indices, w_indices):
            """Add a single multiplication term.
            u_indices: list of (idx, coeff) for U
            v_indices: list of (idx, coeff) for V
            w_indices: list of (idx, coeff) for W
            """
            if not u_indices or not v_indices or not w_indices:
                return
            r = len(U_list)
            # Create U, V, W vectors of zeros
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            for idx, coeff in u_indices:
                u_vec[idx] += coeff
            for idx, coeff in v_indices:
                v_vec[idx] += coeff
            for idx, coeff in w_indices:
                w_vec[idx] += coeff
            
            U_list.append(u_vec)
            V_list.append(v_vec)
            W_list.append(w_vec)
        
        # M1 = (A[0,m_start] + A[1,m_start+1]) * (B[m_start,p_start] + B[m_start+1,p_start+1])
        add_term(
            [(0 * m + m_start, 1.0), (1 * m + (m_start + 1), 1.0)],
            [(m_start * p + p_start, 1.0), ((m_start + 1) * p + (p_start + 1), 1.0)],
            [(p_start * n + 0, 1.0), ((p_start + 1) * n + 1, 1.0)]
        )
        
        # M2 = (A[1,m_start] + A[1,m_start+1]) * B[m_start,p_start]
        add_term(
            [(1 * m + m_start, 1.0), (1 * m + (m_start + 1), 1.0)],
            [(m_start * p + p_start, 1.0)],
            [(p_start * n + 1, 1.0)]
        )
        
        # M3 = A[0,m_start] * (B[m_start+1,p_start] - B[m_start+1,p_start+1])
        add_term(
            [(0 * m + m_start, 1.0)],
            [((m_start + 1) * p + p_start, 1.0), ((m_start + 1) * p + (p_start + 1), -1.0)],
            [(p_start * n + 0, 1.0), (p_start * n + 1, 1.0)]
        )
        
        # M4 = A[1,m_start+1] * (B[m_start,p_start+1] - B[m_start+1,p_start])
        add_term(
            [(1 * m + (m_start + 1), 1.0)],
            [(m_start * p + (p_start + 1), 1.0), ((m_start + 1) * p + p_start, -1.0)],
            [(p_start * n + 0, 1.0), (p_start * n + 1, 1.0)]
        )
        
        # M5 = (A[0,m_start] + A[0,m_start+1]) * B[m_start+1,p_start+1]
        add_term(
            [(0 * m + m_start, 1.0), (0 * m + (m_start + 1), 1.0)],
            [((m_start + 1) * p + (p_start + 1), 1.0)],
            [(p_start * n + 0, -1.0), (p_start * n + 1, 1.0)]
        )
        
        # M6 = (A[1,m_start] - A[0,m_start]) * (B[m_start,p_start] + B[m_start,p_start+1])
        add_term(
            [(1 * m + m_start, 1.0), (0 * m + m_start, -1.0)],
            [(m_start * p + p_start, 1.0), (m_start * p + (p_start + 1), 1.0)],
            [(p_start * n + 1, 1.0)]
        )
        
        # M7 = (A[0,m_start+1] - A[1,m_start+1]) * (B[m_start+1,p_start] + B[m_start+1,p_start+1])
        add_term(
            [(0 * m + (m_start + 1), 1.0), (1 * m + (m_start + 1), -1.0)],
            [((m_start + 1) * p + p_start, 1.0), ((m_start + 1) * p + (p_start + 1), 1.0)],
            [(p_start * n + 0, -1.0), (p_start * n + 1, 1.0)]
        )
    
    def add_schoolbook_block(m_start, m_end, p_start, p_end):
        """Add schoolbook decomposition for a block of size (m_end-m_start) x (p_end-p_start)."""
        for i in range(n):
            for j in range(m_start, m_end):
                for k in range(p_start, p_end):
                    r = len(U_list)
                    u_vec = [0.0] * (n * m)
                    v_vec = [0.0] * (m * p)
                    w_vec = [0.0] * (p * n)
                    u_vec[i * m + j] = 1.0
                    v_vec[j * p + k] = 1.0
                    w_vec[k * n + i] = 1.0
                    U_list.append(u_vec)
                    V_list.append(v_vec)
                    W_list.append(w_vec)
    
    # For each pair of m-blocks and p-blocks:
    for mi, (m_start, m_end) in enumerate(m_blocks):
        for pi, (p_start, p_end) in enumerate(p_blocks):
            m_size = m_end - m_start
            p_size = p_end - p_start
            
            if m_size == 2 and p_size == 2:
                # Use Strassen
                add_strassen_2x2x2(m_start, p_start, 0)
            else:
                # Use schoolbook
                add_schoolbook_block(m_start, m_end, p_start, p_end)
    
    rank = len(U_list)
    U = U_list
    V = V_list
    W = W_list
    
    return {"U": U, "V": V, "W": W}
