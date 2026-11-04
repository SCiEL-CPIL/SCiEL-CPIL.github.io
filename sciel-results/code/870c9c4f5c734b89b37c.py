"""Matrix multiplication 3x5 by 5x6: rank-minimal tensor decomposition."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Partition:
    # n (rows of A): [2, 1]
    # m (cols of A / rows of B): [2, 2, 1]
    # p (cols of B): [2, 2, 2]
    
    n_blocks = [(0, 2), (2, 3)]
    m_blocks = [(0, 2), (2, 4), (4, 5)]
    p_blocks = [(0, 2), (2, 4), (4, 6)]
    
    terms = []
    
    def add_strassen(a_rows, c_cols, d_cols):
        """Add 7 Strassen terms for a 2x2x2 block."""
        def U(la, lc):
            return a_rows[la] * m + c_cols[lc]
        def V(lc, ld):
            return c_cols[lc] * p + d_cols[ld]
        def W(ld, la):
            return d_cols[ld] * n + a_rows[la]
        
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # C[0,0] += 1, C[1,1] += 1
        terms.append((
            [(U(0,0), 1), (U(1,1), 1)],
            [(V(0,0), 1), (V(1,1), 1)],
            [(W(0,0), 1), (W(1,1), 1)]
        ))
        
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        # C[1,0] += 1, C[1,1] += -1
        terms.append((
            [(U(1,0), 1), (U(1,1), 1)],
            [(V(0,0), 1)],
            [(W(0,1), 1), (W(1,1), -1)]
        ))
        
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        # C[0,1] += 1, C[1,1] += 1
        terms.append((
            [(U(0,0), 1)],
            [(V(0,1), 1), (V(1,1), -1)],
            [(W(1,0), 1), (W(1,1), 1)]
        ))
        
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        # C[0,0] += 1, C[1,0] += 1
        terms.append((
            [(U(1,1), 1)],
            [(V(1,0), 1), (V(0,0), -1)],
            [(W(0,0), 1), (W(0,1), 1)]
        ))
        
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        # C[0,0] += -1, C[0,1] += 1
        terms.append((
            [(U(0,0), 1), (U(0,1), 1)],
            [(V(1,1), 1)],
            [(W(0,0), -1), (W(1,0), 1)]
        ))
        
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        # C[1,1] += 1
        terms.append((
            [(U(1,0), 1), (U(0,0), -1)],
            [(V(0,0), 1), (V(0,1), 1)],
            [(W(1,1), 1)]
        ))
        
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        # C[0,0] += 1
        terms.append((
            [(U(0,1), 1), (U(1,1), -1)],
            [(V(1,0), 1), (V(1,1), 1)],
            [(W(0,0), 1)]
        ))
    
    def add_schoolbook_2x2x1(a_rows, c_cols, d_cols):
        """2x2x1: A is 2x2, B is 2x1, C is 2x1. Rank 4."""
        def U(la, lc):
            return a_rows[la] * m + c_cols[lc]
        def V(lc, ld):
            return c_cols[lc] * p + d_cols[0]
        def W(ld, la):
            return d_cols[0] * n + a_rows[la]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,1), 1)], [(V(1,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(1,0), 1)], [(V(0,0), 1)], [(W(0,1), 1)]))
        terms.append(([(U(1,1), 1)], [(V(1,0), 1)], [(W(0,1), 1)]))
    
    def add_schoolbook_2x1x2(a_rows, c_cols, d_cols):
        """2x1x2: A is 2x1, B is 1x2, C is 2x2. Rank 4."""
        def U(la, lc):
            return a_rows[la] * m + c_cols[0]
        def V(lc, ld):
            return c_cols[0] * p + d_cols[ld]
        def W(ld, la):
            return d_cols[ld] * n + a_rows[la]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)]))
        terms.append(([(U(1,0), 1)], [(V(0,0), 1)], [(W(0,1), 1)]))
        terms.append(([(U(1,0), 1)], [(V(0,1), 1)], [(W(1,1), 1)]))
    
    def add_schoolbook_1x2x2(a_rows, c_cols, d_cols):
        """1x2x2: A is 1x2, B is 2x2, C is 1x2. Rank 4."""
        def U(la, lc):
            return a_rows[0] * m + c_cols[lc]
        def V(lc, ld):
            return c_cols[lc] * p + d_cols[ld]
        def W(ld, la):
            return d_cols[ld] * n + a_rows[0]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,1), 1)], [(V(1,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)]))
        terms.append(([(U(0,1), 1)], [(V(1,1), 1)], [(W(1,0), 1)]))
    
    def add_schoolbook_2x1x1(a_rows, c_cols, d_cols):
        """2x1x1: A is 2x1, B is 1x1, C is 2x1. Rank 2."""
        def U(la, lc):
            return a_rows[la] * m + c_cols[0]
        def V(lc, ld):
            return c_cols[0] * p + d_cols[0]
        def W(ld, la):
            return d_cols[0] * n + a_rows[la]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(1,0), 1)], [(V(0,0), 1)], [(W(0,1), 1)]))
    
    def add_schoolbook_1x2x1(a_rows, c_cols, d_cols):
        """1x2x1: A is 1x2, B is 2x1, C is 1x1. Rank 2."""
        def U(la, lc):
            return a_rows[0] * m + c_cols[lc]
        def V(lc, ld):
            return c_cols[lc] * p + d_cols[0]
        def W(ld, la):
            return d_cols[0] * n + a_rows[0]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,1), 1)], [(V(1,0), 1)], [(W(0,0), 1)]))
    
    def add_schoolbook_1x1x2(a_rows, c_cols, d_cols):
        """1x1x2: A is 1x1, B is 1x2, C is 1x2. Rank 2."""
        def U(la, lc):
            return a_rows[0] * m + c_cols[0]
        def V(lc, ld):
            return c_cols[0] * p + d_cols[ld]
        def W(ld, la):
            return d_cols[ld] * n + a_rows[0]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)]))
    
    def add_schoolbook_1x1x1(a_rows, c_cols, d_cols):
        """1x1x1: A is 1x1, B is 1x1, C is 1x1. Rank 1."""
        def U(la, lc):
            return a_rows[0] * m + c_cols[0]
        def V(lc, ld):
            return c_cols[0] * p + d_cols[0]
        def W(ld, la):
            return d_cols[0] * n + a_rows[0]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
    
    # Iterate over all block combinations
    for ni, (ns, ne) in enumerate(n_blocks):
        for mi, (ms, me) in enumerate(m_blocks):
            for pi, (ps, pe) in enumerate(p_blocks):
                n_size = ne - ns
                m_size = me - ms
                p_size = pe - ps
                
                a_rows = list(range(ns, ne))
                c_cols = list(range(ms, me))
                d_cols = list(range(ps, pe))
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    add_strassen(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 2 and p_size == 1:
                    add_schoolbook_2x2x1(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 1 and p_size == 2:
                    add_schoolbook_2x1x2(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 1 and p_size == 1:
                    add_schoolbook_2x1x1(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 2 and p_size == 2:
                    add_schoolbook_1x2x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 2 and p_size == 1:
                    add_schoolbook_1x2x1(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 1 and p_size == 2:
                    add_schoolbook_1x1x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 1 and p_size == 1:
                    add_schoolbook_1x1x1(a_rows, c_cols, d_cols)
                else:
                    raise ValueError(f"Unhandled block: {n_size}x{m_size}x{p_size}")
    
    # Build U, V, W matrices
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for ci, (u_entries, v_entries, w_entries) in enumerate(terms):
        for row, coeff in u_entries:
            U[row, ci] += coeff
        for row, coeff in v_entries:
            V[row, ci] += coeff
        for row, coeff in w_entries:
            W[row, ci] += coeff
    
    # Verify the decomposition
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_fb = np.zeros((n * m, rank_school))
        V_fb = np.zeros((m * p, rank_school))
        W_fb = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is correct."""
    rank = U.shape[1]
    
    # Build the multiplication tensor from U, V, W
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Check a few random entries
    import random
    random.seed(42)
    
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        
        # The tensor entry should be 1 if this is a valid multiplication tensor
        # Actually, for matrix multiplication, T[i,j,k] = 1 for all i,j,k
        # because C[i,k] = sum_j A[i,j] * B[j,k]
        
        val = 0.0
        for c in range(rank):
            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        if abs(val - 1.0) > 1e-10:
            return False
    
    return True
# EVOLVE-BLOCK-END
