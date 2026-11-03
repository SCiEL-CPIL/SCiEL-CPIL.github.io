"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    terms = []
    
    def add_strassen(a_rows, c_cols, d_cols):
        def U(la, lc): return a_rows[la]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) -> C[0,0]+1, C[1,1]+1
        terms.append((
            [(U(0,0), 1), (U(1,1), 1)],
            [(V(0,0), 1), (V(1,1), 1)],
            [(W(0,0), 1), (W(1,1), 1)]
        ))
        # M2 = (A[1,0]+A[1,1])*B[0,0] -> C[1,0]+1
        terms.append((
            [(U(1,0), 1), (U(1,1), 1)],
            [(V(0,0), 1)],
            [(W(0,1), 1)]
        ))
        # M3 = A[0,0]*(B[0,1]-B[1,1]) -> C[0,1]+1
        terms.append((
            [(U(0,0), 1)],
            [(V(0,1), 1), (V(1,1), -1)],
            [(W(1,0), 1)]
        ))
        # M4 = A[1,1]*(B[1,0]-B[0,0]) -> C[0,0]+1, C[1,0]+1
        terms.append((
            [(U(1,1), 1)],
            [(V(1,0), 1), (V(0,0), -1)],
            [(W(0,0), 1), (W(0,1), 1)]
        ))
        # M5 = (A[0,0]+A[0,1])*B[1,1] -> C[0,0]-1, C[0,1]+1
        terms.append((
            [(U(0,0), 1), (U(0,1), 1)],
            [(V(1,1), 1)],
            [(W(0,0), -1), (W(1,0), 1)]
        ))
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1]) -> C[1,1]+1
        terms.append((
            [(U(1,0), 1), (U(0,0), -1)],
            [(V(0,0), 1), (V(0,1), 1)],
            [(W(1,1), 1)]
        ))
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1]) -> C[0,0]+1
        terms.append((
            [(U(0,1), 1), (U(1,1), -1)],
            [(V(1,0), 1), (V(1,1), 1)],
            [(W(0,0), 1)]
        ))
    
    def add_2x1x2(a_rows, c_cols, d_cols):
        c0 = c_cols[0]
        def U(la, lc): return a_rows[la]*m + c0
        def V(lc, ld): return c0*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        terms.append(([(U(0,0),1)], [(V(0,0),1)], [(W(0,0),1)]))
        terms.append(([(U(0,0),1)], [(V(0,1),1)], [(W(1,0),1)]))
        terms.append(([(U(1,0),1)], [(V(0,0),1)], [(W(0,1),1)]))
        terms.append(([(U(1,0),1)], [(V(0,1),1)], [(W(1,1),1)]))
    
    def add_1x2x2(a_rows, c_cols, d_cols):
        a0 = a_rows[0]
        def U(la, lc): return a0*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a0
        terms.append(([(U(0,0),1)], [(V(0,0),1)], [(W(0,0),1)]))
        terms.append(([(U(0,1),1)], [(V(1,0),1)], [(W(0,0),1)]))
        terms.append(([(U(0,0),1)], [(V(0,1),1)], [(W(1,0),1)]))
        terms.append(([(U(0,1),1)], [(V(1,1),1)], [(W(1,0),1)]))
    
    def add_1x1x2(a_rows, c_cols, d_cols):
        a0 = a_rows[0]
        c0 = c_cols[0]
        def U(la, lc): return a0*m + c0
        def V(lc, ld): return c0*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a0
        terms.append(([(U(0,0),1)], [(V(0,0),1)], [(W(0,0),1)]))
        terms.append(([(U(0,0),1)], [(V(0,1),1)], [(W(1,0),1)]))
    
    n_block_ranges = [(0, 2), (2, 3)]
    m_block_ranges = [(0, 2), (2, 4), (4, 5)]
    p_block_ranges = [(0, 2), (2, 4), (4, 6)]
    
    for ns, ne in n_block_ranges:
        for ms, me in m_block_ranges:
            for ps, pe in p_block_ranges:
                n_size = ne - ns
                m_size = me - ms
                p_size = pe - ps
                a_rows = list(range(ns, ne))
                c_cols = list(range(ms, me))
                d_cols = list(range(ps, pe))
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    add_strassen(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 1 and p_size == 2:
                    add_2x1x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 2 and p_size == 2:
                    add_1x2x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 1 and p_size == 2:
                    add_1x1x2(a_rows, c_cols, d_cols)
                else:
                    raise ValueError(f"Unhandled: {n_size}x{m_size}x{p_size}")
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for ci, (u_e, v_e, w_e) in enumerate(terms):
        for row, coeff in u_e:
            U[row][ci] += coeff
        for row, coeff in v_e:
            V[row][ci] += coeff
        for row, coeff in w_e:
            W[row][ci] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
