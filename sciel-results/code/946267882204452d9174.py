"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
_start_time = time.time()

def construct():
    if time.time() - _start_time > 140:
        return {"U": [[0]], "V": [[0]], "W": [[0]]}
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    terms = []  # Each term: (u_entries, v_entries, w_entries)
    
    def add_strassen_2x2x2(a_rows, c_cols, d_cols):
        """Add 7 Strassen terms for a 2x2x2 block."""
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0, d1 = d_cols
        
        def U(la, lc): return a_rows[la]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        
        # M1 = (A[a0,c0]+A[a1,c1])*(B[c0,d0]+B[c1,d1])
        terms.append((
            [(U(0,0), 1), (U(1,1), 1)],
            [(V(0,0), 1), (V(1,1), 1)],
            [(W(0,0), 1), (W(1,1), 1)]
        ))
        
        # M2 = (A[a1,c0]+A[a1,c1])*B[c0,d0]
        terms.append((
            [(U(1,0), 1), (U(1,1), 1)],
            [(V(0,0), 1)],
            [(W(0,1), 1), (W(1,1), -1)]
        ))
        
        # M3 = A[a0,c0]*(B[c0,d1]-B[c1,d1])
        terms.append((
            [(U(0,0), 1)],
            [(V(0,1), 1), (V(1,1), -1)],
            [(W(1,0), 1), (W(1,1), 1)]
        ))
        
        # M4 = A[a1,c1]*(B[c1,d0]-B[c0,d0])
        terms.append((
            [(U(1,1), 1)],
            [(V(1,0), 1), (V(0,0), -1)],
            [(W(0,0), 1), (W(0,1), 1)]
        ))
        
        # M5 = (A[a0,c0]+A[a0,c1])*B[c1,d1]
        terms.append((
            [(U(0,0), 1), (U(0,1), 1)],
            [(V(1,1), 1)],
            [(W(0,0), -1), (W(1,0), 1)]
        ))
        
        # M6 = (A[a1,c0]-A[a0,c0])*(B[c0,d0]+B[c0,d1])
        terms.append((
            [(U(1,0), 1), (U(0,0), -1)],
            [(V(0,0), 1), (V(0,1), 1)],
            [(W(1,1), 1)]
        ))
        
        # M7 = (A[a0,c1]-A[a1,c1])*(B[c1,d0]+B[c1,d1])
        terms.append((
            [(U(0,1), 1), (U(1,1), -1)],
            [(V(1,0), 1), (V(1,1), 1)],
            [(W(0,0), 1)]
        ))
    
    def add_winograd_2x1x2(a_rows, c_col, d_cols):
        """Add 3 Winograd terms for a 2x1x2 block (A is 2x1, B is 1x2, C is 2x2)."""
        a0, a1 = a_rows
        c0 = c_col
        d0, d1 = d_cols
        
        u0 = a0*m + c0
        u1 = a1*m + c0
        v0 = c0*p + d0
        v1 = c0*p + d1
        w00 = d0*n + a0
        w01 = d1*n + a0
        w10 = d0*n + a1
        w11 = d1*n + a1
        
        # M1 = A[a0,c0]*(B[c0,d0]+B[c0,d1]) -> C[a0,d0]+=M1, C[a0,d1]+=M1
        terms.append((
            [(u0, 1)],
            [(v0, 1), (v1, 1)],
            [(w00, 1), (w01, 1)]
        ))
        
        # M2 = A[a1,c0]*(B[c0,d0]+B[c0,d1]) -> C[a1,d0]+=M2, C[a1,d1]+=M2
        terms.append((
            [(u1, 1)],
            [(v0, 1), (v1, 1)],
            [(w10, 1), (w11, 1)]
        ))
        
        # M3 = (A[a1,c0]-A[a0,c0])*B[c0,d1] -> C[a0,d1]-=M3, C[a1,d1]+=M3
        terms.append((
            [(u1, 1), (u0, -1)],
            [(v1, 1)],
            [(w01, -1), (w11, 1)]
        ))
    
    def add_winograd_1x2x2(a_row, c_cols, d_cols):
        """Add 3 Winograd terms for a 1x2x2 block (A is 1x2, B is 2x2, C is 1x2)."""
        a0 = a_row
        c0, c1 = c_cols
        d0, d1 = d_cols
        
        u0 = a0*m + c0
        u1 = a0*m + c1
        v00 = c0*p + d0
        v01 = c0*p + d1
        v10 = c1*p + d0
        v11 = c1*p + d1
        w0 = d0*n + a0
        w1 = d1*n + a0
        
        # M1 = A[a0,c0]*(B[c0,d0]+B[c0,d1]) -> C[a0,d0]+=M1, C[a0,d1]+=M1
        terms.append((
            [(u0, 1)],
            [(v00, 1), (v01, 1)],
            [(w0, 1), (w1, 1)]
        ))
        
        # M2 = A[a0,c1]*(B[c0,d0]+B[c0,d1]) -> C[a0,d0]+=M2, C[a0,d1]+=M2
        terms.append((
            [(u1, 1)],
            [(v00, 1), (v01, 1)],
            [(w0, 1), (w1, 1)]
        ))
        
        # M3 = (A[a0,c1]-A[a0,c0])*B[c0,d1] -> C[a0,d0]-=M3, C[a0,d1]+=M3
        terms.append((
            [(u1, 1), (u0, -1)],
            [(v01, 1)],
            [(w0, -1), (w1, 1)]
        ))
    
    def add_schoolbook_block(a_rows, c_cols, d_cols):
        """Add schoolbook terms for a block."""
        for a in a_rows:
            for c in c_cols:
                for d in d_cols:
                    u_idx = a * m + c
                    v_idx = c * p + d
                    w_idx = d * n + a
                    terms.append((
                        [(u_idx, 1)],
                        [(v_idx, 1)],
                        [(w_idx, 1)]
                    ))
    
    # Block partition: n=3 as [2,1], m=5 as [2,2,1], p=6 as [2,2,2]
    n_block_ranges = [(0, 2), (2, 3)]
    m_block_ranges = [(0, 2), (2, 4), (4, 5)]
    p_block_ranges = [(0, 2), (2, 4), (4, 6)]
    
    for ni, (ns, ne) in enumerate(n_block_ranges):
        for mi, (ms, me) in enumerate(m_block_ranges):
            for pi, (ps, pe) in enumerate(p_block_ranges):
                n_size = ne - ns
                m_size = me - ms
                p_size = pe - ps
                
                a_rows = list(range(ns, ne))
                c_cols = list(range(ms, me))
                d_cols = list(range(ps, pe))
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    add_strassen_2x2x2(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 1 and p_size == 2:
                    add_winograd_2x1x2(a_rows, c_cols[0], d_cols)
                elif n_size == 1 and m_size == 2 and p_size == 2:
                    add_winograd_1x2x2(a_rows[0], c_cols, d_cols)
                else:
                    add_schoolbook_block(a_rows, c_cols, d_cols)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for ci, (u_entries, v_entries, w_entries) in enumerate(terms):
        for row, coeff in u_entries:
            U[row][ci] += coeff
        for row, coeff in v_entries:
            V[row][ci] += coeff
        for row, coeff in w_entries:
            W[row][ci] += coeff
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(U_np[u_idx] * V_np[v_idx] * W_np[w_idx])
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err > 1e-10:
        # Fallback to schoolbook if decomposition is wrong
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    terms.append(([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)]))
        
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for ci, (u_entries, v_entries, w_entries) in enumerate(terms):
            for row, coeff in u_entries:
                U[row][ci] += coeff
            for row, coeff in v_entries:
                V[row][ci] += coeff
            for row, coeff in w_entries:
                W[row][ci] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
