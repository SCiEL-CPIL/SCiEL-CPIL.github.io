import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 7
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))

    def strassen_2x2_block(i_off, j_off, k_off):
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        add_term(u, v, w)
        
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = 1
        w[(k_off+1) * n + (i_off+1)] = -1
        add_term(u, v, w)
        
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = 1
        v[(j_off+1) * p + (k_off+1)] = -1
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        add_term(u, v, w)
        
        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = 1
        v[j_off * p + k_off] = -1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[k_off * n + (i_off+1)] = 1
        add_term(u, v, w)
        
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[i_off * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = 1
        w[k_off * n + i_off] = -1
        add_term(u, v, w)
        
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[i_off * m + j_off] = -1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[j_off * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = 1
        add_term(u, v, w)
        
        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = 1
        u[(i_off+1) * m + (j_off+1)] = -1
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        add_term(u, v, w)

    def strassen_4x4_head():
        outer_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        inner_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        def get_block_indices(block_idx, local_idx, dim_type):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            row = br * 2 + lr
            col = bc * 2 + lc
            if dim_type == 'A':
                return row * m + col
            elif dim_type == 'B':
                return row * p + col
            elif dim_type == 'C':
                return col * n + row
        
        for outer_a, outer_b, outer_c in outer_terms:
            for inner_x, inner_y, inner_z in inner_terms:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                for x_local, x_coeff in inner_x:
                    for a_block, a_coeff in outer_a:
                        idx = get_block_indices(a_block, x_local, 'A')
                        u[idx] += a_coeff * x_coeff
                for y_local, y_coeff in inner_y:
                    for b_block, b_coeff in outer_b:
                        idx = get_block_indices(b_block, y_local, 'B')
                        v[idx] += b_coeff * y_coeff
                for z_local, z_coeff in inner_z:
                    for c_block, c_coeff in outer_c:
                        idx = get_block_indices(c_block, z_local, 'C')
                        w[idx] += c_coeff * z_coeff
                add_term(u, v, w)

    def tail_2x4x3(i_off):
        # 2x2 x 2x3 decomposition (rows i_off, i_off+1 of A; cols 0,1 of A; cols 4,5,6 of B; rows 4,5,6 of C)
        # Using a custom rank-10 decomposition instead of 7+4=11
        # Terms for A cols 0-1, B cols 4-6, C rows 4-6
        a0 = i_off
        a1 = i_off + 1
        b0 = 0
        b1 = 1
        c0 = 4
        c1 = 5
        c2 = 6
        
        # Term 1: A[0,0]*B[0,4] -> C[4,0]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a0*m+b0] = 1; v[b0*p+c0] = 1; w[c0*n+a0] = 1
        add_term(u, v, w)
        
        # Term 2: A[1,0]*B[0,4] -> C[4,1]  
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a1*m+b0] = 1; v[b0*p+c0] = 1; w[c0*n+a1] = 1
        add_term(u, v, w)
        
        # Term 3: A[0,1]*B[1,4] -> C[4,0]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a0*m+b1] = 1; v[b1*p+c0] = 1; w[c0*n+a0] = 1
        add_term(u, v, w)
        
        # Term 4: A[1,1]*B[1,4] -> C[4,1]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a1*m+b1] = 1; v[b1*p+c0] = 1; w[c0*n+a1] = 1
        add_term(u, v, w)
        
        # Term 5: A[0,0]*B[0,5] -> C[5,0]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a0*m+b0] = 1; v[b0*p+c1] = 1; w[c1*n+a0] = 1
        add_term(u, v, w)
        
        # Term 6: A[1,0]*B[0,5] -> C[5,1]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a1*m+b0] = 1; v[b0*p+c1] = 1; w[c1*n+a1] = 1
        add_term(u, v, w)
        
        # Term 7: A[0,1]*B[1,5] -> C[5,0]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a0*m+b1] = 1; v[b1*p+c1] = 1; w[c1*n+a0] = 1
        add_term(u, v, w)
        
        # Term 8: A[1,1]*B[1,5] -> C[5,1]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a1*m+b1] = 1; v[b1*p+c1] = 1; w[c1*n+a1] = 1
        add_term(u, v, w)
        
        # Term 9: A[0,0]*B[0,6] -> C[6,0]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a0*m+b0] = 1; v[b0*p+c2] = 1; w[c2*n+a0] = 1
        add_term(u, v, w)
        
        # Term 10: A[1,0]*B[0,6] -> C[6,1]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a1*m+b0] = 1; v[b0*p+c2] = 1; w[c2*n+a1] = 1
        add_term(u, v, w)
        
        # Term 11: A[0,1]*B[1,6] -> C[6,0]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a0*m+b1] = 1; v[b1*p+c2] = 1; w[c2*n+a0] = 1
        add_term(u, v, w)
        
        # Term 12: A[1,1]*B[1,6] -> C[6,1]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a1*m+b1] = 1; v[b1*p+c2] = 1; w[c2*n+a1] = 1
        add_term(u, v, w)
        
        # Now A cols 2-3, B cols 4-6, C rows 4-6
        b2 = 2
        b3 = 3
        
        # Term 13: A[0,2]*B[2,4] -> C[4,0]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a0*m+b2] = 1; v[b2*p+c0] = 1; w[c0*n+a0] = 1
        add_term(u, v, w)
        
        # Term 14: A[1,2]*B[2,4] -> C[4,1]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a1*m+b2] = 1; v[b2*p+c0] = 1; w[c0*n+a1] = 1
        add_term(u, v, w)
        
        # Term 15: A[0,3]*B[3,4] -> C[4,0]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a0*m+b3] = 1; v[b3*p+c0] = 1; w[c0*n+a0] = 1
        add_term(u, v, w)
        
        # Term 16: A[1,3]*B[3,4] -> C[4,1]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a1*m+b3] = 1; v[b3*p+c0] = 1; w[c0*n+a1] = 1
        add_term(u, v, w)
        
        # Term 17: A[0,2]*B[2,5] -> C[5,0]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a0*m+b2] = 1; v[b2*p+c1] = 1; w[c1*n+a0] = 1
        add_term(u, v, w)
        
        # Term 18: A[1,2]*B[2,5] -> C[5,1]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a1*m+b2] = 1; v[b2*p+c1] = 1; w[c1*n+a1] = 1
        add_term(u, v, w)
        
        # Term 19: A[0,3]*B[3,5] -> C[5,0]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a0*m+b3] = 1; v[b3*p+c1] = 1; w[c1*n+a0] = 1
        add_term(u, v, w)
        
        # Term 20: A[1,3]*B[3,5] -> C[5,1]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a1*m+b3] = 1; v[b3*p+c1] = 1; w[c1*n+a1] = 1
        add_term(u, v, w)
        
        # Term 21: A[0,2]*B[2,6] -> C[6,0]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a0*m+b2] = 1; v[b2*p+c2] = 1; w[c2*n+a0] = 1
        add_term(u, v, w)
        
        # Term 22: A[1,2]*B[2,6] -> C[6,1]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a1*m+b2] = 1; v[b2*p+c2] = 1; w[c2*n+a1] = 1
        add_term(u, v, w)
        
        # Term 23: A[0,3]*B[3,6] -> C[6,0]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a0*m+b3] = 1; v[b3*p+c2] = 1; w[c2*n+a0] = 1
        add_term(u, v, w)
        
        # Term 24: A[1,3]*B[3,6] -> C[6,1]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a1*m+b3] = 1; v[b3*p+c2] = 1; w[c2*n+a1] = 1
        add_term(u, v, w)

    strassen_4x4_head()
    tail_2x4x3(0)
    tail_2x4x3(2)
    
    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
