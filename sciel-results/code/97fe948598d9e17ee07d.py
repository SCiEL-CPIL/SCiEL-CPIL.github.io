import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 80
    rank = 80
    
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a block
    def add_block(i_start, j_start, k_start, i_end, j_end, k_end, block_type):
        nonlocal col
        di = i_end - i_start
        dj = j_end - j_start
        dk = k_end - k_start
        
        if block_type == 'strassen_222':
            # di=2, dj=2, dk=2
            # Strassen vectors
            # u: (i*m+j), v: (j*p+k), w: (k*n+i)
            # Local indices: i in 0..1, j in 0..1, k in 0..1
            # Global indices:
            # u_idx = (i_start + i) * m + (j_start + j)
            # v_idx = (j_start + j) * p + (k_start + k)
            # w_idx = (k_start + k) * n + (i_start + i)
            
            # M1: (A00+A11)(B00+B11) -> C00, C11
            # u1: (0,0)=1, (1,1)=1
            # v1: (0,0)=1, (1,1)=1
            # w1: (0,0)=1, (1,1)=1
            u1 = {0: 1.0, 3: 1.0} # 0:0,0 3:1,1
            v1 = {0: 1.0, 3: 1.0}
            w1 = {0: 1.0, 3: 1.0}
            
            # M2: (A10+A11)B00 -> C10
            # u2: (1,0)=1, (1,1)=1
            # v2: (0,0)=1
            # w2: (1,0)=1
            u2 = {2: 1.0, 3: 1.0} # 2:1,0 3:1,1
            v2 = {0: 1.0}
            w2 = {2: 1.0} # 2:1,0 (k=1, i=0)
            
            # M3: A00(B01-B11) -> C01
            # u3: (0,0)=1
            # v3: (0,1)=1, (1,1)=-1
            # w3: (0,1)=1
            u3 = {0: 1.0}
            v3 = {1: 1.0, 3: -1.0} # 1:0,1 3:1,1
            w3 = {1: 1.0} # 1:0,1 (k=0, i=1)
            
            # M4: A11(B10-B00) -> C10, C11
            # u4: (1,1)=1
            # v4: (1,0)=1, (0,0)=-1
            # w4: (1,0)=1, (1,1)=1
            u4 = {3: 1.0}
            v4 = {2: 1.0, 0: -1.0} # 2:1,0 0:0,0
            w4 = {2: 1.0, 3: 1.0} # 2:1,0 3:1,1
            
            # M5: (A00+A01)B11 -> C01, C11
            # u5: (0,0)=1, (0,1)=1
            # v5: (1,1)=1
            # w5: (0,1)=1, (1,1)=1
            u5 = {0: 1.0, 1: 1.0}
            v5 = {3: 1.0}
            w5 = {1: 1.0, 3: 1.0}
            
            # M6: (A10-A00)(B00+B01) -> C11
            # u6: (1,0)=1, (0,0)=-1
            # v6: (0,0)=1, (0,1)=1
            # w6: (1,1)=1
            u6 = {2: 1.0, 0: -1.0}
            v6 = {0: 1.0, 1: 1.0}
            w6 = {3: 1.0}
            
            # M7: (A01-A11)(B10+B11) -> C00, C10
            # u7: (0,1)=1, (1,1)=-1
            # v7: (1,0)=1, (1,1)=1
            # w7: (0,0)=1, (1,0)=1
            u7 = {1: 1.0, 3: -1.0}
            v7 = {2: 1.0, 3: 1.0}
            w7 = {0: 1.0, 2: 1.0}
            
            terms = [
                (u1, v1, w1),
                (u2, v2, w2),
                (u3, v3, w3),
                (u4, v4, w4),
                (u5, v5, w5),
                (u6, v6, w6),
                (u7, v7, w7)
            ]
            
            for u, v, w in terms:
                for i_loc, val in u.items():
                    i_g = i_start + (i_loc // 2)
                    j_g = j_start + (i_loc % 2)
                    U[i_g * m + j_g][col] += val
                for j_loc, val in v.items():
                    j_g = j_start + (j_loc // 2)
                    k_g = k_start + (j_loc % 2)
                    V[j_g * p + k_g][col] += val
                for k_loc, val in w.items():
                    k_g = k_start + (k_loc // 2)
                    i_g = i_start + (k_loc % 2)
                    W[k_g * n + i_g][col] += val
                col += 1
                
        elif block_type == 'schoolbook':
            for i_loc in range(di):
                for j_loc in range(dj):
                    for k_loc in range(dk):
                        i_g = i_start + i_loc
                        j_g = j_start + j_loc
                        k_g = k_start + k_loc
                        
                        U[i_g * m + j_g][col] = 1.0
                        V[j_g * p + k_g][col] = 1.0
                        W[k_g * n + i_g][col] = 1.0
                        col += 1
                        
    # Block 1: i[0:2], j[0:2], k[0:2] -> Strassen
    add_block(0, 0, 0, 2, 2, 2, 'strassen_222')
    # Block 2: i[0:2], j[0:2], k[2:4] -> Strassen
    add_block(0, 0, 2, 2, 2, 4, 'strassen_222')
    # Block 3: i[0:2], j[0:2], k[4:7] -> Schoolbook (2x2x3)
    add_block(0, 0, 4, 2, 2, 7, 'schoolbook')
    
    # Block 4: i[2:3], j[0:2], k[0:2] -> Schoolbook (1x2x2)
    add_block(2, 0, 0, 3, 2, 2, 'schoolbook')
    # Block 5: i[2:3], j[0:2], k[2:4] -> Schoolbook (1x2x2)
    add_block(2, 0, 2, 3, 2, 4, 'schoolbook')
    # Block 6: i[2:3], j[0:2], k[4:7] -> Schoolbook (1x2x3)
    add_block(2, 0, 4, 3, 2, 7, 'schoolbook')
    
    # Block 7: i[0:2], j[2:4], k[0:2] -> Strassen
    add_block(0, 2, 0, 2, 4, 2, 'strassen_222')
    # Block 8: i[0:2], j[2:4], k[2:4] -> Strassen
    add_block(0, 2, 2, 2, 4, 4, 'strassen_222')
    # Block 9: i[0:2], j[2:4], k[4:7] -> Schoolbook (2x2x3)
    add_block(0, 2, 4, 2, 4, 7, 'schoolbook')
    
    # Block 10: i[2:3], j[2:4], k[0:2] -> Schoolbook (1x2x2)
    add_block(2, 2, 0, 3, 4, 2, 'schoolbook')
    # Block 11: i[2:3], j[2:4], k[2:4] -> Schoolbook (1x2x2)
    add_block(2, 2, 2, 3, 4, 4, 'schoolbook')
    # Block 12: i[2:3], j[2:4], k[4:7] -> Schoolbook (1x2x3)
    add_block(2, 2, 4, 3, 4, 7, 'schoolbook')
    
    # Convert to int if possible
    def to_int_list(lst):
        return [[int(x) if x == int(x) else x for x in row] for row in lst]
        
    return {"U": to_int_list(U), "V": to_int_list(V), "W": to_int_list(W)}
