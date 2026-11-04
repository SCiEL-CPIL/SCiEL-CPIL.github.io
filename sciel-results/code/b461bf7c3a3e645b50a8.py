import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Strassen 2x2x2 decomposition
    strassen_U = [
        [1, 0, 1, 0],   # M1: a00+a11
        [0, 0, 1, 1],   # M2: a10+a11
        [1, 0, 0, 0],   # M3: a00
        [0, 0, 0, 1],   # M4: a11
        [1, 1, 0, 0],   # M5: a00+a01
        [0, 1, -1, 0],  # M6: a10-a00
        [0, 1, 1, 0],   # M7: a01+a11
    ]
    strassen_V = [
        [1, 1, 0, 0],   # M1: b00+b11
        [1, 0, 0, 0],   # M2: b00
        [0, 0, 1, -1],  # M3: b10-b11
        [0, 1, -1, 0],  # M4: b01-b00
        [0, 0, 0, 1],   # M5: b11
        [1, 1, 0, 0],   # M6: b00+b10
        [0, 1, 1, 1],   # M7: b01+b11
    ]
    strassen_W = [
        [1, 0, 0, 1],   # M1: +c00, +c11
        [0, 0, 1, -1],  # M2: +c10, -c11
        [0, 0, 1, 0],   # M3: +c10
        [0, 1, 0, 1],   # M4: +c01, +c11
        [-1, 1, 0, 0],  # M5: -c00, +c01
        [0, 0, 1, 0],   # M6: +c10
        [0, 1, 0, 0],   # M7: +c01
    ]
    
    all_U = []
    all_V = []
    all_W = []
    
    # Row groups of A: [0,1] and [2,3]
    # Mid groups: [0,1], [2,3], [4]
    # Col groups of B: [0,1], [2,3], [4,5]
    
    row_groups = [(0, 1), (2, 3)]
    mid_groups = [(0, 1), (2, 3), (4,)]
    col_groups = [(0, 1), (2, 3), (4, 5)]
    
    for rg in row_groups:
        for mg in mid_groups:
            for cg in col_groups:
                r_start = rg[0]
                m_start = mg[0]
                c_start = cg[0]
                
                if len(mg) == 2:
                    for s in range(7):
                        u = [0] * (n * m)
                        v = [0] * (m * p)
                        w = [0] * (p * n)
                        
                        for ai in range(2):
                            for aj in range(2):
                                u[(r_start + ai) * m + (m_start + aj)] = strassen_U[s][ai * 2 + aj]
                        
                        for bi in range(2):
                            for bj in range(2):
                                v[(m_start + bi) * p + (c_start + bj)] = strassen_V[s][bi * 2 + bj]
                        
                        for ci in range(2):
                            for cj in range(2):
                                w[(c_start + cj) * n + (r_start + ci)] = strassen_W[s][ci * 2 + cj]
                        
                        all_U.append(u)
                        all_V.append(v)
                        all_W.append(w)
                else:
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    
                    for ai in range(2):
                        u[(r_start + ai) * m + m_start] = 1
                    
                    v[m_start * p + c_start] = 1
                    v[m_start * p + c_start + 1] = 1
                    
                    for ci in range(2):
                        for cj in range(2):
                            w[(c_start + cj) * n + (r_start + ci)] = 1
                    
                    all_U.append(u)
                    all_V.append(v)
                    all_W.append(w)
    
    return {"U": all_U, "V": all_V, "W": all_W}
