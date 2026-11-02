import time

DIMENSIONS = [2, 2, 14]

def construct():
    start_time = time.time()
    deadline = 150.0  # seconds
    
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 multiplication
    # U indices: 0:00, 1:01, 2:10, 3:11
    # Local V indices: 0:00, 1:01, 2:10, 3:11
    # Local W indices: 0:00, 1:01, 2:10, 3:11 (Note: C is 2x2)
    # 
    # Term 0: M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    # Term 1: M2 = (A10+A11)B00       -> C10 += 1, C11 += -1
    # Term 2: M3 = A00(B01-B11)       -> C01 += 1, C11 += 1
    # Term 3: M4 = A11(B10-B00)       -> C00 += 1, C10 += 1
    # Term 4: M5 = (A00+A01)B11       -> C00 += -1, C01 += 1
    # Term 5: M6 = (A10-A00)(B00+B01) -> C11 += 1
    # Term 6: M7 = (A01-A11)(B10+B11) -> C00 += 1
    
    # Format: { 'U': {idx: coeff}, 'V': {idx: coeff}, 'W': {idx: coeff} }
    terms = [
        {'U': {0: 1, 3: 1}, 'V': {0: 1, 3: 1}, 'W': {0: 1, 3: 1}}, # M1
        {'U': {2: 1, 3: 1}, 'V': {0: 1},       'W': {1: 1, 3: -1}}, # M2 (C10=1, C11=-1)
        {'U': {0: 1},       'V': {1: 1, 3: -1}, 'W': {2: 1, 3: 1}},  # M3 (C01=1, C11=1)
        {'U': {3: 1},       'V': {2: 1, 0: -1}, 'W': {0: 1, 1: 1}},  # M4 (C00=1, C10=1)
        {'U': {0: 1, 1: 1}, 'V': {3: 1},       'W': {0: -1, 2: 1}},  # M5 (C00=-1, C01=1)
        {'U': {2: 1, 0: -1},'V': {0: 1, 1: 1}, 'W': {3: 1}},         # M6 (C11=1)
        {'U': {1: 1, 3: -1},'V': {2: 1, 3: 1}, 'W': {0: 1}},         # M7 (C00=1)
    ]
    
    # Mapping functions
    # U global index is same as local for A (2x2)
    # V local index l (0..3) for block j:
    # l=0 -> B_{0, 2j} -> global v = 0*p + 2j
    # l=1 -> B_{0, 2j+1} -> global v = 0*p + 2j+1
    # l=2 -> B_{1, 2j} -> global v = 1*p + 2j
    # l=3 -> B_{1, 2j+1} -> global v = 1*p + 2j+1
    # W local index l (0..3) for block j:
    # l=0 -> C_{0, 2j} -> global w = (2j)*n + 0
    # l=1 -> C_{1, 2j} -> global w = (2j)*n + 1  (Wait, C is 2x14. C_{i,k}. k is col. w=k*n+i)
    # Let's re-verify W mapping.
    # C is 2x14. i in 0..1, k in 0..13.
    # w = k*n + i.
    # Block j has k in {2j, 2j+1}.
    # l=0 (C_{0, 2j}): i=0, k=2j. w = 2j*2 + 0 = 4j.
    # l=1 (C_{1, 2j}): i=1, k=2j. w = 2j*2 + 1 = 4j+1.
    # l=2 (C_{0, 2j+1}): i=0, k=2j+1. w = (2j+1)*2 + 0 = 4j+2.
    # l=3 (C_{1, 2j+1}): i=1, k=2j+1. w = (2j+1)*2 + 1 = 4j+3.
    
    for j in range(num_blocks):
        for s in range(7):
            r = j * 7 + s
            term = terms[s]
            
            # Fill U
            for u_idx, coeff in term['U'].items():
                U[u_idx][r] = coeff
                
            # Fill V
            for v_idx_local, coeff in term['V'].items():
                row = v_idx_local // 2
                col_local = v_idx_local % 2
                global_col = 2 * j + col_local
                v_idx_global = row * p + global_col
                V[v_idx_global][r] = coeff
                
            # Fill W
            for w_idx_local, coeff in term['W'].items():
                # w_idx_local 0..3 corresponds to C_{0,0}, C_{1,0}, C_{0,1}, C_{1,1} in local 2x2 block
                # which maps to C_{0, 2j}, C_{1, 2j}, C_{0, 2j+1}, C_{1, 2j+1}
                w_idx_global = 4 * j + w_idx_local
                W[w_idx_global][r] = coeff
                
    if time.time() > deadline - 5:
        return None
        
    return {"U": U, "V": V, "W": W}
