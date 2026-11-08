"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
_DIMENSIONS = [2, 3, 6]
DIMENSIONS = _DIMENSIONS  # Keep name for compatibility

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _deadline = _t0 + 110  # Safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # Decomposition: split m=3 into m1=1, m2=2
    # A = [A1 | A2] where A1 is 2x1, A2 is 2x2
    # B = [B1; B2] where B1 is 1x6, B2 is 2x6
    # C = A1*B1 + A2*B2 + A1*B2 + A2*B1
    # 
    # A1*B1: 2x1x6, rank 2 (outer product)
    # A2*B2: 2x2x6, split into 3 blocks of 2x2x2, rank 3*7=21 (Strassen)
    # A1*B2: 2x1x6, rank 2 (outer product)
    # A2*B1: 2x2x1, rank 4 (4 scalar products)
    # Total rank = 2 + 21 + 2 + 4 = 29
    
    rank = 29
    U = [[0] * rank for _ in range(n * m)]  # 6 x 29
    V = [[0] * rank for _ in range(m * p)]  # 18 x 29
    W = [[0] * rank for _ in range(p * n)]  # 12 x 29
    
    # A indices: i in 0..1, j in 0..2, flat = i*m + j = i*3 + j
    #   A1: j=0 -> flat indices 0, 3
    #   A2: j=1,2 -> flat indices 1, 2, 4, 5
    # B indices: j in 0..2, k in 0..5, flat = j*p + k = j*6 + k
    #   B1: j=0 -> flat indices 0..5
    #   B2: j=1,2 -> flat indices 6..17
    # C indices: i in 0..1, k in 0..5, W index = k*n + i = k*2 + i
    
    col = 0
    
    # --- Part 1: A1*B1 (2x1x6), rank 2 ---
    # C[i,k] += A[i,0] * B[0,k]
    # For i=0: u has A[0,0]=1 at flat 0, v has B[0,k] for all k, w has C[0,k] for all k
    # For i=1: u has A[1,0]=1 at flat 3, v has B[0,k] for all k, w has C[1,k] for all k
    
    # Term 1: i=0
    U[0][col] = 1
    for k in range(p):
        V[k][col] = 1  # B[0,k], flat = 0*6+k = k
        W[k*n + 0][col] = 1  # C[0,k], W index = k*2+0
    col += 1
    
    # Term 2: i=1
    U[3][col] = 1  # A[1,0], flat = 1*3+0 = 3
    for k in range(p):
        V[k][col] = 1  # B[0,k]
        W[k*n + 1][col] = 1  # C[1,k], W index = k*2+1
    col += 1
    
    # --- Part 2: A2*B2 (2x2x6), 3 blocks of 2x2x2, rank 21 ---
    # A2: rows i=0,1; cols j=1,2 -> flat indices: (0,1)=1, (0,2)=2, (1,1)=4, (1,2)=5
    # B2: rows j=1,2; cols k=0..5 -> flat indices: (1,k)=6+k, (2,k)=12+k
    # Split k into 3 blocks of 2: k_block = 0,1,2; local_k = 0,1
    # For each block b (k = 2b, 2b+1):
    #   A2 is 2x2: a00=A[0,1]=flat1, a01=A[0,2]=flat2, a10=A[1,1]=flat4, a11=A[1,2]=flat5
    #   B2_b is 2x2: b00=B[1,2b]=flat6+2b, b01=B[1,2b+1]=flat7+2b,
    #                   b10=B[2,2b]=flat12+2b, b11=B[2,2b+1]=flat13+2b
    #   C block: c00=C[0,2b], c01=C[0,2b+1], c10=C[1,2b], c11=C[1,2b+1]
    #   W indices: c00=(2b)*2+0=4b, c01=4b+1, c10=4b+2, c11=4b+3
    
    strassen_terms = [
        # (u_dict, v_dict, w_dict) for local 2x2x2
        # M1 = (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
        ({'a00': 1, 'a11': 1}, {'b00': 1, 'b11': 1}, {'c00': 1, 'c11': 1}),
        # M2 = (a10+a11)b00 -> c10 += M2, c11 -= M2
        ({'a10': 1, 'a11': 1}, {'b00': 1}, {'c10': 1, 'c11': -1}),
        # M3 = a00(b01-b11) -> c01 += M3, c11 += M3
        ({'a00': 1}, {'b01': 1, 'b11': -1}, {'c01': 1, 'c11': 1}),
        # M4 = a11(b10-b00) -> c00 += M4, c10 += M4
        ({'a11': 1}, {'b10': 1, 'b00': -1}, {'c00': 1, 'c10': 1}),
        # M5 = (a00+a01)b11 -> c00 -= M5, c01 += M5
        ({'a00': 1, 'a01': 1}, {'b11': 1}, {'c00': -1, 'c01': 1}),
        # M6 = (a10-a00)(b00+b01) -> c11 += M6
        ({'a10': 1, 'a00': -1}, {'b00': 1, 'b01': 1}, {'c11': 1}),
        # M7 = (a01-a11)(b10+b11) -> c00 += M7
        ({'a01': 1, 'a11': -1}, {'b10': 1, 'b11': 1}, {'c00': 1}),
    ]
    
    # Map local names to flat indices for A2, B2
    a_flat = {'a00': 1, 'a01': 2, 'a10': 4, 'a11': 5}
    
    for b in range(3):
        if time.time() > _deadline:
            break
        # B2 block b: local b00,b01,b10,b11
        b_flat = {
            'b00': 6 + 2*b,       # B[1, 2b]
            'b01': 7 + 2*b,       # B[1, 2b+1]
            'b10': 12 + 2*b,      # B[2, 2b]
            'b11': 13 + 2*b,      # B[2, 2b+1]
        }
        # C block b: local c00,c01,c10,c11
        c_widx = {
            'c00': (2*b)*n + 0,   # C[0, 2b]
            'c01': (2*b+1)*n + 0, # C[0, 2b+1]
            'c10': (2*b)*n + 1,   # C[1, 2b]
            'c11': (2*b+1)*n + 1, # C[1, 2b+1]
        }
        
        for u_dict, v_dict, w_dict in strassen_terms:
            if time.time() > _deadline:
                break
            # U
            for name, val in u_dict.items():
                U[a_flat[name]][col] = val
            # V
            for name, val in v_dict.items():
                V[b_flat[name]][col] = val
            # W
            for name, val in w_dict.items():
                W[c_widx[name]][col] = val
            col += 1
    
    # --- Part 3: A1*B2 (2x1x6), rank 2 ---
    # C[i,k] += A[i,0] * B[1,k] + A[i,0] * B[2,k]... wait no.
    # A1 is 2x1 (j=0), B2 is 2x6 (j=1,2).
    # C[i,k] += A[i,0]*B[1,k] + A[i,0]*B[2,k]
    # This is A[i,0] * (B[1,k] + B[2,k])
    # So for each i, it's an outer product of A[i,0] with (B[1,:] + B[2,:])
    # Rank 2 total.
    
    # Term: i=0
    U[0][col] = 1  # A[0,0]
    for k in range(p):
        V[6 + k][col] = 1    # B[1,k], flat = 1*6+k
        V[12 + k][col] = 1   # B[2,k], flat = 2*6+k
        W[k*n + 0][col] = 1  # C[0,k]
    col += 1
    
    # Term: i=1
    U[3][col] = 1  # A[1,0]
    for k in range(p):
        V[6 + k][col] = 1    # B[1,k]
        V[12 + k][col] = 1   # B[2,k]
        W[k*n + 1][col] = 1  # C[1,k]
    col += 1
    
    # --- Part 4: A2*B1 (2x2x1), rank 4 ---
    # C[i,0] += A[i,1]*B[1,0] + A[i,2]*B[2,0]
    # 4 scalar products:
    # C[0,0] += A[0,1]*B[1,0]
    # C[0,0] += A[0,2]*B[2,0]
    # C[1,0] += A[1,1]*B[1,0]
    # C[1,0] += A[1,2]*B[2,0]
    
    # A2*B1: B1 is 1x6, so B[0,k] for k=0..5.
    # Wait, I mislabeled. Let me re-examine.
    # A = [A1 | A2] where A1 has column j=0, A2 has columns j=1,2
    # B = [B1; B2] where B1 has row j=0, B2 has rows j=1,2
    # C = A1*B1 + A2*B2 + A1*B2 + A2*B1
    # A2*B1: A2 is 2x2 (j=1,2), B1 is 1x6 (j=0)
    # C[i,k] += A[i,1]*B[0,k] + A[i,2]*B[0,k]... 
    # No wait: C[i,k] = sum_j A[i,j]*B[j,k]
    # A2*B1 contributes: sum_{j in {1,2}} A[i,j]*B[j,k] but B1 is only j=0.
    # I think I confused myself. Let me redo.
    
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # 
    # Split: A1 = column j=0 of A, A2 = columns j=1,2 of A
    #         B1 = row j=0 of B, B2 = rows j=1,2 of B
    # 
    # C = A1*B1 + A2*B2
    # where A1*B1: (2x1)*(1x6) = 2x6, and A2*B2: (2x2)*(2x6) = 2x6
    # 
    # So there's no cross term! I made an error above.
    # C = A1*B1 + A2*B2 only.
    # 
    # A1*B1: rank 2
    # A2*B2: 2x2x6, rank 21
    # Total: 23
    
    # Wait, that's even better! Let me verify.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # Let A1 = [[A[0,0]], [A[1,0]]] (2x1), A2 = [[A[0,1], A[0,2]], [A[1,1], A[1,2]]] (2x2)
    # Let B1 = [[B[0,0], ..., B[0,5]]] (1x6), B2 = [[B[1,0], ..., B[1,5]], [B[2,0], ..., B[2,5]]] (2x6)
    # Then A1*B1 is 2x6, A2*B2 is 2x6, and C = A1*B1 + A2*B2. Correct!
    # 
    # So total rank = 2 + 21 = 23.
    
    # But wait, I already allocated rank=29 and filled some columns. Let me restart with rank=23.
    
    # Actually, let me just rebuild everything with the correct decomposition.
    # I'll overwrite U, V, W with the correct 23-rank decomposition.
    
    rank = 23
    U = [[0] * rank for _ in range(n * m)]  # 6 x 23
    V = [[0] * rank for _ in range(m * p)]  # 18 x 23
    W = [[0] * rank for _ in range(p * n)]  # 12 x 23
    
    col = 0
    
    # Part 1: A1*B1 (2x1x6), rank 2
    # C[i,k] += A[i,0]*B[0,k]
    
    # Term 1: i=0
    U[0][col] = 1
    for k in range(p):
        V[k][col] = 1  # B[0,k], flat = 0*6+k
        W[k*n + 0][col] = 1
    col += 1
    
    # Term 2: i=1
    U[3][col] = 1  # A[1,0], flat = 1*3+0
    for k in range(p):
        V[k][col] = 1
        W[k*n + 1][col] = 1
    col += 1
    
    # Part 2: A2*B2 (2x2x6), 3 blocks of 2x2x2, rank 21
    # A2: a00=A[0,1]=flat1, a01=A[0,2]=flat2, a10=A[1,1]=flat4, a11=A[1,2]=flat5
    # B2: for block b (k=2b,2b+1):
    #   b00=B[1,2b]=flat 6+2b, b01=B[1,2b+1]=flat 7+2b
    #   b10=B[2,2b]=flat 12+2b, b11=B[2,2b+1]=flat 13+2b
    
    for b in range(3):
        if time.time() > _deadline:
            break
        b_flat = {
            'b00': 6 + 2*b,
            'b01': 7 + 2*b,
            'b10': 12 + 2*b,
            'b11': 13 + 2*b,
        }
        c_widx = {
            'c00': (2*b)*n + 0,
            'c01': (2*b+1)*n + 0,
            'c10': (2*b)*n + 1,
            'c11': (2*b+1)*n + 1,
        }
        
        for u_dict, v_dict, w_dict in strassen_terms:
            if time.time() > _deadline:
                break
            for name, val in u_dict.items():
                U[a_flat[name]][col] = val
            for name, val in v_dict.items():
                V[b_flat[name]][col] = val
            for name, val in w_dict.items():
                W[c_widx[name]][col] = val
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
