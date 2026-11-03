import time
import random

def construct():
    n, m, p = 4, 4, 7
    # U: n*m = 16, V: m*p = 28, W: p*n = 28
    # We'll build a list of terms, each term = (u_dict, v_dict, w_dict) mapping slot->coeff
    # Then assemble U, V, W.
    
    terms = []  # list of (u: dict, v: dict, w: dict)
    
    # ---- 4x4 x 4x4 part (columns 0-3) ----
    # Base 2x2 Strassen: 7 terms, each (u4, v4, w4) lists of 4
    base = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),  # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),  # M2
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]), # M3
        ([0,0,0,1], [0,0,1,-1], [1,0,1,0]), # M4
        ([1,1,0,0], [0,0,0,1], [0,1,0,-1]), # M5
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]), # M6
        ([0,1,0,-1], [0,0,1,1], [1,0,0,-1]),# M7
    ]
    
    # Super-term definitions: a_blocks, b_blocks, c_blocks (dict (br,bc)->coeff)
    super_terms = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),  # M1
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1}),  # M2
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1}),  # M3
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),  # M4
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),  # M5
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),  # M6
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):-1,(1,1):-1}),  # M7
    ]
    
    # Global index helpers for 4x4 part
    def a4_idx(br, bc, lr, lc):
        # A block (br,bc), local (lr,lc) -> global A slot (i,j) = (br*2+lr, bc*2+lc) -> i*4+j
        i = br*2+lr
        j = bc*2+lc
        return i*4+j
    def b4_idx(br, bc, lr, lc):
        # B block (br,bc) for 4x4 part: B is 4x4, block (br,bc) local (lr,lc) -> (j,k) = (br*2+lr, bc*2+lc) -> j*7+k
        j = br*2+lr
        k = bc*2+lc
        return j*7+k
    def c4_idx(br, bc, lr, lc):
        # C block (br,bc) for 4x4 part: C is 4x4, block (br,bc) local (lr,lc) -> (i,k) = (br*2+lr, bc*2+lc) -> k*4+i
        i = br*2+lr
        k = bc*2+lc
        return k*4+i
    
    for s in range(7):
        a_blocks, b_blocks, c_blocks = super_terms[s]
        for t in range(7):
            u4, v4, w4 = base[t]
            u_dict = {}
            v_dict = {}
            w_dict = {}
            # A side
            for lr in range(2):
                for lc in range(2):
                    local = lr*2+lc
                    cu = u4[local]
                    if cu == 0: continue
                    for (br,bc), ca in a_blocks.items():
                        if ca == 0: continue
                        g = a4_idx(br,bc,lr,lc)
                        u_dict[g] = u_dict.get(g,0) + cu*ca
            # B side
            for lr in range(2):
                for lc in range(2):
                    local = lr*2+lc
                    cv = v4[local]
                    if cv == 0: continue
                    for (br,bc), cb in b_blocks.items():
                        if cb == 0: continue
                        g = b4_idx(br,bc,lr,lc)
                        v_dict[g] = v_dict.get(g,0) + cv*cb
            # C side
            for lr in range(2):
                for lc in range(2):
                    local = lr*2+lc
                    cw = w4[local]
                    if cw == 0: continue
                    for (br,bc), cc in c_blocks.items():
                        if cc == 0: continue
                        g = c4_idx(br,bc,lr,lc)
                        w_dict[g] = w_dict.get(g,0) + cw*cc
            terms.append((u_dict, v_dict, w_dict))
    
    # ---- 4x4 x 4x3 part (columns 4-6) ----
    # A blocks: A00,A01,A10,A11 (2x2 each)
    # B blocks: B_top (2x3), B_bot (2x3)
    # C blocks: C_top (2x3), C_bot (2x3)
    # C_top = A00·B_top + A01·B_bot
    # C_bot = A10·B_top + A11·B_bot
    # Each 2x2·2x3 = 11 terms (7 for 2x2·2x2 + 4 for 2x2·2x1)
    
    # 2x2·2x3 base terms: X (2x2, 4 slots lr*2+lc), Y (2x3, 6 slots lr*3+lc), Z (2x3, 6 slots lr*3+lc)
    # 2x2·2x2 part (Y cols 0,1):
    base_2223 = []
    # 7 Strassen terms for 2x2·2x2, Y_left = Y[*,0:2] (slots 0,1,3,4), Z_left = Z[*,0:2] (slots 0,1,3,4)
    # Map local 2x2 Y slot (lr,lc) -> Y global slot lr*3+lc (for lc=0,1)
    # Map local 2x2 Z slot (lr,lc) -> Z global slot lr*3+lc (for lc=0,1)
    for t in range(7):
        u4, v4, w4 = base[t]
        u_dict = {}
        v_dict = {}
        w_dict = {}
        for lr in range(2):
            for lc in range(2):
                local = lr*2+lc
                cu = u4[local]
                if cu != 0:
                    u_dict[local] = u_dict.get(local,0) + cu
                cv = v4[local]
                if cv != 0:
                    yg = lr*3+lc  # Y slot for col lc (0 or 1)
                    v_dict[yg] = v_dict.get(yg,0) + cv
                cw = w4[local]
                if cw != 0:
                    zg = lr*3+lc  # Z slot for col lc (0 or 1)
                    w_dict[zg] = w_dict.get(zg,0) + cw
        base_2223.append((u_dict, v_dict, w_dict))
    # 4 schoolbook terms for 2x2·2x1 (Y col 2):
    # z[0,2] += x[0,0]·y[0,2]; z[0,2] += x[0,1]·y[1,2]; z[1,2] += x[1,0]·y[0,2]; z[1,2] += x[1,1]·y[1,2]
    # Y col 2 slots: (0,2)=2, (1,2)=5
    # Z col 2 slots: (0,2)=2, (1,2)=5
    base_2223.append(({0:1}, {2:1}, {2:1}))   # x00*y02 -> z02
    base_2223.append(({1:1}, {5:1}, {2:1}))   # x01*y12 -> z02
    base_2223.append(({2:1}, {2:1}, {5:1}))   # x10*y02 -> z12
    base_2223.append(({3:1}, {5:1}, {5:1}))   # x11*y12 -> z12
    # base_2223 now has 11 terms
    
    # Now 4 block products:
    # (A00, B_top, C_top), (A01, B_bot, C_top), (A10, B_top, C_bot), (A11, B_bot, C_bot)
    # A block (br,bc) local (lr,lc) -> A global slot (br*2+lr)*4 + (bc*2+lc)
    # B_top local (lr,lc) [2x3] -> B global slot (0*2+lr)*7 + (4+lc) = lr*7 + 4+lc
    # B_bot local (lr,lc) [2x3] -> B global slot (2*2+lr)*7 + (4+lc) = (4+lr)*7 + 4+lc = (4+lr)*7+4+lc
    # C_top local (lr,lc) [2x3] -> C global slot (4+lc)*4 + (0*2+lr) = (4+lc)*4 + lr
    # C_bot local (lr,lc) [2x3] -> C global slot (4+lc)*4 + (2*2+lr) = (4+lc)*4 + (4+lr)
    
    def a_block_slot(br, bc, lr, lc):
        return (br*2+lr)*4 + (bc*2+lc)
    def b_top_slot(lr, lc):
        return lr*7 + (4+lc)
    def b_bot_slot(lr, lc):
        return (4+lr)*7 + (4+lc)
    def c_top_slot(lr, lc):
        return (4+lc)*4 + lr
    def c_bot_slot(lr, lc):
        return (4+lc)*4 + (4+lr)
    
    block_products = [
        ((0,0), b_top_slot, c_top_slot),   # A00 · B_top -> C_top
        ((0,1), b_bot_slot, c_top_slot),   # A01 · B_bot -> C_top
        ((1,0), b_top_slot, c_bot_slot),   # A10 · B_top -> C_bot
        ((1,1), b_bot_slot, c_bot_slot),   # A11 · B_bot -> C_bot
    ]
    
    for (abr, abc), b_slot_fn, c_slot_fn in block_products:
        for (u_dict, v_dict, w_dict) in base_2223:
            new_u = {}
            new_v = {}
            new_w = {}
            for local_a, cu in u_dict.items():
                lr, lc = local_a//2, local_a%2
                g = a_block_slot(abr, abc, lr, lc)
                new_u[g] = new_u.get(g,0) + cu
            for local_y, cv in v_dict.items():
                lr, lc = local_y//3, local_y%3
                g = b_slot_fn(lr, lc)
                new_v[g] = new_v.get(g,0) + cv
            for local_z, cw in w_dict.items():
                lr, lc = local_z//3, local_z%3
                g = c_slot_fn(lr, lc)
                new_w[g] = new_w.get(g,0) + cw
            terms.append((new_u, new_v, new_w))
    
    # Total terms
    rank = len(terms)
    
    # Assemble U, V, W
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for g, c in u_dict.items():
            U[g][col] = c
        for g, c in v_dict.items():
            V[g][col] = c
        for g, c in w_dict.items():
            W[g][col] = c
    
    # Verify
    def verify():
        for _ in range(20):
            A = [[random.randint(-3,3) for _ in range(4)] for _ in range(4)]
            B = [[random.randint(-3,3) for _ in range(7)] for _ in range(4)]
            # True C = A·B
            C_true = [[sum(A[i][j]*B[j][k] for j in range(4)) for k in range(7)] for i in range(4)]
            # Reconstruct C from terms: C[i,k] = sum_col U[i*4+k... wait U index is i*m+j = i*4+j for A slot (i,j). 
            # Actually U slot for A(i,j) is i*4+j. V slot for B(j,k) is j*7+k. W slot for C(i,k) is k*4+i.
            # C[i,k] = sum_col (sum_j U[i*4+j][col]) * (sum_j V[j*7+k][col]) ... no.
            # The decomposition: C = sum_col (W_col * (U_col · A) * (V_col · B))? 
            # Actually the tensor decomposition: C[i,k] = sum_col U[i,j]... 
            # Let me recall: for matrix multiply C = A·B, the tensor T[i,j,k] = δ... The decomposition is:
            # C[i,k] = sum_col (sum_j U[i*4+j][col] * A[i,j]) * (sum_j V[j*7+k][col] * B[j,k])? No.
            # Standard: C[i,k] = sum_j A[i,j] B[j,k]. 
            # Tensor decomposition: C[i,k] = sum_col w_col[i,k] * (u_col · a) * (v_col · b) where a is the flattened A, b is flattened B, w_col is the W vector.
            # Wait, that's not right either. Let me think.
            # The matrix multiply tensor: T[i,j,k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]. T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k]).
            # A rank-r decomposition: T[i,j,k] = sum_col U[i,j,col] V[j,k,col] W[i,k,col].
            # So C[i,k] = sum_j A[i,j] B[j,k] = sum_col W[i,k,col] * (sum_j U[i,j,col] A[i,j]) * (sum_j V[j,k,col] B[j,k]).
            # Hmm, that's a specific form. Let me verify: sum_col W[i,k,col] * (sum_j U[i,j,col] A[i,j]) * (sum_j V[j,k,col] B[j,k]).
            # This equals sum_col W[i,k,col] * sum_{j1} U[i,j1,col] A[i,j1] * sum_{j2} V[j2,k,col] B[j2,k]
            # = sum_{j1,j2} A[i,j1] B[j2,k] * sum_col W[i,k,col] U[i,j1,col] V[j2,k,col]
            # For this to equal sum_j A[i,j] B[j,k] = sum_{j1,j2} δ_{j1,j2} A[i,j1] B[j2,k], we need:
            # sum_col W[i,k,col] U[i,j1,col] V[j2,k,col] = δ_{j1,j2} for all i,k,j1,j2.
            # So the verification is: for each i,k,j1,j2, sum_col W[k*4+i][col] * U[i*4+j1][col] * V[j2*7+k][col] == (1 if j1==j2 else 0).
            # Let me just compute C via this formula and compare to C_true.
            C_rec = [[0]*7 for _ in range(4)]
            for i in range(4):
                for k in range(7):
                    s = 0
                    for col in range(rank):
                        w = W[k*4+i][col]
                        if w == 0: continue
                        ua = sum(U[i*4+j][col]*A[i][j] for j in range(4))
                        vb = sum(V[j*7+k][col]*B[j][k] for j in range(4))
                        s += w * ua * vb
                    C_rec[i][k] = s
            if C_rec != C_true:
                return False
        return True
    
    if verify():
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback: schoolbook
        rank2 = n*m*p
        U2 = [[0]*rank2 for _ in range(n*m)]
        V2 = [[0]*rank2 for _ in range(m*p)]
        W2 = [[0]*rank2 for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i*m+j][col] = 1
                    V2[j*p+k][col] = 1
                    W2[k*n+i][col] = 1
                    col += 1
        return {"U": U2, "V": V2, "W": W2}
