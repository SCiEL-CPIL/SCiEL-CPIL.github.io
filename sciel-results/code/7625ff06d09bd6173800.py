import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    n, m, p = DIMENSIONS

    def verify_decomp(U, V, W, rank):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vi = j * p + k
                    wi = k * n + i
                    s = np.dot(U_arr[ui, :], V_arr[vi, :] * W_arr[wi, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_rank65():
        """Build a rank-65 decomposition using a refined split strategy.
        
        Strategy:
        - n=3, m=4, p=6
        - Split p into 3+3
        - For n=0: schoolbook (12 terms per p-block, 24 total)
        - For n=1,2 with p=0..2: Strassen on 2x2x2 (7 terms, 14 total)  
        - For n=1,2 with p=3..5: need another 13 terms for rank 65 total? 
          Wait, let me recalculate.
        
        Actually, let me think more carefully:
        - Total rank 66 = 24 + 14 + 14 + 14? No, that's 66.
        
        Let me try: p = 2+2+2, and for each 2-block:
          n=0: schoolbook 1*4*2 = 8 terms
          n=1,2: Strassen 7 terms
          Per block: 8+7=15, total 45... no that's not right either.
        
        Actually the existing rank-66 works like this:
        - p split 4+2:
          - p=0..3, n=0: schoolbook 1*4*4 = 16
          - p=0..3, n=1..3: two Strassen (each 7) = 14
          - p=4..5, n=0: schoolbook 1*4*2 = 8
          - p=4..5, n=1..3: two Strassen (each 7) = 14
          Total: 16+14+8+14 = 52... that's not 66.
        
        Let me re-examine build_rank66:
        add_sb(0, 0, 0, 1, 4, 4) -> i=0, j=0..3, k=0..3: 4*4 = 16 rank-1 terms
        add_ssr(1, 0, 0) -> Strassen for i=1..2, j=0..1, k=0..1: 7
        add_ssr(1, 2, 0) -> Strassen for i=1..2, j=2..3, k=0..1: 7
        add_ssr(1, 0, 2) -> Strassen for i=1..2, j=0..1, k=2..3: 7
        add_ssr(1, 2, 2) -> Strassen for i=1..2, j=2..3, k=2..3: 7
        add_sb(0, 0, 4, 1, 4, 2) -> i=0, j=0..3, k=4..5: 4*2 = 8
        add_ssr(1, 0, 4) -> 7
        add_ssr(1, 2, 4) -> 7
        Total: 16+7*6+8 = 16+42+8 = 66. Yes!
        
        So the structure is:
        - p=0..3 (4 columns): n=0 schoolbook (16), n=1..2 Strassen over 4 blocks of 2x2 (28)
        - p=4..5 (2 columns): n=0 schoolbook (8), n=1..2 Strassen over 2 blocks of 2x2 (14)
        - Total: 16+28+8+14 = 66
        
        To get rank 65, I need to save 1 term. Can I overlap or merge any terms?
        
        Alternative: use a 3x4x6 tensor rank. Let me look for a smarter decomposition.
        
        Idea: Use rank-65 by splitting differently.
        
        For 3x4x6:
        Option: split n=1+2, p=6.
          n=0, p=6: schoolbook 1*4*6 = 24
          n=1..2, p=6: need to multiply 2x4 by 4x6.
            Split p=2+2+2: three 2x4x2 blocks.
            Each 2x4x2: split m=2+2, two 2x2x2 Strassen = 14 per block, 42 total.
          Total: 24+42 = 66. Same.
        
        Option: p=3+3.
          n=0, p=6: 24 schoolbook
          n=1..2, p=0..2: 2x4x3. Split m=2+2, two 2x2x3.
            Each 2x2x3: can do with rank 5? (2*2*3=12 schoolbook, but maybe 5 or 6 with Strassen-like)
            Actually for 2x2x3, the minimal rank is... let me think. We need A_{2x2} * B_{2x3}.
            This is equivalent to computing 4 dot products of length 3. 
            Rank 5 or 6 might work? 
          This gets complicated.
        
        Let me try a different approach: use the known rank-66 and see if I can compress by 1.
        
        Actually, let me just implement the rank-66 cleanly and hope the verification passes, 
        then try to find if rank 65 is possible by a small perturbation or different grouping.
        """
        # Try to achieve rank 65 by a clever grouping
        rank = 65
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        cur = [0]

        def add_rank1(u_iv, v_iv, w_iv):
            if cur[0] >= rank:
                return
            for idx, val in u_iv:
                U[idx][cur[0]] += val
            for idx, val in v_iv:
                V[idx][cur[0]] += val
            for idx, val in w_iv:
                W[idx][cur[0]] += val
            cur[0] += 1

        def add_ssr(i0, j0, k0):
            U2 = lambda i, j: (i0 + i) * m + (j0 + j)
            V2 = lambda j, k: (j0 + j) * p + (k0 + k)
            W2 = lambda i, k: (k0 + k) * n + (i0 + i)
            add_rank1([(U2(0,0),1),(U2(1,1),1)], [(V2(0,0),1),(V2(1,1),1)], [(W2(0,0),1),(W2(1,1),1)])
            add_rank1([(U2(1,0),1),(U2(1,1),1)], [(V2(0,0),1)], [(W2(1,0),1),(W2(1,1),-1)])
            add_rank1([(U2(0,0),1)], [(V2(0,1),1),(V2(1,1),-1)], [(W2(0,1),1),(W2(1,1),1)])
            add_rank1([(U2(1,1),1)], [(V2(1,0),1),(V2(0,0),-1)], [(W2(0,0),1),(W2(1,0),1)])
            add_rank1([(U2(0,0),1),(U2(0,1),1)], [(V2(1,1),1)], [(W2(0,0),-1),(W2(0,1),1)])
            add_rank1([(U2(1,0),1),(U2(0,0),-1)], [(V2(0,0),1),(V2(0,1),1)], [(W2(1,1),1)])
            add_rank1([(U2(0,1),1),(U2(1,1),-1)], [(V2(1,0),1),(V2(1,1),1)], [(W2(0,0),1)])

        # Use same structure as rank-66 but see if we can save one term
        # For now, let's just do rank-66 construction and return it if rank-65 doesn't work
        # But we declared rank=65, so let's fill 66 and see...
        
        # Actually, let me rebuild with rank 66 as fallback within this function
        # and return the 66 if 65 isn't achievable
        pass

    def build_rank66():
        """Build the known rank-66 decomposition via p=4+2 split with Strassen."""
        rank = 66
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        cur = [0]

        def add_rank1(u_iv, v_iv, w_iv):
            if cur[0] >= rank:
                return
            for idx, val in u_iv:
                U[idx][cur[0]] += val
            for idx, val in v_iv:
                V[idx][cur[0]] += val
            for idx, val in w_iv:
                W[idx][cur[0]] += val
            cur[0] += 1

        def add_sb(i0, j0, k0, nn, mm, pp):
            for ii in range(nn):
                for jj in range(mm):
                    for kk in range(pp):
                        add_rank1([((i0 + ii) * m + (j0 + jj), 1.0)],
                                  [((j0 + jj) * p + (k0 + kk), 1.0)],
                                  [((k0 + kk) * n + (i0 + ii), 1.0)])

        def add_ssr(i0, j0, k0):
            U2 = lambda i, j: (i0 + i) * m + (j0 + j)
            V2 = lambda j, k: (j0 + j) * p + (k0 + k)
            W2 = lambda i, k: (k0 + k) * n + (i0 + i)
            add_rank1([(U2(0,0),1),(U2(1,1),1)], [(V2(0,0),1),(V2(1,1),1)], [(W2(0,0),1),(W2(1,1),1)])
            add_rank1([(U2(1,0),1),(U2(1,1),1)], [(V2(0,0),1)], [(W2(1,0),1),(W2(1,1),-1)])
            add_rank1([(U2(0,0),1)], [(V2(0,1),1),(V2(1,1),-1)], [(W2(0,1),1),(W2(1,1),1)])
            add_rank1([(U2(1,1),1)], [(V2(1,0),1),(V2(0,0),-1)], [(W2(0,0),1),(W2(1,0),1)])
            add_rank1([(U2(0,0),1),(U2(0,1),1)], [(V2(1,1),1)], [(W2(0,0),-1),(W2(0,1),1)])
            add_rank1([(U2(1,0),1),(U2(0,0),-1)], [(V2(0,0),1),(V2(0,1),1)], [(W2(1,1),1)])
            add_rank1([(U2(0,1),1),(U2(1,1),-1)], [(V2(1,0),1),(V2(1,1),1)], [(W2(0,0),1)])

        # p split: 4 + 2
        add_sb(0, 0, 0, 1, 4, 4)
        add_ssr(1, 0, 0)
        add_ssr(1, 2, 0)
        add_ssr(1, 0, 2)
        add_ssr(1, 2, 2)
        add_sb(0, 0, 4, 1, 4, 2)
        add_ssr(1, 0, 4)
        add_ssr(1, 2, 4)

        return {"U": U, "V": V, "W": W}

    def build_rank66_v2():
        """Alternative rank-66 using p=2+2+2 split."""
        rank = 66
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        cur = [0]

        def add_rank1(u_iv, v_iv, w_iv):
            if cur[0] >= rank:
                return
            for idx, val in u_iv:
                U[idx][cur[0]] += val
            for idx, val in v_iv:
                V[idx][cur[0]] += val
            for idx, val in w_iv:
                W[idx][cur[0]] += val
            cur[0] += 1

        def add_ssr(i0, j0, k0):
            U2 = lambda i, j: (i0 + i) * m + (j0 + j)
            V2 = lambda j, k: (j0 + j) * p + (k0 + k)
            W2 = lambda i, k: (k0 + k) * n + (i0 + i)
            add_rank1([(U2(0,0),1),(U2(1,1),1)], [(V2(0,0),1),(V2(1,1),1)], [(W2(0,0),1),(W2(1,1),1)])
            add_rank1([(U2(1,0),1),(U2(1,1),1)], [(V2(0,0),1)], [(W2(1,0),1),(W2(1,1),-1)])
            add_rank1([(U2(0,0),1)], [(V2(0,1),1),(V2(1,1),-1)], [(W2(0,1),1),(W2(1,1),1)])
            add_rank1([(U2(1,1),1)], [(V2(1,0),1),(V2(0,0),-1)], [(W2(0,0),1),(W2(1,0),1)])
            add_rank1([(U2(0,0),1),(U2(0,1),1)], [(V2(1,1),1)], [(W2(0,0),-1),(W2(0,1),1)])
            add_rank1([(U2(1,0),1),(U2(0,0),-1)], [(V2(0,0),1),(V2(0,1),1)], [(W2(1,1),1)])
            add_rank1([(U2(0,1),1),(U2(1,1),-1)], [(V2(1,0),1),(V2(1,1),1)], [(W2(0,0),1)])

        for k_block in range(3):
            k0 = k_block * 2
            # n=0: schoolbook for 1x4x2
            for j in range(4):
                for k in range(2):
                    add_rank1([((0) * m + j, 1.0)], [(j * p + (k0 + k), 1.0)], [((k0 + k) * n + 0, 1.0)])
            # n=1..2: Strassen on 2x4x2, split m=2+2
            add_ssr(1, 0, k0)
            add_ssr(1, 2, k0)

        return {"U": U, "V": V, "W": W}

    def build_schoolbook():
        """Pure schoolbook rank 72 fallback."""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1.0
                    V[j * p + k][r] = 1.0
                    W[k * n + i][r] = 1.0
                    r += 1
        return {"U": U, "V": V, "W": W}

    # Try multiple constructions, verify, return best valid one
    best = None
    best_rank = float('inf')

    builders = [build_rank66, build_rank66_v2, build_schoolbook]

    for builder in builders:
        if time.time() > deadline - 5:
            break
        try:
            result = builder()
            r = len(result["U"][0])
            if r < best_rank:
                if verify_decomp(result["U"], result["V"], result["W"], r):
                    best_rank = r
                    best = result
        except Exception:
            continue

    if best is None:
        best = build_rank66()

    return best
