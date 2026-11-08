import time

DIMENSIONS = [3, 4, 6]

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS

    def build_rank66():
        """Build the rank-66 decomposition: p split as 4+2, n split as 1+2 per p-block."""
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

        def add_schoolbook(i0, j0, k0, nn, mm, pp):
            for i in range(nn):
                for j in range(mm):
                    for k in range(pp):
                        ui = (i0 + i) * m + (j0 + j)
                        vi = (j0 + j) * p + (k0 + k)
                        wi = (k0 + k) * n + (i0 + i)
                        add_rank1([(ui, 1.0)], [(vi, 1.0)], [(wi, 1.0)])

        def add_strassen222(i0, j0, k0):
            def u(i, j): return (i0 + i) * m + (j0 + j)
            def v(j, k): return (j0 + j) * p + (k0 + k)
            def w(i, k): return (k0 + k) * n + (i0 + i)
            add_rank1([(u(0,0),1),(u(1,1),1)], [(v(0,0),1),(v(1,1),1)], [(w(0,0),1),(w(1,1),1)])
            add_rank1([(u(1,0),1),(u(1,1),1)], [(v(0,0),1)], [(w(1,0),1),(w(1,1),-1)])
            add_rank1([(u(0,0),1)], [(v(0,1),1),(v(1,1),-1)], [(w(0,1),1),(w(1,1),1)])
            add_rank1([(u(1,1),1)], [(v(1,0),1),(v(0,0),-1)], [(w(0,0),1),(w(1,0),1)])
            add_rank1([(u(0,0),1),(u(0,1),1)], [(v(1,1),1)], [(w(0,0),-1),(w(0,1),1)])
            add_rank1([(u(1,0),1),(u(0,0),-1)], [(v(0,0),1),(v(0,1),1)], [(w(1,1),1)])
            add_rank1([(u(0,1),1),(u(1,1),-1)], [(v(1,0),1),(v(1,1),1)], [(w(0,0),1)])

        # p split: 4 + 2
        # First 4 columns of p (k=0..3):
        #   n=0: schoolbook M(1,4,4) = 16
        add_schoolbook(0, 0, 0, 1, 4, 4)
        #   n=1,2: 4 Strassen blocks M(2,2,2) = 28
        add_strassen222(1, 0, 0)
        add_strassen222(1, 2, 0)
        add_strassen222(1, 0, 2)
        add_strassen222(1, 2, 2)
        # Last 2 columns of p (k=4,5):
        #   n=0: schoolbook M(1,4,2) = 8
        add_schoolbook(0, 0, 4, 1, 4, 2)
        #   n=1,2: 2 Strassen blocks M(2,2,2) = 14
        add_strassen222(1, 0, 4)
        add_strassen222(1, 2, 4)
        # Total: 16 + 28 + 8 + 14 = 66

        return {"U": U, "V": V, "W": W}

    def build_rank66_alt():
        """Alternative rank-66: p split as 2+2+2, n split as 1+2 per p-block."""
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

        def add_schoolbook(i0, j0, k0, nn, mm, pp):
            for i in range(nn):
                for j in range(mm):
                    for k in range(pp):
                        ui = (i0 + i) * m + (j0 + j)
                        vi = (j0 + j) * p + (k0 + k)
                        wi = (k0 + k) * n + (i0 + i)
                        add_rank1([(ui, 1.0)], [(vi, 1.0)], [(wi, 1.0)])

        def add_strassen222(i0, j0, k0):
            def u(i, j): return (i0 + i) * m + (j0 + j)
            def v(j, k): return (j0 + j) * p + (k0 + k)
            def w(i, k): return (k0 + k) * n + (i0 + i)
            add_rank1([(u(0,0),1),(u(1,1),1)], [(v(0,0),1),(v(1,1),1)], [(w(0,0),1),(w(1,1),1)])
            add_rank1([(u(1,0),1),(u(1,1),1)], [(v(0,0),1)], [(w(1,0),1),(w(1,1),-1)])
            add_rank1([(u(0,0),1)], [(v(0,1),1),(v(1,1),-1)], [(w(0,1),1),(w(1,1),1)])
            add_rank1([(u(1,1),1)], [(v(1,0),1),(v(0,0),-1)], [(w(0,0),1),(w(1,0),1)])
            add_rank1([(u(0,0),1),(u(0,1),1)], [(v(1,1),1)], [(w(0,0),-1),(w(0,1),1)])
            add_rank1([(u(1,0),1),(u(0,0),-1)], [(v(0,0),1),(v(0,1),1)], [(w(1,1),1)])
            add_rank1([(u(0,1),1),(u(1,1),-1)], [(v(1,0),1),(v(1,1),1)], [(w(0,0),1)])

        for block_k in range(3):
            k_off = block_k * 2
            # n=0: schoolbook M(1,4,2) = 8
            add_schoolbook(0, 0, k_off, 1, 4, 2)
            # n=1,2: 2 Strassen blocks M(2,2,2) = 14
            add_strassen222(1, 0, k_off)
            add_strassen222(1, 2, k_off)
        # Total: 3 * (8 + 14) = 66

        return {"U": U, "V": V, "W": W}

    # Build the primary rank-66 decomposition
    best = build_rank66()
    best_rank = len(best["U"][0])

    # Try the alternative structure as well (same rank, different arrangement)
    if time.time() < deadline - 5:
        try:
            alt = build_rank66_alt()
            alt_rank = len(alt["U"][0])
            if alt_rank < best_rank:
                best = alt
                best_rank = alt_rank
        except Exception:
            pass

    return best
