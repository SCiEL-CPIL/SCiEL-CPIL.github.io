"""Matrix multiplication 3,5,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    import time
    _start = time.time()
    _deadline = _start + 110.0  # safety margin before 120s kill

    n, m, p = DIMENSIONS

    # Strassen patterns for 2x2x2 (verified correct)
    strassen_patterns = [
        # M1 = (a00+a11)(b00+b11)
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]
        },
        # M2 = (a10+a11)b00
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)]
        },
        # M3 = a00(b01-b11)
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(1,0,1), (1,1,1)]
        },
        # M4 = a11(b10-b00)
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (0,1,1)]
        },
        # M5 = (a00+a01)b11
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (1,0,1)]
        },
        # M6 = (a10-a00)(b00+b01)
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)]
        },
        # M7 = (a01-a11)(b10+b11)
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)]
        }
    ]

    def get_offsets(splits):
        offsets = [0]
        for s in splits[:-1]:
            offsets.append(offsets[-1] + s)
        return offsets

    def build_decomposition(n_splits, m_splits, p_splits):
        """Build U, V, W entries for given splits."""
        n_off = get_offsets(n_splits)
        m_off = get_offsets(m_splits)
        p_off = get_offsets(p_splits)

        U_entries = []
        V_entries = []
        W_entries = []
        rank = 0

        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    n_b = n_splits[i_b]
                    m_b = m_splits[j_b]
                    p_b = p_splits[k_b]

                    i_off = n_off[i_b]
                    j_off = m_off[j_b]
                    k_off = p_off[k_b]

                    if n_b == 2 and m_b == 2 and p_b == 2:
                        for pattern in strassen_patterns:
                            col = rank
                            for li, lj, val in pattern['U']:
                                gi = i_off + li
                                gj = j_off + lj
                                U_entries.append((gi * m + gj, col, val))
                            for lj, lk, val in pattern['V']:
                                gj = j_off + lj
                                gk = k_off + lk
                                V_entries.append((gj * p + gk, col, val))
                            for lk, li, val in pattern['W']:
                                gk = k_off + lk
                                gi = i_off + li
                                W_entries.append((gk * n + gi, col, val))
                            rank += 1
                    else:
                        for li in range(n_b):
                            for lj in range(m_b):
                                for lk in range(p_b):
                                    col = rank
                                    gi = i_off + li
                                    gj = j_off + lj
                                    gk = k_off + lk
                                    U_entries.append((gi * m + gj, col, 1))
                                    V_entries.append((gj * p + gk, col, 1))
                                    W_entries.append((gk * n + gi, col, 1))
                                    rank += 1

        return U_entries, V_entries, W_entries, rank

    def build_matrices(U_entries, V_entries, W_entries, rank):
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for idx, col, val in U_entries:
            U[idx][col] = val
        for idx, col, val in V_entries:
            V[idx][col] = val
        for idx, col, val in W_entries:
            W[idx][col] = val
        return U, V, W

    def compositions(total, min_part=1):
        """Generate all compositions of total into parts >= min_part, order matters."""
        if total < min_part:
            return
        if total == min_part:
            yield [min_part]
            return
        for first in range(min_part, total + 1):
            if total - first < min_part and total - first > 0:
                continue
            if total - first == 0:
                yield [first]
            else:
                for rest in compositions(total - first, min_part):
                    yield [first] + rest

    def rank_of(n_s, m_s, p_s):
        """Compute rank for given splits without building matrices."""
        total = 0
        for n_b in n_s:
            for m_b in m_s:
                for p_b in p_s:
                    if n_b == 2 and m_b == 2 and p_b == 2:
                        total += 7
                    else:
                        total += n_b * m_b * p_b
        return total

    # Search over all split compositions to find minimum rank
    best_rank = n * m * p  # schoolbook
    best_splits = ([n], [m], [p])

    n_comps = list(compositions(n))
    m_comps = list(compositions(m))
    p_comps = list(compositions(p))

    for nc in n_comps:
        for mc in m_comps:
            for pc in p_comps:
                r = rank_of(nc, mc, pc)
                if r < best_rank:
                    best_rank = r
                    best_splits = (nc, mc, pc)

    n_splits, m_splits, p_splits = best_splits

    # Build the decomposition
    U_entries, V_entries, W_entries, rank = build_decomposition(n_splits, m_splits, p_splits)

    # Deadline guard
    if time.time() > _deadline:
        # Return schoolbook as fallback
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    U, V, W = build_matrices(U_entries, V_entries, W_entries, rank)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
