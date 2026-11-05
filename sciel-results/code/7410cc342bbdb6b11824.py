"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook baseline
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1

    # Try Strassen 2x2x2 substitutions on disjoint blocks
    # Verify a candidate Strassen decomposition for 2x2x2
    def make_strassen_block(i0, j0, k0):
        """Return list of (u, v, w) dicts for a 2x2x2 Strassen block."""
        # Local indices: a,b in {0,1} for i,j; c in {0,1} for k
        # u indexed by (a,b), v by (b,c), w by (c,a)
        # 7 Strassen terms:
        terms = [
            # M1=(A00+A11)(B00+B11)
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            # M2=(A10+A11)B00
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,0,1)]),
            # M3=A00(B10-B11)
            ([(0,0,1)], [(1,0,1),(1,1,-1)], [(1,0,1)]),
            # M4=A11(B01-B11)
            ([(1,1,1)], [(0,1,1),(1,1,-1)], [(1,1,1)]),
            # M5=(A00+A01)B11
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(1,1,1)]),
            # M6=(A10-A00)(B00+B01)
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,0,1)]),
            # M7=(A01-A11)(B10+B11)
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(1,1,1)]),
        ]
        # Convert to global
        result = []
        for u_list, v_list, w_list in terms:
            u = {(i0+a, j0+b): val for (a, b, val) in u_list}
            v = {(j0+b, k0+c): val for (b, c, val) in v_list}
            w = {(k0+c, i0+a): val for (c, a, val) in w_list}
            result.append((u, v, w))
        return result

    # Verify Strassen block
    def verify_block(block_terms):
        T = {}
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    T[(i,j,k)] = 0
        for u, v, w in block_terms:
            for (ii, jj), uv in u.items():
                for (jj2, kk), vv in v.items():
                    if jj != jj2:
                        continue
                    for (kk2, ii2), ww in w.items():
                        if kk != kk2 or ii != ii2:
                            continue
                        T[(ii, jj, kk)] += uv * vv * ww
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    if T[(i,j,k)] != 1:
                        return False
        return True

    strassen_ok = verify_block(make_strassen_block(0, 0, 0))
    if not strassen_ok:
        # Strassen verification failed, return schoolbook
        return {"U": U, "V": V, "W": W}

    # Find disjoint 2x2x2 blocks
    i_pairs = [(0,1), (1,2)]
    j_pairs = [(0,1), (1,2), (2,3)]
    k_pairs = [(0,1), (2,3), (4,5)]

    all_blocks = [(ip, jp, kp) for ip in i_pairs for jp in j_pairs for kp in k_pairs]

    # Greedy selection of disjoint blocks
    selected = []
    for block in all_blocks:
        ip, jp, kp = block
        compatible = True
        for (ip2, jp2, kp2) in selected:
            if set(ip) & set(ip2) or set(jp) & set(jp2) or set(kp) & set(kp2):
                compatible = False
                break
        if compatible:
            selected.append(block)

    # Build new term list
    new_terms = []
    for block in selected:
        ip, jp, kp = block
        new_terms.extend(make_strassen_block(ip[0], jp[0], kp[0]))

    # Add remaining schoolbook terms not in any selected block
    for i in range(n):
        for j in range(m):
            for k in range(p):
                in_block = False
                for (ip, jp, kp) in selected:
                    if i in ip and j in jp and k in kp:
                        in_block = True
                        break
                if not in_block:
                    new_terms.append((
                        {(i, j): 1},
                        {(j, k): 1},
                        {(k, i): 1}
                    ))

    rank_new = len(new_terms)
    if rank_new >= rank:
        return {"U": U, "V": V, "W": W}

    U2 = [[0] * rank_new for _ in range(n * m)]
    V2 = [[0] * rank_new for _ in range(m * p)]
    W2 = [[0] * rank_new for _ in range(p * n)]
    for c, (u, v, w) in enumerate(new_terms):
        for (ii, jj), val in u.items():
            U2[ii*m+jj][c] = val
        for (jj, kk), val in v.items():
            V2[jj*p+kk][c] = val
        for (kk, ii), val in w.items():
            W2[kk*n+ii][c] = val

    if _verify(U2, V2, W2, n, m, p):
        return {"U": U2, "V": V2, "W": W2}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
