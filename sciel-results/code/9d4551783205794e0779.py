import time
import random
import numpy as np

_DEADLINE = 110.0  # seconds, safety margin under 120s

def _enumerate_4cycles(n):
    """Return list of 4-cycles as (a,b,c,d) with a<b<c<d and the cycle a-b-c-d-a.
    For each 4-set {a,b,c,d} (a<b<c<d), the 3 distinct 4-cycles (up to rotation/reversal)
    using all 4 vertices in a cycle are the 3 pairings of the 6 edges into a 4-cycle.
    Actually a 4-cycle on 4 labeled vertices: the cycle uses 4 of the 6 edges.
    The 3 possible 4-cycles (each omits a pair of opposite edges):
      omit (a,c) and (b,d): cycle a-b-c-d-a  -> edges ab,bc,cd,da
      omit (a,b) and (c,d): cycle a-c-b-d-a  -> edges ac,cb,bd,da
      omit (a,d) and (b,c): cycle a-b-d-c-a  -> edges ab,bd,dc,ca
    We represent each by its 4 edge-vertex-pairs.
    """
    cycles = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    # three 4-cycles
                    cycles.append(((a,b),(b,c),(c,d),(d,a)))
                    cycles.append(((a,c),(c,b),(b,d),(d,a)))
                    cycles.append(((a,b),(b,d),(d,c),(c,a)))
    return cycles

def _edge_index(u, v):
    if u > v:
        u, v = v, u
    return u * 20 + v  # u<v, u in 0..19, v in u+1..19 -> index < 400

def construct():
    t0 = time.time()
    n = 20
    random.seed(12345)
    np.random.seed(12345)

    # Precompute 4-cycles and their edge indices
    cycles = _enumerate_4cycles(n)
    cyc_edges = []
    for cyc in cycles:
        cyc_edges.append([_edge_index(u, v) for (u, v) in cyc])
    cyc_edges = np.array(cyc_edges, dtype=np.int32)  # (M,4)
    M = len(cyc_edges)

    # Build edge list for K_n
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            edges.append((u, v))
    E = len(edges)  # 190
    edge_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i

    # Map cycle edge-indices (u*20+v) to edge positions
    # Build a lookup: u*20+v (u<v) -> position
    key_to_pos = {}
    for i, (u, v) in enumerate(edges):
        key_to_pos[u * 20 + v] = i
    cyc_pos = np.array([[key_to_pos[k] for k in row] for row in cyc_edges], dtype=np.int32)  # (M,4)

    def _coloring_to_validity(col):
        # col: array (E,) of colours 0..19
        # For each cycle, check 4 colours distinct
        cc = col[cyc_pos]  # (M,4)
        # distinct iff all pairwise different
        bad = 0
        # compare pairs
        bad += int(np.sum(cc[:,0]==cc[:,1]))
        bad += int(np.sum(cc[:,0]==cc[:,2]))
        bad += int(np.sum(cc[:,0]==cc[:,3]))
        bad += int(np.sum(cc[:,1]==cc[:,2]))
        bad += int(np.sum(cc[:,1]==cc[:,3]))
        bad += int(np.sum(cc[:,2]==cc[:,3]))
        return bad

    # Initial colouring: try a few analytic starts, pick best
    best_col = None
    best_bad = None
    starts = []
    # analytic 1: (u+v)%20
    c = np.array([(u+v) % 20 for (u,v) in edges], dtype=np.int32); starts.append(c)
    # analytic 2: (v-u)%20
    c = np.array([(v-u) % 20 for (u,v) in edges], dtype=np.int32); starts.append(c)
    # analytic 3: (u*7+v*3)%20
    c = np.array([(u*7+v*3) % 20 for (u,v) in edges], dtype=np.int32); starts.append(c)
    # random
    c = np.random.randint(0,20,E).astype(np.int32); starts.append(c)
    for c in starts:
        b = _coloring_to_validity(c)
        if best_bad is None or b < best_bad:
            best_bad = b
            best_col = c.copy()

    col = best_col

    # Local search: repeatedly fix a bad cycle by recoloring one of its edges
    def _cycle_bad_mask(col):
        cc = col[cyc_pos]
        m = np.zeros(M, dtype=bool)
        m |= (cc[:,0]==cc[:,1])
        m |= (cc[:,0]==cc[:,2])
        m |= (cc[:,0]==cc[:,3])
        m |= (cc[:,1]==cc[:,2])
        m |= (cc[:,1]==cc[:,3])
        m |= (cc[:,2]==cc[:,3])
        return m

    def _bad_count(col):
        return int(np.sum(_cycle_bad_mask(col)))

    it = 0
    while time.time() - t0 < _DEADLINE - 5:
        bad_mask = _cycle_bad_mask(col)
        nbad = int(np.sum(bad_mask))
        if nbad == 0:
            break
        # pick a random bad cycle
        bad_idx = np.where(bad_mask)[0]
        ci = int(bad_idx[random.randrange(len(bad_idx))])
        ep = cyc_pos[ci]  # 4 edge positions
        # try recoloring each of the 4 edges to each colour, pick best
        cur = _bad_count(col)
        best_move = None
        best_val = cur
        for e in ep:
            orig = col[e]
            for cval in range(20):
                if cval == orig:
                    continue
                col[e] = cval
                v = _bad_count(col)
                col[e] = orig
                if v < best_val or (v == best_val and best_move is None and random.random() < 0.1):
                    if v < best_val:
                        best_move = (e, cval); best_val = v
                    elif v == best_val and best_move is None and random.random() < 0.3:
                        best_move = (e, cval); best_val = v
        if best_move is not None and best_val < cur:
            col[best_move[0]] = best_move[1]
            it += 1
        else:
            # no improving move on this cycle; try a random perturbation to escape
            if it % 50 == 0 and it > 0:
                e = int(random.randrange(E))
                col[e] = int(random.randrange(20))
            else:
                # move to next bad cycle
                pass
        if time.time() - t0 > _DEADLINE - 8:
            break

    # Ensure validity: if still bad, remove edges greedily until valid
    def _is_valid(col, present):
        # present: bool (E,)
        for ci in range(M):
            ep = cyc_pos[ci]
            if not all(present[e] for e in ep):
                continue
            ccs = [col[e] for e in ep]
            if len(set(ccs)) != 4:
                return False
        return True

    present = np.ones(E, dtype=bool)
    if not _is_valid(col, present):
        # greedy removal: while invalid, remove edge in most bad cycles
        def _bad_cycle_count_per_edge(col, present):
            cnt = np.zeros(E, dtype=np.int32)
            for ci in range(M):
                ep = cyc_pos[ci]
                if not all(present[e] for e in ep):
                    continue
                ccs = [col[e] for e in ep]
                if len(set(ccs)) != 4:
                    for e in ep:
                        cnt[e] += 1
            return cnt
        guard = 0
        while not _is_valid(col, present) and time.time() - t0 < _DEADLINE - 2:
            cnt = _bad_cycle_count_per_edge(col, present)
            # among present edges, pick max count
            cand = np.where(present & (cnt > 0))[0]
            if len(cand) == 0:
                # no edge can fix; remove a random present edge
                pres = np.where(present)[0]
                if len(pres) == 0:
                    break
                e = int(pres[0])
            else:
                e = int(cand[np.argmax(cnt[cand])])
            present[e] = False
            guard += 1
            if guard > E + 5:
                break

    # Build result
    result = []
    for i, (u, v) in enumerate(edges):
        if present[i]:
            result.append([u, v, int(col[i])])
    return result
