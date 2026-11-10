"""Cayley graph on SL(2,16) for cage construction."""
import time
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    deadline = time.time() + 135  # safety margin before 150s kill

    q = 16

    # Build GF(16) with primitive polynomial x^4 + x + 1
    def gf16_mul(a, b):
        result = 0
        for i in range(4):
            if b & (1 << i):
                result ^= a
            a <<= 1
            if a & 0x10:
                a ^= 0x0B
        return result & 0x0F

    mul_tbl = [[gf16_mul(i, j) for j in range(16)] for i in range(16)]
    inv_tbl = [0] * 16
    for i in range(1, 16):
        for j in range(1, 16):
            if mul_tbl[i][j] == 1:
                inv_tbl[i] = j
                break

    # Build SL(2, 16)
    elements = []
    elem_idx = {}
    for a in range(16):
        for b in range(16):
            for c in range(16):
                if a == 0:
                    continue
                bc = mul_tbl[b][c]
                rhs = 1 ^ bc
                d = mul_tbl[rhs][inv_tbl[a]]
                mat = (a, b, c, d)
                elements.append(mat)
                elem_idx[mat] = len(elements) - 1

    n = len(elements)
    identity = (1, 0, 0, 1)
    id_i = elem_idx[identity]

    def mat_mul(m1, m2):
        a1, b1, c1, d1 = m1
        a2, b2, c2, d2 = m2
        return (
            mul_tbl[a1][a2] ^ mul_tbl[b1][c2],
            mul_tbl[a1][b2] ^ mul_tbl[b1][d2],
            mul_tbl[c1][a2] ^ mul_tbl[d1][c2],
            mul_tbl[c1][b2] ^ mul_tbl[d1][d2],
        )

    def mat_inv(m):
        a, b, c, d = m
        return (d, b, c, a)  # In char 2, -x = x, det=1 so inv = [[d,-b],[-c,a]] = [[d,b],[c,a]]

    # Precompute right multiplication table for speed
    # right_mult[i][j] = elem_idx[elements[i] * elements[j]]
    # 4080 * 4080 = 16,646,400 entries. Each is a small int. ~66MB. Might be OK.
    # Actually let's compute on the fly to save memory, but cache the generator multiplications
    
    # For each generator, precompute its action
    # Find involutions
    involutions = []
    for i in range(n):
        if i == id_i:
            continue
        sq = mat_mul(elements[i], elements[i])
        if sq == identity:
            involutions.append(i)

    # Find non-involution, non-identity elements as s candidates
    s_candidates = [i for i in range(n) if i != id_i and i not in set(involutions)]

    def girth_ok(s_idx, t_idx, target_girth):
        """Check if Cayley graph with {s, s^{-1}, t} has girth >= target_girth."""
        s = elements[s_idx]
        s_inv = mat_inv(s)
        s_inv_i = elem_idx.get(s_inv)
        if s_inv_i is None:
            return False
        
        gens = [s_idx, s_inv_i, t_idx]
        
        # BFS to find girth
        dist = [-1] * n
        dist[id_i] = 0
        queue = deque([id_i])
        min_cycle = target_girth + 1  # We only care if it's < target_girth
        
        while queue:
            if time.time() > deadline - 15:
                return False  # timeout, be conservative
            u = queue.popleft()
            du = dist[u]
            if du >= target_girth // 2:
                # We've explored enough; any cycle found so far has length >= 2*du+1 >= target_girth
                # Actually we need to be more careful
                pass
            for gi in gens:
                v = elem_idx[mat_mul(elements[u], elements[gi])]
                if dist[v] == -1:
                    dist[v] = du + 1
                    queue.append(v)
                else:
                    if dist[v] <= du:
                        # Non-tree edge (or same level)
                        cycle_len = du + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            if min_cycle < target_girth:
                                return False
        
        return min_cycle >= target_girth

    # Try combinations
    best = None
    # Try first 50 s candidates and first 50 involutions
    for si in s_candidates[:50]:
        if time.time() > deadline - 40:
            break
        for ti in involutions[:50]:
            if time.time() > deadline - 40:
                break
            if girth_ok(si, ti, g):
                # Build edges
                s = elements[si]
                s_inv = mat_inv(s)
                s_inv_i = elem_idx[s_inv]
                edges = []
                for i in range(n):
                    for gi in [si, s_inv_i, ti]:
                        v = elem_idx[mat_mul(elements[i], elements[gi])]
                        if i < v:
                            edges.append((i, v))
                best = (n, edges)
                break
        if best is not None:
            break

    if best is None:
        # Fallback to LUW
        return _luw(k, g)

    return best


def _luw(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
    else:
        add, sub, mul = (lambda x,y: (x+y)%q), (lambda x,y: (x-y)%q), (lambda x,y: x*y%q)
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    def nbrs(vtx):
        side, x = vtx
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    start = (0, (None,)+(0,)*d)
    idx, order, edges = {start:0}, [start], []
    for vtx in order:
        for o in nbrs(vtx):
            if o not in idx:
                idx[o] = len(order)
                order.append(o)
            if vtx[0] == 0:
                edges.append((idx[vtx], idx[o]))
    return len(order), edges

# EVOLVE-BLOCK-END
