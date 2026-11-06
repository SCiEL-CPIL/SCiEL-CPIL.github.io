import random
import time
import math

N = 18

def is_k4_free(adj):
    """Check K4-free using bitmask adjacency."""
    for a in range(N):
        for b in range(a + 1, N):
            if not (adj[a] >> b) & 1:
                continue
            for c in range(b + 1, N):
                if not ((adj[a] >> c) & 1 and (adj[b] >> c) & 1):
                    continue
                common = adj[a] & adj[b] & adj[c]
                mask = ((1 << N) - 1) ^ ((1 << (c + 1)) - 1)
                if common & mask:
                    return False
    return True

def max_triangle_free_size(adj):
    """Exact max triangle-free induced subgraph size via min vertex cover of triangles."""
    triangles = []
    for a in range(N):
        for b in range(a + 1, N):
            if not (adj[a] >> b) & 1:
                continue
            common = adj[a] & adj[b]
            mask = ((1 << N) - 1) ^ ((1 << (b + 1)) - 1)
            cands = common & mask
            while cands:
                lsb = cands & (-cands)
                c = lsb.bit_length() - 1
                triangles.append((a, b, c))
                cands ^= lsb

    if not triangles:
        return N

    tri_masks = [((1 << a) | (1 << b) | (1 << c)) for (a, b, c) in triangles]

    def greedy_vc():
        remaining = set(range(len(tri_masks)))
        cover = 0
        while remaining:
            counts = {}
            for ti in remaining:
                m = tri_masks[ti]
                while m:
                    lsb = m & (-m)
                    v = lsb.bit_length() - 1
                    counts[v] = counts.get(v, 0) + 1
                    m ^= lsb
            v = max(counts, key=counts.get)
            cover |= (1 << v)
            remaining = {ti for ti in remaining if not (tri_masks[ti] & (1 << v))}
        return bin(cover).count('1')

    ub = greedy_vc()
    best = [ub]

    def solve(idx, cover, deadline):
        if time.time() > deadline:
            return
        if bin(cover).count('1') >= best[0]:
            return
        while idx < len(tri_masks) and (tri_masks[idx] & cover):
            idx += 1
        if idx == len(tri_masks):
            cnt = bin(cover).count('1')
            if cnt < best[0]:
                best[0] = cnt
            return
        m = tri_masks[idx]
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m ^= lsb
            solve(idx + 1, cover | (1 << v), deadline)

    deadline = time.time() + 2.0
    solve(0, 0, deadline)

    return N - best[0]

def construct():
    start_time = time.time()
    deadline = start_time + 130

    def make_k666():
        adj = [0] * N
        for i in range(3):
            for j in range(i + 1, 3):
                for a in range(i * 6, (i + 1) * 6):
                    for b in range(j * 6, (j + 1) * 6):
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        return adj

    current = make_k666()
    best_adj = current[:]
    best_score = max_triangle_free_size(current)
    current_score = best_score

    T = 1.0
    cooling = 0.9995

    while time.time() < deadline:
        a = random.randint(0, N - 1)
        b = random.randint(0, N - 1)
        if a == b:
            continue

        new_adj = current[:]
        if (new_adj[a] >> b) & 1:
            new_adj[a] &= ~(1 << b)
            new_adj[b] &= ~(1 << a)
        else:
            new_adj[a] |= (1 << b)
            new_adj[b] |= (1 << a)

        if not is_k4_free(new_adj):
            continue

        new_score = max_triangle_free_size(new_adj)
        delta = new_score - current_score

        if delta <= 0 or random.random() < math.exp(-delta / T):
            current = new_adj
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = new_adj[:]

        T *= cooling
        if T < 0.005:
            T = 0.005

    edges = []
    for a in range(N):
        for b in range(a + 1, N):
            if (best_adj[a] >> b) & 1:
                edges.append([a, b])
    return edges
