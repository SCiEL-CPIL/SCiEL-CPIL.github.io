import time
import random
import math

N = 18
FULL = (1 << N) - 1
DEADLINE_MARGIN = 10  # safety margin before the 150 s wall-clock kill


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    rng = random.Random(42)

    def is_k4_free(adj_mask):
        for a in range(N):
            for b in range(a + 1, N):
                if (adj_mask[a] >> b) & 1:
                    common = adj_mask[a] & adj_mask[b]
                    common &= ~((1 << a) | (1 << b))
                    m = common
                    # Check if any two vertices in common are adjacent
                    while m:
                        c = (m & -m).bit_length() - 1
                        m &= m - 1
                        if (adj_mask[c] & m) != 0:
                            return False
        return True

    def max_tri_free(adj_mask):
        """Exact size of the largest triangle-free induced subgraph (branch & bound)."""
        # Find triangles
        tris = []
        for a in range(N):
            for b in range(a + 1, N):
                if (adj_mask[a] >> b) & 1:
                    common = adj_mask[a] & adj_mask[b]
                    common &= ~((1 << a) | (1 << b))
                    m = common
                    while m:
                        c = (m & -m).bit_length() - 1
                        if c > b:
                            tris.append((a, b, c))
                        m &= m - 1
        
        if not tris:
            return N
            
        best = [0]
        tri_masks = [(1 << a) | (1 << b) | (1 << c) for (a, b, c) in tris]
        
        def bnb(avail):
            cnt = avail.bit_count()
            if cnt <= best[0]:
                return
            
            # Find a triangle in avail
            tri = None
            for tm in tri_masks:
                if (tm & avail) == tm:
                    tri = tm
                    break
                    
            if tri is None:
                best[0] = cnt
                return
                
            a_bit = tri & -tri
            a = a_bit.bit_length() - 1
            rest = tri ^ a_bit
            b_bit = rest & -rest
            b = b_bit.bit_length() - 1
            c_bit = tri ^ a_bit ^ b_bit
            c = c_bit.bit_length() - 1
            
            # Heuristic: remove vertex with highest degree in avail
            da = (avail & adj_mask[a]).bit_count()
            db = (avail & adj_mask[b]).bit_count()
            dc = (avail & adj_mask[c]).bit_count()
            
            verts = sorted([(da, a), (db, b), (dc, c)], reverse=True)
            
            for _, v in verts:
                bnb(avail & ~(1 << v))
                if cnt - 1 <= best[0]:
                    return

        bnb(FULL)
        return best[0]

    def make_turan_3():
        adj_mask = [0] * N
        for i in range(N):
            for j in range(i + 1, N):
                if i // 6 != j // 6:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        return adj_mask

    def random_k4_free_graph(density=0.5):
        adj_mask = [0] * N
        # Random order to break symmetry
        order = list(range(N))
        rng.shuffle(order)
        for i in range(N):
            for j in range(i + 1, N):
                u, v = order[i], order[j]
                if rng.random() < density:
                    # Check K4
                    common = adj_mask[u] & adj_mask[v]
                    common &= ~((1 << u) | (1 << v))
                    m = common
                    ok = True
                    while m:
                        w = (m & -m).bit_length() - 1
                        m &= m - 1
                        if (adj_mask[w] & m) != 0:
                            ok = False
                            break
                    if ok:
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
        return adj_mask

    def flip(adj_mask, u, v):
        adj_mask[u] ^= (1 << v)
        adj_mask[v] ^= (1 << u)

    best_score = N + 1
    best_mask = None

    seeds = [make_turan_3()]
    for d in [0.4, 0.5, 0.6, 0.7, 0.8]:
        seeds.append(random_k4_free_graph(d))

    # Run restarts
    for r in range(30):
        if time.time() >= deadline:
            break
        if r < len(seeds):
            adj = seeds[r][:]
        else:
            adj = random_k4_free_graph(rng.random() * 0.4 + 0.4)
            
        if not is_k4_free(adj):
            continue
            
        cur = max_tri_free(adj)
        
        # Local Search
        improved = True
        steps = 0
        while improved and time.time() < deadline and steps < 200:
            improved = False
            steps += 1
            # Try random flips
            u, v = rng.sample(range(N), 2)
            flip(adj, u, v)
            if is_k4_free(adj):
                new_score = max_tri_free(adj)
                if new_score < cur:
                    cur = new_score
                    improved = True
                else:
                    flip(adj, u, v)
            else:
                flip(adj, u, v)
                
        if cur < best_score:
            best_score = cur
            best_mask = adj[:]

    if best_mask is None:
        best_mask = make_turan_3()

    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (best_mask[i] >> j) & 1:
                edges.append([i, j])
    return edges
