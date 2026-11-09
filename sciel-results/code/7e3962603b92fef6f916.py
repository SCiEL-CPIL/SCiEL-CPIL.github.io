import time
import random
import itertools

N = 18
ALL_EDGES = list(itertools.combinations(range(N), 2))
EDGE_IDX = {e: i for i, e in enumerate(ALL_EDGES)}
NE = len(ALL_EDGES)

def construct():
    deadline = time.time() + 140.0
    rng = random.Random(12345)
    
    def is_k4_free_add(adj, u, v):
        """Check if adding edge (u,v) to adj keeps K4-free. Assumes (u,v) not yet in adj."""
        common = adj[u] & adj[v]
        c = common
        while c:
            w_bit = (c & -c).bit_length() - 1
            if adj[w_bit] & common & ~(1 << w_bit):
                return False
            c &= c - 1
        return True
    
    def max_triangle_free(adj, time_limit=None):
        """Find max size of triangle-free induced subgraph."""
        best = [0]
        full_set = (1 << N) - 1
        
        def is_tf(s):
            for u in range(N):
                if not (s >> u) & 1:
                    continue
                for v in range(u + 1, N):
                    if not (s >> v) & 1:
                        continue
                    if (adj[u] >> v) & 1:
                        if adj[u] & adj[v] & s:
                            return False
            return True
        
        def bnb(remaining, current):
            if time.time() > deadline - 5.0:
                return
            if current.bit_count() + remaining.bit_count() <= best[0]:
                return
            if remaining == 0:
                if current.bit_count() > best[0]:
                    best[0] = current.bit_count()
                return
            
            # Pick vertex with highest degree in current|remaining
            v = -1
            best_deg = -1
            temp = remaining
            while temp:
                v_bit = (temp & -temp).bit_length() - 1
                deg = (adj[v_bit] & (current | remaining)).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    v = v_bit
                temp &= temp - 1
            
            if v < 0:
                return
                
            new_rem = remaining & ~(1 << v)
            
            # Try including v
            if is_tf(current | (1 << v)):
                bnb(new_rem, current | (1 << v))
            
            # Try excluding v
            bnb(new_rem, current)
        
        bnb(full_set, 0)
        return best[0]
    
    def gen_initial():
        adj = [0] * N
        edges = list(range(NE))
        rng.shuffle(edges)
        for ei in edges:
            u, v = ALL_EDGES[ei]
            if is_k4_free_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    best_adj = None
    best_score = 18
    
    for _ in range(40):
        if time.time() > deadline - 40.0:
            break
        adj = gen_initial()
        s = max_triangle_free(adj)
        if s < best_score:
            best_score = s
            best_adj = adj[:]
    
    if best_adj is not None and time.time() < deadline - 20.0:
        adj = best_adj[:]
        cur_score = max_triangle_free(adj)
        temp = 3.0
        cool = 0.9995
        
        while time.time() < deadline - 8.0:
            u = rng.randint(0, N - 1)
            v = rng.randint(0, N - 1)
            if u == v:
                continue
            if v < u:
                u, v = v, u
            
            adj_u_old = adj[u]
            adj_v_old = adj[v]
            
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                if not is_k4_free_add(adj, u, v):
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            new_score = max_triangle_free(adj)
            
            if new_score <= cur_score or rng.random() < 2.718281828 ** (-(new_score - cur_score) / max(temp, 0.01)):
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj[:]
            else:
                adj[u] = adj_u_old
                adj[v] = adj_v_old
            
            temp *= cool
    
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
