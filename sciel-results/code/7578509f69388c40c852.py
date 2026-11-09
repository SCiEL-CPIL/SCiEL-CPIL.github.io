import random
import time
import sys

_START_TIME = time.time()
DEADLINE = 105.0

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return DEADLINE - _elapsed()

def _is_k4_free(adj, n=18):
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            common = adj[a] & adj[b]
            c = common
            while c:
                lsb = c & -c
                x = lsb.bit_length() - 1
                c -= lsb
                if (adj[x] & common) != 0:
                    return False
    return True

def _max_triangle_free_induced(adj, n=18):
    best = [0]
    
    def popcount(x):
        return bin(x).count('1')
    
    order = list(range(n))
    order.sort(key=lambda v: bin(adj[v]).count('1'), reverse=True)
    
    cn = [[0]*n for _ in range(n)]
    for u in range(n):
        for v in range(u+1, n):
            c = adj[u] & adj[v]
            cn[u][v] = c
            cn[v][u] = c
    
    def bb(idx, current_mask, forbidden_mask):
        if _elapsed() > DEADLINE:
            return
        
        if idx == n:
            sz = popcount(current_mask)
            if sz > best[0]:
                best[0] = sz
            return
        
        avail = 0
        for i in range(idx, n):
            v = order[i]
            if not (forbidden_mask >> v) & 1:
                avail += 1
        
        if popcount(current_mask) + avail <= best[0]:
            return
        
        v = order[idx]
        bit = 1 << v
        
        bb(idx + 1, current_mask, forbidden_mask)
        
        if not (forbidden_mask & bit):
            nf = forbidden_mask
            cm = current_mask
            while cm:
                lsb = cm & -cm
                u = lsb.bit_length() - 1
                cm -= lsb
                if (adj[v] >> u) & 1:
                    nf |= cn[v][u]
            bb(idx + 1, current_mask | bit, nf)
    
    bb(0, 0, 0)
    return best[0]

def construct():
    n = 18
    random.seed(42)
    
    # K6,6,6 initial
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    adj = [0] * n
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
    
    current_adj = adj[:]
    best_adj = adj[:]
    current_score = _max_triangle_free_induced(current_adj, n)
    best_score = current_score
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    while _time_left() > 5:
        # Mix of single flips and edge swaps
        if random.random() < 0.3:
            # Edge swap
            u1, v1 = random.choice(all_edges)
            u2, v2 = random.choice(all_edges)
            if (u1, v1) == (u2, v2):
                continue
            
            new_adj = current_adj[:]
            # Toggle edge (u1, v1)
            if (new_adj[u1] >> v1) & 1:
                new_adj[u1] &= ~(1 << v1)
                new_adj[v1] &= ~(1 << u1)
            else:
                new_adj[u1] |= (1 << v1)
                new_adj[v1] |= (1 << u1)
            
            # Toggle edge (u2, v2)
            if (new_adj[u2] >> v2) & 1:
                new_adj[u2] &= ~(1 << v2)
                new_adj[v2] &= ~(1 << u2)
            else:
                new_adj[u2] |= (1 << v2)
                new_adj[v2] |= (1 << u2)
        else:
            # Single flip
            u, v = random.choice(all_edges)
            new_adj = current_adj[:]
            if (new_adj[u] >> v) & 1:
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
            else:
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
        
        if not _is_k4_free(new_adj, n):
            continue
        
        new_score = _max_triangle_free_induced(new_adj, n)
        
        if new_score < best_score:
            best_score = new_score
            best_adj = new_adj
            current_adj = new_adj
        elif new_score == current_score:
            current_adj = new_adj
        elif random.random() < 0.05:
            current_adj = new_adj
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
