import time
import random

N = 15

def _is_valid(adj, deg=None):
    if deg is None:
        deg = [bin(adj[i]).count('1') for i in range(N)]
    for u in range(N):
        seen = set()
        au = adj[u]
        while au:
            bit = au & -au
            v = bit.bit_length() - 1
            au ^= bit
            d = deg[v]
            if d in seen:
                return False
            seen.add(d)
    return True

def _count_edges(adj):
    total = 0
    for i in range(N):
        total += bin(adj[i]).count('1')
    return total // 2

def _k78_baseline():
    adj = [0] * N
    for u in range(7):
        for v in range(7, 15):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return adj

def construct():
    start = time.time()
    deadline = start + 80.0
    
    # Baseline
    best_adj = _k78_baseline()
    best_deg = [bin(best_adj[i]).count('1') for i in range(N)]
    best_count = _count_edges(best_adj)
    
    # Local Search
    rng = random.Random(42)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try random edge swap
        for _ in range(1000):
            if time.time() > deadline:
                break
            
            # Pick two edges to swap
            # Or just try to add an edge
            # Let's try add edge
            u = rng.randint(0, N-1)
            v = rng.randint(0, N-1)
            if u == v or u > v:
                continue
            if adj[u] & (1 << v): # Typo, should be best_adj
                pass
            
            # Corrected:
            if best_adj[u] & (1 << v):
                continue
            
            new_adj = best_adj[:]
            new_deg = best_deg[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            new_deg[u] += 1
            new_deg[v] += 1
            
            if _is_valid(new_adj, new_deg):
                best_adj = new_adj
                best_deg = new_deg
                best_count += 1
                improved = True
                break
                
            # Try swap
            # Pick existing edge (a,b) and non-edge (c,d)
            a = rng.randint(0, N-1)
            b = rng.randint(0, N-1)
            if a == b or a > b:
                continue
            if not (best_adj[a] & (1 << b)):
                continue
                
            c = rng.randint(0, N-1)
            d = rng.randint(0, N-1)
            if c == d or c > d:
                continue
            if (c, d) == (a, b):
                continue
            if best_adj[c] & (1 << d):
                continue
                
            new_adj = best_adj[:]
            new_deg = best_deg[:]
            new_adj[a] &= ~(1 << b)
            new_adj[b] &= ~(1 << a)
            new_adj[c] |= (1 << d)
            new_adj[d] |= (1 << c)
            new_deg[a] -= 1
            new_deg[b] -= 1
            new_deg[c] += 1
            new_deg[d] += 1
            
            if _is_valid(new_adj, new_deg):
                best_adj = new_adj
                best_deg = new_deg
                # Count unchanged
                improved = True
                break

    # Convert to list
    edges = []
    for i in range(N):
        ai = best_adj[i]
        while ai:
            bit = ai & -ai
            j = bit.bit_length() - 1
            ai ^= bit
            if i < j:
                edges.append([int(i), int(j)])
                
    return edges
