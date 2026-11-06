"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    q = 5
    n = 31
    
    # Enumerate points of PG(2,5)
    points = []
    for x in range(q):
        for y in range(q):
            points.append([x, y, 1])
    for t in range(q):
        points.append([1, t, 0])
    points.append([0, 1, 0])
    
    # Build polarity graph: edge if dot product is 0 mod q
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dot = (points[i][0]*points[j][0] + points[i][1]*points[j][1] + points[i][2]*points[j][2]) % q
            if dot == 0:
                edges.append([i, j])
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def count_c4():
        c4 = 0
        for i in range(n):
            for j in range(i + 1, n):
                common = len(adj[i] & adj[j])
                c4 += common * (common - 1) // 2
        return c4 // 2
    
    # Adjust to exactly 91 edges
    if len(edges) > 91:
        # Remove edges that contribute most to C4 count
        excess = len(edges) - 91
        for _ in range(excess):
            if time.time() > deadline:
                break
            best_idx = -1
            best_c4_increase = -1
            for idx in range(len(edges)):
                u, v = edges[idx]
                # C4s containing edge (u,v): for each w in adj[u]∩adj[v], 
                # number of x such that x is common neighbor of w and (u or v)
                # Simpler: count C4s that include this edge
                common_w = adj[u] & adj[v]
                c4_with_edge = 0
                for w in common_w:
                    # x must be in (adj[u]∩adj[w]) ∪ (adj[v]∩adj[w]) but not w
                    # Actually for 4-cycle u-w-v-x-u: x in adj[u]∩adj[v]... no
                    # 4-cycle: u-w-x-v-u requires x in adj[w]∩adj[v] and x in adj[u]... 
                    # Let me just estimate: each common neighbor w gives some C4s
                    # For simplicity, use |adj[u]∩adj[v]| as proxy
                    pass
                # Use a simpler metric: remove edge with highest degree product
                if best_idx == -1 or (len(adj[u]) * len(adj[v])) > best_c4_increase:
                    best_c4_increase = len(adj[u]) * len(adj[v])
                    best_idx = idx
            if best_idx >= 0:
                u, v = edges.pop(best_idx)
                adj[u].discard(v)
                adj[v].discard(u)
    
    elif len(edges) < 91:
        deficit = 91 - len(edges)
        random.seed(42)
        for _ in range(deficit):
            if time.time() > deadline:
                break
            for attempt in range(10000):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u < v and v not in adj[u]:
                    edges.append([u, v])
                    adj[u].add(v)
                    adj[v].add(u)
                    break
    
    # Local search: try to reduce C4 by edge swaps
    current_c4 = count_c4()
    
    def try_swap():
        nonlocal current_c4
        if time.time() > deadline:
            return False
        # Try removing an edge and adding another
        for _ in range(50):
            if time.time() > deadline:
                return False
            # Pick a random edge to remove
            if not edges:
                return False
            idx = random.randint(0, len(edges) - 1)
            u, v = edges[idx]
            
            # Remove edge
            adj[u].discard(v)
            adj[v].discard(u)
            edges.pop(idx)
            
            # Find best edge to add
            best_add = None
            best_c4 = current_c4 + 1000000
            candidates = []
            for _ in range(200):
                a = random.randint(0, n - 1)
                b = random.randint(0, n - 1)
                if a < b and b not in adj[a]:
                    candidates.append((a, b))
            
            for a, b in candidates:
                # Estimate C4 change
                # Adding (a,b): new C4s = |adj[a]∩adj[b]| choose 2 + contributions
                common_ab = adj[a] & adj[b]
                new_c4 = 0
                for w in common_ab:
                    # Each w in common gives C(|adj[a]∩adj[w]| - 1 + ...) C4s
                    # Simplified: just count common neighbors
                    pass
                # Use actual count for accuracy if fast enough
                adj[a].add(b)
                adj[b].add(a)
                c4_new = count_c4()
                adj[a].discard(b)
                adj[b].discard(a)
                
                if c4_new < best_c4:
                    best_c4 = c4_new
                    best_add = (a, b)
            
            if best_add and best_c4 < current_c4:
                a, b = best_add
                edges.append([a, b])
                adj[a].add(b)
                adj[b].add(a)
                current_c4 = best_c4
                return True
            else:
                # Restore removed edge
                edges.insert(idx, [u, v])
                adj[u].add(v)
                adj[v].add(u)
                return False
    
    # Run local search with time budget
    no_improve = 0
    while time.time() < deadline:
        if try_swap():
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 50:
                break
    
    return edges
# EVOLVE-BLOCK-END
