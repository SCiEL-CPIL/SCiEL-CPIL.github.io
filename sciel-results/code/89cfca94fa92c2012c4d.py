import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 40
    
    def check_c6(adj, u, v):
        """
        Returns True if adding edge (u, v) creates a C6.
        We look for a path of length 5 from u to v in G \ {u, v}.
        u - x1 - x2 - x3 - x4 - v
        x1 in N(u), x4 in N(v).
        Path x1 to x4 of length 3 in G \ {u, v, x1, x4}.
        x1 - x2 - x3 - x4
        """
        nu = [x for x in adj[u] if x != v]
        nv = [x for x in adj[v] if x != u]
        
        for x1 in nu:
            for x4 in nv:
                if x1 == x4:
                    continue
                for x2 in adj[x1]:
                    if x2 in (u, v, x1, x4):
                        continue
                    for x3 in adj[x4]:
                        if x3 in (u, v, x1, x4, x2):
                            continue
                        if x3 in adj[x2]:
                            return True
        return False

    def is_c6_free(adj, n):
        for u in range(n):
            for v in list(adj[u]):
                if u < v:
                    adj[u].remove(v)
                    adj[v].remove(u)
                    if check_c6(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                        return False
                    adj[u].add(v)
                    adj[v].add(u)
        return True

    best_edges = []
    best_count = 0
    
    while time.time() < deadline - 5:
        adj = [set() for _ in range(n)]
        
        # Greedy Phase
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if time.time() > deadline - 10:
                break
            if v not in adj[u]:
                if not check_c6(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
        
        count = sum(len(a) for a in adj) // 2
        
        # Local Search Phase: Try to add any missing edge
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if v not in adj[u]:
                        if not check_c6(adj, u, v):
                            adj[u].add(v)
                            adj[v].add(u)
                            count += 1
                            improved = True
                            break
                if improved:
                    break
        
        if count > best_count:
            best_count = count
            best_edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        best_edges.append([u, v])
        
        # If stuck, we could try swaps, but for now restart is simpler and effective with fast check
        # We can also try to remove a random edge and see if we can add two? No, that reduces count.
        # Just restart with new random order.
        
    if best_count > 0:
        adj = [set() for _ in range(n)]
        for u, v in best_edges:
            adj[u].add(v)
            adj[v].add(u)
        if is_c6_free(adj, n):
            return best_edges
    
    return _fallback()

def _fallback():
    edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([5 * k + a, 5 * k + b])
    return edges
