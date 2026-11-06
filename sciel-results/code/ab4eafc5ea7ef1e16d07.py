import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 seconds margin before 150s kill

    def has_c5(n, adj):
        """Check if graph has a 5-cycle. Returns True if C5 exists."""
        for v in range(n):
            nbrs = list(adj[v])
            for i in range(len(nbrs)):
                u = nbrs[i]
                for j in range(i + 1, len(nbrs)):
                    w = nbrs[j]
                    # Need path u-x-y-w in G-{v} with x,y distinct, x!=w, y!=u
                    for x in adj[u]:
                        if x == v or x == w:
                            continue
                        for y in adj[w]:
                            if y == v or y == u or y == x:
                                continue
                            if y in adj[x]:
                                # v-u-x-y-w-v is a C5
                                return True
        return False

    def exact_alpha(n, adj, k):
        """Check if independent set of size k exists. Returns True if alpha >= k."""
        # Branch and bound
        # Use recursive search with pruning
        def search(candidates, depth, current_size):
            if current_size + len(candidates) < k:
                return False
            if current_size >= k:
                return True
            if not candidates:
                return False
            
            # Pick a vertex to branch on
            # Heuristic: pick vertex with highest degree in candidate subgraph
            best_v = candidates[0]
            best_deg = -1
            for v in candidates:
                deg = sum(1 for u in candidates if u != v and u in adj[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            # Branch 1: include best_v
            new_candidates = [v for v in candidates if v != best_v and v not in adj[best_v]]
            if search(new_candidates, depth + 1, current_size + 1):
                return True
            
            # Branch 2: exclude best_v
            remaining = [v for v in candidates if v != best_v]
            if search(remaining, depth + 1, current_size):
                return True
            
            return False
        
        return search(list(range(n)), 0, 0)

    def is_valid(n, edges):
        """Check C5-free and alpha < 9 exactly."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        if has_c5(n, adj):
            return False
        if exact_alpha(n, adj, 9):
            return False
        return True

    # Base: 8 disjoint K4s (n=32, alpha=8, C5-free)
    base_edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                base_edges.append([base + i, base + j])
    
    best_n = 32
    best_edges = list(base_edges)
    
    # Try to extend to n=33, 34, 35
    for target_n in [33, 34, 35]:
        if time.time() > deadline:
            break
        
        found = False
        attempts = 0
        while time.time() < deadline and not found:
            attempts += 1
            
            if attempts % 1000 == 0:
                # Periodic check
                pass
            
            # Strategy: add new vertices with specific connection patterns
            # to maintain C5-freeness and keep alpha < 9
            
            edges = list(base_edges)
            
            for new_v in range(32, target_n):
                # Connect new_v to a small set of existing vertices
                # To avoid creating C5, limit connections
                # Try different strategies based on attempt number
                
                strategy = attempts % 5
                
                if strategy == 0:
                    # Connect to 1 vertex in a random K4
                    k4_idx = random.randint(0, 7)
                    target = 4 * k4_idx + random.randint(0, 3)
                    edges.append([new_v, target])
                elif strategy == 1:
                    # Connect to 2 vertices in the same K4
                    k4_idx = random.randint(0, 7)
                    targets = random.sample(range(4 * k4_idx, 4 * k4_idx + 4), 2)
                    for t in targets:
                        edges.append([new_v, t])
                elif strategy == 2:
                    # Connect to 2 vertices in different K4s
                    k4_idx1 = random.randint(0, 7)
                    k4_idx2 = random.randint(0, 7)
                    while k4_idx2 == k4_idx1:
                        k4_idx2 = random.randint(0, 7)
                    t1 = 4 * k4_idx1 + random.randint(0, 3)
                    t2 = 4 * k4_idx2 + random.randint(0, 3)
                    edges.append([new_v, t1])
                    edges.append([new_v, t2])
                elif strategy == 3:
                    # Connect to 3 vertices in same K4
                    k4_idx = random.randint(0, 7)
                    targets = random.sample(range(4 * k4_idx, 4 * k4_idx + 4), 3)
                    for t in targets:
                        edges.append([new_v, t])
                else:
                    # Connect to 4 vertices (all in one K4) - makes new_v part of K5-like structure
                    k4_idx = random.randint(0, 7)
                    targets = list(range(4 * k4_idx, 4 * k4_idx + 4))
                    for t in targets:
                        edges.append([new_v, t])
            
            # Check validity
            if is_valid(target_n, edges):
                best_n = target_n
                best_edges = edges
                found = True
    
    return {'n': best_n, 'edges': best_edges}
