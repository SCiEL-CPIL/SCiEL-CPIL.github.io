import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n = 16
    
    # Build Clebsch graph: vertices 0..15, edge (u,v) if popcount(u^v) in {1,4}
    def build_clebsch():
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                xor = u ^ v
                dist = bin(xor).count('1')
                if dist == 1 or dist == 4:
                    edges.add(frozenset((u, v)))
        return edges
    
    clebsch = build_clebsch()
    
    # Convert to adjacency lists
    def edges_to_adj(edge_set):
        adj = [set() for _ in range(n)]
        for e in edge_set:
            u, v = tuple(e)
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def is_triangle_free_adj(adj):
        for u in range(n):
            for v in adj[u]:
                for w in adj[v]:
                    if w != u and w in adj[u]:
                        return False
        return True
    
    def is_3_colorable_fast(adj):
        """Fast 3-colorability check with forward checking and MRV heuristic."""
        color = [-1] * n
        # Precompute neighbors
        neighbors = [list(adj[v]) for v in range(n)]
        
        def backtrack(idx):
            if idx == n:
                return True
            
            # MRV: pick uncolored vertex with fewest available colors
            best_v = -1
            best_available = None
            best_count = 4  # more than 3
            
            for v in range(n):
                if color[v] != -1:
                    continue
                used_colors = set(color[u] for u in neighbors[v] if color[u] != -1)
                available = [c for c in range(3) if c not in used_colors]
                if len(available) < best_count:
                    best_count = len(available)
                    best_v = v
                    best_available = available
                    if best_count == 0:
                        return False
            
            if best_v == -1:
                return True  # all colored
            
            for c in best_available:
                color[best_v] = c
                if backtrack(idx + 1):
                    return True
                color[best_v] = -1
            
            return False
        
        return backtrack(0)
    
    # Verify Clebsch is valid
    adj = edges_to_adj(clebsch)
    assert is_triangle_free_adj(adj), "Clebsch should be triangle-free"
    assert not is_3_colorable_fast(adj), "Clebsch should not be 3-colorable"
    
    best_edges = set(clebsch)
    best_adj = [set(s) for s in adj]
    best_count = len(best_edges)
    
    # Simulated annealing
    def get_missing_edges(edge_set):
        return [(a, b) for a in range(n) for b in range(a + 1, n) 
                if frozenset((a, b)) not in edge_set]
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph triangle-free."""
        for w in adj[u]:
            if w in adj[v]:
                return False
        return True
    
    # Initial temperature
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    current_adj = [set(s) for s in best_adj]
    current_edges = set(best_edges)
    current_count = best_count
    
    for step in range(50000):
        if time.time() > deadline:
            break
        
        # Generate a move: add, remove, or swap an edge
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add an edge
            missing = get_missing_edges(current_edges)
            if not missing:
                continue
            u, v = random.choice(missing)
            if not can_add_edge(current_adj, u, v):
                continue
            
            # Apply move
            new_adj = [set(s) for s in current_adj]
            new_adj[u].add(v)
            new_adj[v].add(u)
            new_edges = set(current_edges)
            new_edges.add(frozenset((u, v)))
            
            # Check if valid (not 3-colorable)
            if is_3_colorable_fast(new_adj):
                # Invalid move: only accept if temperature is high enough
                delta = -10  # penalty for invalid
                if random.random() < min(1.0, math.exp(-delta / T)):
                    current_adj = new_adj
                    current_edges = new_edges
                    current_count = len(current_edges)
            else:
                # Valid move: always accept if better, otherwise with probability
                if len(new_edges) > current_count:
                    current_adj = new_adj
                    current_edges = new_edges
                    current_count = len(current_edges)
                    if current_count > best_count:
                        best_adj = [set(s) for s in current_adj]
                        best_edges = set(current_edges)
                        best_count = current_count
                else:
                    # Same or worse: accept with probability
                    if random.random() < min(1.0, math.exp(0 / T)):
                        current_adj = new_adj
                        current_edges = new_edges
                        current_count = len(current_edges)
        
        elif move_type < 0.7:
            # Try to remove an edge
            edge_list = list(current_edges)
            if not edge_list:
                continue
            u, v = tuple(random.choice(edge_list))
            
            new_adj = [set(s) for s in current_adj]
            new_adj[u].discard(v)
            new_adj[v].discard(u)
            new_edges = set(current_edges)
            new_edges.discard(frozenset((u, v)))
            
            # Removing an edge always keeps triangle-free
            # Check if still not 3-colorable
            if not is_3_colorable_fast(new_adj):
                # Still 4-chromatic: accept if better or with probability
                if len(new_edges) > current_count:
                    current_adj = new_adj
                    current_edges = new_edges
                    current_count = len(current_edges)
                    if current_count > best_count:
                        best_adj = [set(s) for s in current_adj]
                        best_edges = set(current_edges)
                        best_count = current_count
                else:
                    if random.random() < min(1.0, math.exp(0 / T)):
                        current_adj = new_adj
                        current_edges = new_edges
                        current_count = len(current_edges)
            else:
                # Now 3-colorable: reject (we want to maintain 4-chromatic)
                pass
        
        else:
            # Swap: remove one edge, add another
            edge_list = list(current_edges)
            missing = get_missing_edges(current_edges)
            if not edge_list or not missing:
                continue
            
            u, v = tuple(random.choice(edge_list))
            x, y = random.choice(missing)
            
            # Remove (u,v)
            new_adj = [set(s) for s in current_adj]
            new_adj[u].discard(v)
            new_adj[v].discard(u)
            
            # Add (x,y) if triangle-free
            if not can_add_edge(new_adj, x, y):
                continue
            
            new_adj[x].add(y)
            new_adj[y].add(x)
            
            new_edges = set(current_edges)
            new_edges.discard(frozenset((u, v)))
            new_edges.add(frozenset((x, y)))
            
            # Check 4-chromatic
            if not is_3_colorable_fast(new_adj):
                # Valid: accept if better or with probability
                if len(new_edges) > current_count:
                    current_adj = new_adj
                    current_edges = new_edges
                    current_count = len(current_edges)
                    if current_count > best_count:
                        best_adj = [set(s) for s in current_adj]
                        best_edges = set(current_edges)
                        best_count = current_count
                else:
                    if random.random() < min(1.0, math.exp(0 / T)):
                        current_adj = new_adj
                        current_edges = new_edges
                        current_count = len(current_edges)
            else:
                # 3-colorable: reject
                pass
        
        # Cool down
        T = max(T * cooling, T_min)
    
    # Final greedy improvement on best solution
    for _ in range(100):
        if time.time() > deadline:
            break
        improved = False
        missing = get_missing_edges(best_edges)
        for u, v in missing:
            if not can_add_edge(best_adj, u, v):
                continue
            new_adj = [set(s) for s in best_adj]
            new_adj[u].add(v)
            new_adj[v].add(u)
            if not is_3_colorable_fast(new_adj):
                best_adj = new_adj
                best_edges.add(frozenset((u, v)))
                best_count += 1
                improved = True
                break
        if not improved:
            break
    
    # Convert best_adj to edge list
    result = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                result.append([u, v])
    
    return result
