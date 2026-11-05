"""Optimized greedy for rainbow 4-cycle edge coloring on 20 vertices."""
import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    n = 20
    colors = list(range(n))
    
    best_edges = None
    best_count = 0
    
    # Try multiple random restarts with different edge orderings
    random.seed(42)
    
    max_restarts = 500
    
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        # Random permutation of vertices to vary the order
        vertex_order = list(range(n))
        random.shuffle(vertex_order)
        
        # Build candidate edge list in a structured order
        # Use a "fill" strategy: try to build dense subgraphs
        candidate_edges = []
        for i in range(n):
            for j in range(i+1, n):
                candidate_edges.append((i, j))
        
        # Shuffle with some structure: group by proximity in random order
        random.shuffle(candidate_edges)
        
        # Greedy construction
        adj = {}  # (u,v) -> color, u < v
        edge_list = []
        
        # Track adjacency for fast 4-cycle checking
        neighbors = {v: set() for v in range(n)}
        edge_color = {}  # frozenset({u,v}) -> color
        
        for (u, v) in candidate_edges:
            if time.time() - start_time > DEADLINE:
                break
            
            # Find common neighbors of u and v
            common = neighbors[u] & neighbors[v]
            if len(common) < 2:
                # No 4-cycle can be formed, use any color
                c = random.choice(colors)
                key = frozenset((u, v))
                adj[key] = c
                edge_color[key] = c
                edge_list.append([u, v, c])
                neighbors[u].add(v)
                neighbors[v].add(u)
                continue
            
            # Find forbidden colors
            forbidden = set()
            common_list = list(common)
            for idx_x in range(len(common_list)):
                for idx_y in range(idx_x + 1, len(common_list)):
                    x = common_list[idx_x]
                    y = common_list[idx_y]
                    # 4-cycle: u-x-v-y-u
                    # Edges: (u,x), (x,v), (v,y), (y,u)
                    for a, b in [(u, x), (x, v), (v, y), (y, u)]:
                        key2 = frozenset((a, b))
                        if key2 in edge_color:
                            forbidden.add(edge_color[key2])
            
            # Pick a color not in forbidden
            available = [c for c in colors if c not in forbidden]
            if not available:
                continue  # Skip this edge
            
            c = random.choice(available)
            key = frozenset((u, v))
            adj[key] = c
            edge_color[key] = c
            edge_list.append([u, v, c])
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        count = len(edge_list)
        if count > best_count:
            best_count = count
            best_edges = [list(e) for e in edge_list]
        
        # Early termination if we found a very good solution
        if best_count >= 70:
            break
    
    if best_edges is None:
        # Fallback: star
        return [[0, v, 0] for v in range(1, n)]
    
    return best_edges
