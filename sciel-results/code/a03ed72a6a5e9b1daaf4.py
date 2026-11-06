"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n = 16
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def is_triangle_free(adj):
        """Check if adjacency list represents a triangle-free graph."""
        for u in range(n):
            for v in adj[u]:
                for w in adj[u]:
                    if w != v and w in adj[v]:
                        return False
        return True
    
    def is_3_colorable(adj):
        """Check if graph is 3-colorable using backtracking with bitmask color tracking."""
        # Precompute neighbor bitmasks for faster forbidden color computation
        neighbor_mask = [0] * n
        for u in range(n):
            for v in adj[u]:
                neighbor_mask[u] |= (1 << v)
        
        color = [-1] * n  # -1 = uncolored, 0/1/2 = color
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Compute forbidden colors using bitmask
            # For each neighbor, if colored, set the corresponding bit
            forbidden = 0
            vmask = neighbor_mask[v]
            for c in range(3):
                # Check if any neighbor has color c
                # We need to check: for each neighbor u of v, if color[u] == c
                # Instead, precompute: for each color c, which vertices have color c
                # Simpler: iterate neighbors
                pass
            # Actually, let's use a different approach: maintain color_mask[c] = bitmask of vertices with color c
            # But that requires global state. Let's just iterate neighbors but use bit ops
            forbidden = 0
            for u in adj[v]:
                cu = color[u]
                if cu != -1:
                    forbidden |= (1 << cu)
            # Try each available color
            for c in range(3):
                if not (forbidden & (1 << c)):
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)
    
    def edge_count(adj):
        return sum(len(adj[i]) for i in range(n)) // 2
    
    def build_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    # Start with a known 4-chromatic triangle-free graph on 11 vertices (Grötzsch)
    # and add vertices 11-15 as isolated, then try to add edges
    def groetzsch_graph():
        # Grötzsch graph: Mycielski of C5
        edges = []
        # Cycle 0-1-2-3-4-0
        for i in range(5):
            edges.append((i, (i+1)%5))
        # Add vertex 5 connected to 1,3 (neighbors of 0)
        # Actually Mycielski: for each edge (u,v) in C5, add edges from new vertex u' to v and v' to u
        # Let me use the standard construction
        # Vertices 0-4: cycle
        # Vertices 5-9: copies, vertex 5+i connects to (i+1)%5 and (i-1)%5... 
        # Actually let me just hardcode the Grötzsch graph
        # Standard Grötzsch: 
        # 0-1, 1-2, 2-3, 3-4, 4-0 (cycle)
        # 5-2, 5-4 (vertex 5 connects to 2 and 4, which are neighbors of 0... no)
        # Mycielski of C5: 
        # Original: 0-1-2-3-4-0
        # New vertices 5-9: vertex 5+i connects to the two neighbors of vertex i in C5
        # So 5 connects to 1 and 4 (neighbors of 0)
        # 6 connects to 2 and 0 (neighbors of 1)
        # 7 connects to 3 and 1 (neighbors of 2)
        # 8 connects to 4 and 2 (neighbors of 3)
        # 9 connects to 0 and 3 (neighbors of 4)
        # New vertex 10 connects to all of 5-9
        edges = [(0,1),(1,2),(2,3),(3,4),(4,0)]
        edges += [(5,1),(5,4),(6,2),(6,0),(7,3),(7,1),(8,4),(8,2),(9,0),(9,3)]
        edges += [(10,5),(10,6),(10,7),(10,8),(10,9)]
        return edges
    
    def add_edges_greedy(adj, deadline):
        """Try to add edges greedily, maintaining triangle-free and non-3-colorable."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all non-edges in random order
            non_edges = []
            for u, v in all_pairs:
                if v not in adj[u]:
                    non_edges.append((u, v))
            random.shuffle(non_edges)
            for u, v in non_edges:
                if time.time() >= deadline:
                    break
                # Check triangle-free: no common neighbor
                common = adj[u] & adj[v]
                if common:
                    continue
                # Try adding edge
                adj[u].add(v)
                adj[v].add(u)
                # Check 3-colorability
                if not is_3_colorable(adj):
                    improved = True
                else:
                    adj[u].remove(v)
                    adj[v].remove(u)
        return adj
    
    def local_search_swap(adj, deadline):
        """Try 1-1 swaps: remove one edge, add another, maintaining constraints."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        current_edges.append((u, v))
            random.shuffle(current_edges)
            
            # Try all possible swaps
            for u, v in current_edges:
                if time.time() >= deadline:
                    return adj, False
                # Remove edge (u,v)
                adj[u].remove(v)
                adj[v].remove(u)
                
                # Try adding a new edge
                found_swap = False
                for a, b in all_pairs:
                    if time.time() >= deadline:
                        break
                    if (a, b) == (u, v) or (a, b) == (v, u):
                        continue
                    if b in adj[a]:
                        continue
                    # Check triangle-free
                    common = adj[a] & adj[b]
                    if common:
                        continue
                    adj[a].add(b)
                    adj[b].add(a)
                    if not is_3_colorable(adj):
                        # Accept the swap
                        improved = True
                        found_swap = True
                        break
                    adj[a].remove(b)
                    adj[b].remove(a)
                
                if found_swap:
                    return adj, True
                
                # Restore edge (u,v)
                adj[u].add(v)
                adj[v].add(u)
        return adj, False
    
    # Start with Grötzsch graph
    initial_edges = groetzsch_graph()
    best_adj = build_adj(initial_edges)
    best_count = edge_count(best_adj)
    
    # First, greedily add edges
    best_adj = add_edges_greedy(best_adj, deadline)
    best_count = edge_count(best_adj)
    
    # Then do local search with multiple restarts
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() >= deadline:
            break
        
        # Start from best known
        adj = [set(s) for s in best_adj]
        
        # Do local search
        adj, improved = local_search_swap(adj, deadline)
        if improved:
            count = edge_count(adj)
            if count > best_count:
                best_adj = adj
                best_count = count
        
        # Also try from a perturbed version
        if time.time() >= deadline:
            break
        adj = [set(s) for s in best_adj]
        # Remove 1-2 random edges to perturb more aggressively
        edges_list = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges_list.append((u, v))
        if edges_list:
            num_remove = min(2, len(edges_list))
            for _ in range(num_remove):
                rem_u, rem_v = random.choice(edges_list)
                adj[rem_u].remove(rem_v)
                adj[rem_v].remove(rem_u)
                edges_list.remove((rem_u, rem_v))
        
        adj = add_edges_greedy(adj, deadline)
        adj, improved = local_search_swap(adj, deadline)
        count = edge_count(adj)
        if count > best_count:
            best_adj = adj
            best_count = count
    
    # Final polish: try to add more edges with remaining time
    if time.time() < deadline - 10:
        best_adj = add_edges_greedy(best_adj, deadline - 5)
        best_count = edge_count(best_adj)
    
    # Convert best_adj to edge list
    result_edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                result_edges.append([u, v])
    
    return result_edges
# EVOLVE-BLOCK-END
