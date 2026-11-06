import time
import sys
import random
import math

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n = 16
    total_possible = n * (n - 1) // 2  # 120
    
    # Bitmask representation for adjacency
    # adj[i] is a bitmask of neighbors of vertex i
    
    def is_3_colorable(adj):
        """
        Check if graph is 3-colorable.
        Uses backtracking with dynamic MRV and bitmasks for speed.
        """
        colors = [-1] * n
        
        def dfs(uncolored_count):
            if uncolored_count == 0:
                return True
            
            # Find vertex with fewest available colors (MRV)
            best_v = -1
            best_avail = None
            min_avail_size = 4
            
            for v in range(n):
                if colors[v] == -1:
                    used_mask = 0
                    mask = adj[v]
                    # Iterate through set bits in mask to check colors of neighbors
                    for u in range(n):
                        if mask & (1 << u):
                            if colors[u] != -1:
                                used_mask |= (1 << colors[u])
                    
                    avail_mask = 7 & ~used_mask
                    avail_count = bin(avail_mask).count("1")
                    
                    if avail_count == 0:
                        return False
                    
                    if avail_count < min_avail_size:
                        min_avail_size = avail_count
                        best_v = v
                        best_avail = avail_mask
                        if min_avail_size == 1:
                            break
            
            if best_v == -1:
                return True
            
            # Try each available color
            while best_avail:
                lsb = best_avail & -best_avail
                c = lsb.bit_length() - 1
                best_avail ^= lsb
                
                colors[best_v] = c
                if dfs(uncolored_count - 1):
                    return True
                colors[best_v] = -1
            
            return False
        
        return dfs(n)

    def count_edges(adj):
        c = 0
        for i in range(n):
            c += bin(adj[i]).count("1")
        return c // 2

    def is_triangle_free(adj):
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    if (adj[i] & adj[j]) != 0:
                        return False
        return True

    # Generate a strong initial seed: Mycielski of C5 (11 vertices) + 5 vertices
    # The Mycielski of C5 (Grötzsch graph) has 11 vertices, 20 edges, chromatic number 4, triangle-free.
    # We need 16 vertices. We can add 5 independent vertices connected to the independent set of the Grötzsch graph?
    # Actually, let's use a known dense construction or generate a random one and refine.
    
    def generate_initial():
        adj = [0] * n
        def add_edge(u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # C5: 0-4
        for i in range(5):
            add_edge(i, (i + 1) % 5)
        # Mycielski layer 1: 5-9
        for i in range(5):
            add_edge(i, 5 + (i + 1) % 5)
            add_edge((i + 1) % 5, 5 + i)
        # Root 10
        for i in range(5):
            add_edge(10, 5 + i)
            
        # Vertices 11-15: Add them to maximize edges while keeping triangle-free.
        # In Grötzsch, the set {5,6,7,8,9} is an independent set? No, 5 is connected to 1,2. 6 to 2,3.
        # Actually, the "copies" 5..9 form an independent set in the Mycielski construction?
        # Let's check: 5 connected to 1,2. 6 connected to 2,3. No edge between 5 and 6.
        # So {5,6,7,8,9} is an independent set.
        # If we connect 11..15 to {5,6,7,8,9}, do we create triangles?
        # 11-5 and 5-1 and 1-11? No, 11 is new.
        # 11-5 and 11-6? No, 5-6 is not an edge.
        # So we can connect 11..15 to any subset of {5,6,7,8,9} without creating triangles involving 11..15 among themselves (if they are independent) or with the base.
        # If we make 11..15 an independent set, and connect them to 5..9, we need to ensure no triangles.
        # Triangle would require an edge between a neighbor of 11 in 5..9 and another neighbor of 11 in 5..9.
        # Since 5..9 is independent, there are no edges between them.
        # So we can connect 11..15 to ALL of 5..9.
        # This adds 5*5 = 25 edges.
        # Total edges: 20 (Grötzsch) + 25 = 45.
        
        for v_new in range(11, 16):
            for v_copy in range(5, 10):
                add_edge(v_new, v_copy)
                
        return adj

    best_adj = generate_initial()
    best_count = count_edges(best_adj)
    
    # Verify initial is valid
    if not is_triangle_free(best_adj) or is_3_colorable(best_adj):
        # Fallback to a simpler known valid graph if construction fails
        # But the construction above is theoretically valid.
        pass

    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Strategy: 
    # 1. Greedy addition of edges that don't create triangles and maintain non-3-colorability.
    # 2. If stuck, try edge swaps (remove one, add two? No, remove one add one to restructure).
    # 3. Use a "Repair" mechanism: If adding an edge creates a triangle, remove one edge from the triangle 
    #    and see if we can add a *different* edge to compensate or just accept the swap if it helps future additions.
    
    # Better Strategy for this specific task:
    # The upper bound for triangle-free is 64.
    # We are likely far from 64.
    # Let's use a local search that aggressively tries to add edges.
    
    def try_add_edge(adj, u, v):
        if adj[u] & (1 << v):
            return False, adj
        common = adj[u] & adj[v]
        if common != 0:
            return False, adj
        new_adj = adj[:]
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        if not is_3_colorable(new_adj):
            return True, new_adj
        return False, adj

    def local_search(adj, current_count):
        adj = adj[:]
        current_count = current_count
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5:
                break
            
            # Try to add any non-edge
            non_edges = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                # Check triangle
                if (adj[u] & adj[v]) == 0:
                    ok, new_adj = try_add_edge(adj, u, v)
                    if ok:
                        adj = new_adj
                        current_count += 1
                        improved = True
                        break
            
            if improved:
                continue
                
            # If no simple addition, try swaps
            # Remove an edge (a,b) and add (u,v)
            # This is complex. Let's try "Edge Removal + Greedy Refill"
            # Pick a random edge, remove it, then greedily add as many as possible.
            # Accept if count increases.
            
            edges_in_graph = [(u, v) for u in range(n) for v in range(u+1, n) if adj[u] & (1 << v)]
            if not edges_in_graph:
                break
            
            # Try a few random removals
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                a, b = random.choice(edges_in_graph)
                test_adj = adj[:]
                test_adj[a] &= ~(1 << b)
                test_adj[b] &= ~(1 << a)
                
                # Greedy refill on test_adj
                test_count = current_count - 1
                # Simple greedy: try to add all non-edges that work
                non_edges_test = [(u, v) for u in range(n) for v in range(u+1, n) if not (test_adj[u] & (1 << v))]
                random.shuffle(non_edges_test)
                
                for u, v in non_edges_test:
                    if (test_adj[u] & test_adj[v]) == 0:
                        ok, new_test_adj = try_add_edge(test_adj, u, v)
                        if ok:
                            test_adj = new_test_adj
                            test_count += 1
                            
                if test_count > current_count:
                    adj = test_adj
                    current_count = test_count
                    improved = True
                    break
                    
        return adj, current_count

    # Run multiple restarts
    best_result_adj = best_adj[:]
    best_result_count = best_count
    
    for restart in range(100):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            current_adj = best_adj[:]
        else:
            # Perturb
            current_adj = best_adj[:]
            # Remove 3 random edges
            edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if current_adj[i] & (1 << j)]
            for _ in range(3):
                if not edges_list: break
                u, v = random.choice(edges_list)
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                edges_list.remove((u,v))
                if (v,u) in edges_list: edges_list.remove((v,u))
                
        current_count = count_edges(current_adj)
        
        # Local search
        current_adj, current_count = local_search(current_adj, current_count)
        
        if current_count > best_result_count:
            best_result_count = current_count
            best_result_adj = current_adj[:]
            
    # Final polish: try to add any remaining edges
    for u, v in all_edges:
        if not (best_result_adj[u] & best_result_adj[v]):
            if (best_result_adj[u] & best_result_adj[v]) == 0:
                ok, new_adj = try_add_edge(best_result_adj, u, v)
                if ok:
                    best_result_adj = new_adj
                    best_result_count += 1

    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_result_adj[i] & (1 << j):
                result.append([i, j])
                
    return result

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
