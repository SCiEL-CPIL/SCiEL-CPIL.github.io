import random
import time

def construct():
    n = 20
    # Use sets for neighbors and a 2D list for common neighbor counts.
    # This is faster than numpy for the small n=20 case due to lower overhead
    # in Python loops and set operations for the specific access patterns
    # required by the K2,3 constraint checks.
    
    adj = [set() for _ in range(n)]
    common = [[0] * n for _ in range(n)]
    
    def can_add(u, v):
        if u == v:
            return False
        if v in adj[u]:
            return False
        # Check 1: common neighbors of u and v must be <= 2
        if common[u][v] > 2:
            return False
        
        # Check 2: For all y in N(v), common[u][y] must be <= 1
        # Adding edge (u,v) increases common(u,y) by 1 for all y in N(v) \ {u}
        # because v becomes a common neighbor of u and y.
        # We need common(u,y) + 1 <= 2 => common(u,y) <= 1.
        for y in adj[v]:
            if y == u:
                continue
            if common[u][y] > 1:
                return False
        
        # Check 3: For all y in N(u), common[v][y] must be <= 1
        for y in adj[u]:
            if y == v:
                continue
            if common[v][y] > 1:
                return False
        
        return True

    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
        # Update common counts
        # For y in N(v) (excluding u): common[u][y] += 1
        for y in adj[v]:
            if y == u:
                continue
            common[u][y] += 1
            common[y][u] += 1
            
        # For y in N(u) (excluding v): common[v][y] += 1
        for y in adj[u]:
            if y == v:
                continue
            common[v][y] += 1
            common[y][v] += 1

    def remove_edge(u, v):
        adj[u].remove(v)
        adj[v].remove(u)
        # Revert updates
        # Note: adj[v] no longer contains u, so we iterate over current adj[v]
        for y in adj[v]:
            common[u][y] -= 1
            common[y][u] -= 1
        # Note: adj[u] no longer contains v, so we iterate over current adj[u]
        for y in adj[u]:
            common[v][y] -= 1
            common[y][v] -= 1

    def get_edge_count():
        return sum(len(s) for s in adj) // 2

    def get_edges_list():
        return [[u, v] for u in range(n) for v in adj[u] if u < v]

    def verify(adj_list, common_mat):
        """
        Verify if the graph defined by adj_list and common_mat is K2,3-free.
        Returns True if valid.
        """
        # Reconstruct common matrix from adj_list to be safe?
        # No, we trust the internal state if it's consistent.
        # But for a standalone check of a candidate, we should recompute or check consistency.
        # Here we assume adj_list and common_mat are consistent.
        
        for u in range(n):
            for v in range(u + 1, n):
                if common_mat[u][v] > 2:
                    return False
        return True

    def build_clebsch():
        # Clebsch graph on vertices 0-15
        for u in range(16):
            for v in range(16):
                if u >= v:
                    continue
                diff = u ^ v
                popcount = bin(diff).count('1')
                if popcount == 1 or popcount == 4:
                    add_edge(u, v)

    def build_initial_graph():
        build_clebsch()
        
        # Find all edges in Clebsch graph
        edges_in_clebsch = [(u, v) for u in range(16) for v in adj[u] if u < v]
        
        # Greedily pick 4 disjoint edges
        used_vertices = set()
        selected_pairs = []
        for u, v in edges_in_clebsch:
            if u not in used_vertices and v not in used_vertices:
                selected_pairs.append((u, v))
                used_vertices.add(u)
                used_vertices.add(v)
                if len(selected_pairs) == 4:
                    break
            
        # Connect new vertices 16-19 to these pairs
        for i, (u, v) in enumerate(selected_pairs):
            new_v = 16 + i
            add_edge(new_v, u)
            add_edge(new_v, v)
            
        # Connect new vertices among themselves (K4)
        for i in range(4):
            for j in range(i + 1, 4):
                u = 16 + i
                v = 16 + j
                if can_add(u, v):
                    add_edge(u, v)

    start_time = time.time()
    time_limit = 115.0 # Safety margin for 120s limit
    
    # Build initial structure
    build_initial_graph()
    
    best_edges = get_edges_list()
    best_count = get_edge_count()
    
    # Local search with simulated annealing-like swaps
    # We will try to improve the graph by:
    # 1. Adding edges if possible.
    # 2. Swapping edges: remove one, add another (or two) to escape local optima.
    
    while time.time() - start_time < time_limit:
        # Phase 1: Try to add random edges
        improved = False
        # Try many random additions
        for _ in range(200):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if can_add(u, v):
                add_edge(u, v)
                improved = True
                break
        
        if not improved:
            # Phase 2: Try a swap move
            # Remove a random edge, then try to add 1 or 2 edges to compensate/gain
            curr_edges = get_edges_list()
            if curr_edges:
                u_rem, v_rem = random.choice(curr_edges)
                remove_edge(u_rem, v_rem)
                
                # Try to add edges back and maybe more
                added_count = 0
                # Try to add the same edge back? No, that's useless.
                # Try to add other edges.
                for _ in range(100):
                    u_add = random.randint(0, n - 1)
                    v_add = random.randint(0, n - 1)
                    if can_add(u_add, v_add):
                        add_edge(u_add, v_add)
                        added_count += 1
                        # Stop after adding a few to keep the move local
                        if added_count >= 2:
                            break
                
                # If we didn't add enough to compensate, revert?
                # Or just keep if edge count didn't drop?
                # Simulated Annealing style: accept if better or same, occasionally worse.
                # Here, simple hill climbing with random restarts/swaps.
                # If we lost an edge and didn't gain, revert to best?
                # Actually, let's just track the best found so far.
                
                curr_count = get_edge_count()
                if curr_count > best_count:
                    best_count = curr_count
                    best_edges = get_edges_list()
                elif curr_count == best_count:
                    # If same, we might keep the new structure for diversity
                    # but for now, let's just continue.
                    pass
                else:
                    # If we got worse, revert to the state before the swap?
                    # This requires saving state. 
                    # Alternatively, just let it be, and rely on the "add edge" phase to recover.
                    # But if we get stuck in a bad local minimum, we need restarts.
                    pass

    # Final verification of best_edges
    # Reconstruct adjacency and common counts for best_edges to ensure validity
    adj_check = [set() for _ in range(n)]
    for u, v in best_edges:
        adj_check[u].add(v)
        adj_check[v].add(u)
    
    common_check = [[0] * n for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            # common[u][v] = |N(u) ∩ N(v)|
            # Efficiently compute:
            # for y in N(u): if y in N(v): common[u][v] += 1
            pass # This is O(n^3) but n=20 so 8000 ops, fine.
    
    # Faster check:
    valid = True
    for u in range(n):
        for v in range(u + 1, n):
            # Count common neighbors
            cnt = 0
            for y in adj_check[u]:
                if y in adj_check[v]:
                    cnt += 1
                    if cnt > 2:
                        break
            if cnt > 2:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to a known valid graph (e.g., the initial one or seed)
        # Let's return the seed structure
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    return best_edges
