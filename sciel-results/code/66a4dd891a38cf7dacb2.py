import random
import time
import numpy as np

def construct():
    n = 20
    # Use numpy adjacency matrix for fast vectorized common-neighbor computations
    adj = np.zeros((n, n), dtype=np.int8)
    
    def get_edge_count():
        return int(np.sum(adj)) // 2
    
    def common_neighbors(u, v):
        # Number of common neighbors of u and v
        return int(np.dot(adj[u], adj[v]))
    
    def can_add(u, v):
        if u == v:
            return False
        if adj[u, v] == 1:
            return False
        
        # Check 1: common neighbors of u and v must be <= 2
        if np.dot(adj[u], adj[v]) > 2:
            return False
        
        # Check 2: For all y in N(v)\{u}, common(u,y) must be <= 1
        common_u = adj @ adj[u]
        for y in np.nonzero(adj[v])[0]:
            if y == u:
                continue
            if common_u[y] > 1:
                return False
        
        # Check 3: For all y in N(u)\{v}, common(v,y) must be <= 1
        common_v = adj @ adj[v]
        for y in np.nonzero(adj[u])[0]:
            if y == v:
                continue
            if common_v[y] > 1:
                return False
        
        return True
    
    def add_edge(u, v):
        adj[u, v] = 1
        adj[v, u] = 1
    
    def remove_edge(u, v):
        adj[u, v] = 0
        adj[v, u] = 0
    
    def is_valid():
        """Check if the current graph is K2,3-free."""
        for u in range(n):
            for v in range(u + 1, n):
                if np.dot(adj[u], adj[v]) > 2:
                    return False
        return True
    
    def build_clebsch():
        # Clebsch graph on vertices 0-15
        # Vertices are 4-bit strings.
        # Adjacency: Hamming distance 1 or 4.
        for u in range(16):
            for v in range(16):
                if u >= v:
                    continue
                diff = u ^ v
                popcount = bin(diff).count('1')
                if popcount == 1 or popcount == 4:
                    add_edge(u, v)
    
    def build_initial_graph():
        # Build Clebsch graph
        build_clebsch()
        
        # Find all edges in Clebsch graph
        edges_in_clebsch = []
        for u in range(16):
            for v in range(u + 1, 16):
                if adj[u, v] == 1:
                    edges_in_clebsch.append((u, v))
        
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
    
    def systematic_polish():
        """Try all possible edge additions systematically."""
        improved = False
        for u in range(n):
            for v in range(u + 1, n):
                if can_add(u, v):
                    add_edge(u, v)
                    improved = True
        return improved
    
    def swap_search(max_swaps=50):
        """Try to find beneficial edge swaps (remove one, add one or two)."""
        edges_list = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v] == 1]
        if not edges_list:
            return False
        
        current_count = get_edge_count()
        
        for _ in range(max_swaps):
            # Pick a random edge to remove
            u_rem, v_rem = random.choice(edges_list)
            remove_edge(u_rem, v_rem)
            
            # Try to add edges
            added = []
            for u_add in range(n):
                for v_add in range(u_add + 1, n):
                    if can_add(u_add, v_add):
                        add_edge(u_add, v_add)
                        added.append((u_add, v_add))
                        if get_edge_count() > current_count:
                            # Beneficial swap
                            return True
            
            # Not beneficial, revert
            for u_add, v_add in added:
                remove_edge(u_add, v_add)
            add_edge(u_rem, v_rem)
        
        return False
    
    def get_edges_list():
        return [[u, v] for u in range(n) for v in range(u + 1, n) if adj[u, v] == 1]
    
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # Build initial structure
    build_initial_graph()
    
    best_adj = adj.copy()
    best_count = get_edge_count()
    
    stagnant_iterations = 0
    max_stagnant = 80
    
    while time.time() - start_time < time_limit:
        # Check time more frequently
        if time.time() - start_time > time_limit - 5:
            break
        
        improved = False
        
        # Phase 1: Try random edge additions
        for _ in range(300):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if can_add(u, v):
                add_edge(u, v)
                improved = True
                break
        
        if not improved:
            # Phase 2: Try systematic additions
            if systematic_polish():
                improved = True
            else:
                # Phase 3: Try swaps
                if swap_search(30):
                    improved = True
        
        curr_count = get_edge_count()
        
        if curr_count > best_count:
            best_count = curr_count
            best_adj = adj.copy()
            stagnant_iterations = 0
        else:
            stagnant_iterations += 1
        
        if stagnant_iterations >= max_stagnant:
            # Restart with new random seed
            stagnant_iterations = 0
            # Keep best and try to improve from scratch
            adj[:] = best_adj
    
    # Final polish pass
    adj[:] = best_adj
    systematic_polish()
    
    # Verify and return
    best_edges = get_edges_list()
    
    # Verification step
    adj_check = np.zeros((n, n), dtype=np.int8)
    for u, v in best_edges:
        adj_check[u, v] = 1
        adj_check[v, u] = 1
    
    valid = True
    for u in range(n):
        for v in range(u + 1, n):
            if np.dot(adj_check[u], adj_check[v]) > 2:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: return the best_adj version which should be valid
        best_edges = get_edges_list()
        # Re-verify
        adj_check2 = np.zeros((n, n), dtype=np.int8)
        for u, v in best_edges:
            adj_check2[u, v] = 1
            adj_check2[v, u] = 1
        
        valid2 = True
        for u in range(n):
            for v in range(u + 1, n):
                if np.dot(adj_check2[u], adj_check2[v]) > 2:
                    valid2 = False
                    break
            if not valid2:
                break
        
        if not valid2:
            # Last resort fallback
            edges = []
            for k in range(5):
                edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
            return edges
    
    return best_edges
