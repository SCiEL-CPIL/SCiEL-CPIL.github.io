"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Construct a K2,3-free graph on 20 vertices maximizing edges.
    Uses simulated annealing with local search polish.
    """
    n = 20
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Initialize with a dense random graph, then fix violations
    # Or start from a structured graph and optimize
    
    def random_graph():
        """Generate a random graph and remove K2,3 violations."""
        # Start with all edges, then remove to make K2,3-free
        # Better: start with moderate density and add edges greedily
        adj = np.zeros((n, n), dtype=np.uint8)
        # Random initial edges
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.4:
                    adj[i, j] = adj[j, i] = 1
        return adj
    
    def count_edges(adj):
        return int(adj.sum() // 2)
    
    def is_k23_free(adj):
        """Check if graph has no K2,3."""
        # Common neighbors of u and v = sum of adj[u,:] * adj[v,:]
        # For each pair, if common neighbors >= 3, violation
        common = adj @ adj  # common[u,v] = number of common neighbors
        for u in range(n):
            for v in range(u+1, n):
                if common[u, v] >= 3:
                    return False
        return True
    
    def common_neighbors(adj, u, v):
        """Count common neighbors of u and v."""
        return int(np.sum(adj[u, :] * adj[v, :]))
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        if adj[u, v]:
            return True
        # Adding edge (u,v) creates K2,3 if there exists a vertex w 
        # such that w is a common neighbor of u and v, AND there exist 
        # two other vertices w1, w2 that are common neighbors of u and v.
        # Actually, K2,3 means two vertices with 3 common neighbors.
        # When we add edge (u,v), we need to check if any pair now has >= 3 common neighbors.
        # The pairs that could be affected are:
        # 1. (u, v) itself: they now share all their common neighbors
        # 2. (u, w) for each neighbor w of v: u and w now share v as a common neighbor
        # 3. (v, w) for each neighbor w of u: v and w now share u as a common neighbor
        
        # Check (u,v): count common neighbors
        cn_uv = int(np.sum(adj[u, :] * adj[v, :]))
        if cn_uv >= 3:
            return False
        
        # Check (u, w) for w in neighbors of v
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v, w]:
                # u and w: common neighbors include v (since u-v and v-w)
                # Plus other common neighbors
                cn_uw = int(np.sum(adj[u, :] * adj[w, :]))
                # v is a common neighbor of u and w iff adj[u,v] and adj[v,w]
                # But adj[u,v] is 0 currently, so v is NOT currently a common neighbor
                # After adding (u,v), v becomes a common neighbor of u and w
                # So new common count = cn_uw + 1
                if cn_uw + 1 >= 3:
                    return False
        
        # Check (v, w) for w in neighbors of u
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[u, w]:
                cn_vw = int(np.sum(adj[v, :] * adj[w, :]))
                if cn_vw + 1 >= 3:
                    return False
        
        return True
    
    def local_search(adj):
        """Try to add edges greedily until no more can be added."""
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if not adj[u, v] and can_add_edge(adj, u, v):
                        adj[u, v] = adj[v, u] = 1
                        improved = True
        return adj
    
    def simulated_annealing(adj, start_temp, end_temp, steps_per_temp, temps):
        """Run simulated annealing."""
        current = adj.copy()
        current_edges = count_edges(current)
        best = current.copy()
        best_edges = current_edges
        
        for temp in temps:
            for _ in range(steps_per_temp):
                if time.time() > deadline:
                    break
                # Random move: add or remove a random edge
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                
                if current[u, v]:
                    # Remove edge
                    new_adj = current.copy()
                    new_adj[u, v] = new_adj[v, u] = 0
                    delta = -1
                else:
                    # Try to add edge
                    if not can_add_edge(current, u, v):
                        continue
                    new_adj = current.copy()
                    new_adj[u, v] = new_adj[v, u] = 1
                    delta = 1
                
                new_edges = current_edges + delta
                
                # Accept if better or with probability exp(delta/temp)
                if delta > 0 or random.random() < np.exp(delta / max(temp, 1e-10)):
                    current = new_adj
                    current_edges = new_edges
                    if current_edges > best_edges:
                        best = current.copy()
                        best_edges = current_edges
            
            if time.time() > deadline:
                break
        
        return best, best_edges
    
    # Start with multiple random initializations
    best_overall = None
    best_overall_edges = 0
    
    # Try a few different strategies
    strategies = 0
    
    # Strategy 1: Random initialization + local search + SA
    for attempt in range(5):
        if time.time() > deadline:
            break
        adj = random_graph()
        adj = local_search(adj)
        edges = count_edges(adj)
        
        if edges > best_overall_edges:
            best_overall = adj.copy()
            best_overall_edges = edges
        
        # SA polish
        temps = np.linspace(1.0, 0.01, 20)
        steps_per_temp = 500
        adj, edges = simulated_annealing(adj, 1.0, 0.01, steps_per_temp, temps)
        adj = local_search(adj)
        edges = count_edges(adj)
        
        if edges > best_overall_edges:
            best_overall = adj.copy()
            best_overall_edges = edges
    
    # Strategy 2: Start from a structured graph (e.g., Paley-like or random regular)
    # Try starting from a higher density random graph
    for attempt in range(3):
        if time.time() > deadline:
            break
        adj = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.6:
                    adj[i, j] = adj[j, i] = 1
        # Fix violations by removing edges
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u, v] and common_neighbors(adj, u, v) >= 3:
                        # Remove one of the edges causing the violation
                        # Remove edge (u,v)
                        adj[u, v] = adj[v, u] = 0
                        changed = True
        adj = local_search(adj)
        edges = count_edges(adj)
        
        if edges > best_overall_edges:
            best_overall = adj.copy()
            best_overall_edges = edges
        
        # SA polish
        if time.time() < deadline:
            temps = np.linspace(1.0, 0.01, 15)
            steps_per_temp = 300
            adj, edges = simulated_annealing(adj, 1.0, 0.01, steps_per_temp, temps)
            adj = local_search(adj)
            edges = count_edges(adj)
            if edges > best_overall_edges:
                best_overall = adj.copy()
                best_overall_edges = edges
    
    # Final local search
    if best_overall is not None:
        best_overall = local_search(best_overall)
        best_overall_edges = count_edges(best_overall)
    
    # Convert to edge list
    if best_overall is None:
        # Fallback: return 5 disjoint K4s
        edges = []
        for k in range(5):
            edges += [[4*k+a, 4*k+b] for a in range(4) for b in range(a+1, 4)]
        return edges
    
    edge_list = []
    for u in range(n):
        for v in range(u+1, n):
            if best_overall[u, v]:
                edge_list.append([u, v])
    
    return edge_list

# EVOLVE-BLOCK-END
