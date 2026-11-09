import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 20
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(all_edges)  # 190
    
    # Precompute for each edge (u,v), which other edges share endpoints
    # This helps with fast validity checking
    edge_index = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
    
    # For each edge, store the list of edges that would be affected if we toggle it
    # Actually, for K3,3-free check, we need to check triples of vertices
    
    def count_violations(M):
        """Count number of K3,3 violations in the graph represented by M (list of bitmasks)."""
        violations = 0
        # A K3,3 exists if there are three vertices u,v,w with at least 3 common neighbors
        for u in range(n):
            for v in range(u + 1, n):
                for w in range(v + 1, n):
                    common = M[u] & M[v] & M[w]
                    if common.bit_count() >= 3:
                        violations += 1
        return violations
    
    def can_add_edge(M, u, v):
        """Check if adding edge (u,v) keeps the graph K3,3-free."""
        if u == v:
            return False
        if (M[u] >> v) & 1:
            return False
        
        # Adding (u,v) affects triples containing u or v
        # Triples {u, x, y} where x,y are neighbors of v:
        # v becomes a common neighbor of u,x,y
        # Violation if |N(u) & N(x) & N(y)| >= 2 (then adding v makes it 3)
        nv = M[v]
        neigh_v = [i for i in range(n) if (nv >> i) & 1]
        
        for i in range(len(neigh_v)):
            x = neigh_v[i]
            for j in range(i + 1, len(neigh_v)):
                y = neigh_v[j]
                common = M[u] & M[x] & M[y]
                if common.bit_count() >= 2:
                    return False
        
        # Triples {v, x, y} where x,y are neighbors of u:
        nu = M[u]
        neigh_u = [i for i in range(n) if (nu >> i) & 1]
        
        for i in range(len(neigh_u)):
            x = neigh_u[i]
            for j in range(i + 1, len(neigh_u)):
                y = neigh_u[j]
                common = M[v] & M[x] & M[y]
                if common.bit_count() >= 2:
                    return False
        
        return True
    
    def add_edge(M, u, v):
        M[u] |= (1 << v)
        M[v] |= (1 << u)
    
    def remove_edge(M, u, v):
        M[u] &= ~(1 << v)
        M[v] &= ~(1 << u)
    
    def count_edges(M):
        return sum(M[i].bit_count() for i in range(n)) // 2
    
    def get_edge_list(M):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (M[u] >> v) & 1:
                    edges.append((u, v))
        return edges
    
    def get_missing_edges(M):
        missing = []
        for u in range(n):
            for v in range(u + 1, n):
                if not ((M[u] >> v) & 1):
                    missing.append((u, v))
        return missing
    
    def simulated_annealing(M, max_iters, start_temp, end_temp):
        """Simulated annealing to maximize edges while maintaining K3,3-free property."""
        current_M = M[:]
        current_edges = count_edges(current_M)
        current_violations = count_violations(current_M)
        
        best_M = current_M[:]
        best_edges = current_edges
        best_violations = current_violations
        
        temp = start_temp
        cooling = (end_temp / start_temp) ** (1.0 / max_iters) if max_iters > 0 else 1.0
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Choose a random edge to toggle
            u, v = random.choice(all_edges)
            
            old_state = (current_M[u] >> v) & 1
            
            if old_state:
                # Remove edge
                remove_edge(current_M, u, v)
            else:
                # Add edge
                add_edge(current_M, u, v)
            
            new_edges = count_edges(current_M)
            new_violations = count_violations(current_M)
            
            # Energy: we want to maximize edges and minimize violations
            # Use a penalty for violations
            old_energy = -current_edges + 100 * current_violations
            new_energy = -new_edges + 100 * new_violations
            
            delta_e = new_energy - old_energy
            
            if delta_e < 0:
                # Accept
                current_edges = new_edges
                current_violations = new_violations
            else:
                # Accept with probability exp(-delta_e / temp)
                if random.random() < np.exp(-delta_e / max(temp, 1e-10)):
                    current_edges = new_edges
                    current_violations = new_violations
                else:
                    # Revert
                    if old_state:
                        add_edge(current_M, u, v)
                    else:
                        remove_edge(current_M, u, v)
            
            # Update best
            if current_violations == 0 and current_edges > best_edges:
                best_edges = current_edges
                best_M = current_M[:]
            elif current_violations < best_violations:
                best_violations = current_violations
                best_M = current_M[:]
            
            temp *= cooling
        
        return best_M, best_edges
    
    def local_search(M):
        """Local search to add edges greedily."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            missing = get_missing_edges(M)
            random.shuffle(missing)
            for u, v in missing:
                if can_add_edge(M, u, v):
                    add_edge(M, u, v)
                    improved = True
                    break
        return count_edges(M)
    
    best_edges = 0
    best_M = [0] * n
    random.seed(42)
    
    # Run multiple restarts with simulated annealing
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        M = [0] * n
        
        # Random initialization with moderate density
        density = random.uniform(0.3, 0.5)
        for u, v in all_edges:
            if random.random() < density:
                add_edge(M, u, v)
        
        # First, remove violations by removing edges
        violations = count_violations(M)
        while violations > 0 and time.time() < deadline:
            # Find a violating triple and remove an edge
            found = False
            for u in range(n):
                for v in range(u + 1, n):
                    for w in range(v + 1, n):
                        common = M[u] & M[v] & M[w]
                        if common.bit_count() >= 3:
                            # Remove a random edge from one of the vertices to a common neighbor
                            common_list = [i for i in range(n) if (common >> i) & 1]
                            x = random.choice(common_list)
                            # Remove edge (u, x) or (v, x) or (w, x)
                            choice = random.randint(0, 2)
                            if choice == 0 and (M[u] >> x) & 1:
                                remove_edge(M, u, x)
                            elif choice == 1 and (M[v] >> x) & 1:
                                remove_edge(M, v, x)
                            elif (M[w] >> x) & 1:
                                remove_edge(M, w, x)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
            violations = count_violations(M)
        
        # Greedy fill
        local_search(M)
        
        # Simulated annealing
        max_iters = 5000
        start_temp = 10.0
        end_temp = 0.1
        M, edges = simulated_annealing(M, max_iters, start_temp, end_temp)
        
        # Final local search
        M2 = M[:]
        edges2 = local_search(M2)
        
        if edges2 > best_edges:
            best_edges = edges2
            best_M = M2[:]
        
        # Also try the SA result directly
        if edges > best_edges and count_violations(M) == 0:
            best_edges = edges
            best_M = M[:]
    
    # Final refinement with more aggressive local search
    if best_edges > 0 and time.time() < deadline:
        M = best_M[:]
        
        # Try remove-1-add-2 moves
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            edges_list = get_edge_list(M)
            random.shuffle(edges_list)
            for a, b in edges_list:
                if time.time() > deadline - 5:
                    break
                remove_edge(M, a, b)
                
                added = 0
                added_edges = []
                missing = get_missing_edges(M)
                random.shuffle(missing)
                
                for u, v in missing:
                    if added == 2:
                        break
                    if can_add_edge(M, u, v):
                        add_edge(M, u, v)
                        added += 1
                        added_edges.append((u, v))
                
                if added == 2:
                    improved = True
                    break
                else:
                    for u, v in added_edges:
                        remove_edge(M, u, v)
                    add_edge(M, a, b)
        
        current = count_edges(M)
        if current > best_edges and count_violations(M) == 0:
            best_edges = current
            best_M = M[:]
    
    # Verify final result is valid
    if count_violations(best_M) > 0:
        # Fallback: build a valid graph from scratch
        best_M = [0] * n
        for u, v in all_edges:
            if can_add_edge(best_M, u, v):
                add_edge(best_M, u, v)
        best_edges = count_edges(best_M)
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_M[u] >> v) & 1:
                edges.append([u, v])
    return edges
