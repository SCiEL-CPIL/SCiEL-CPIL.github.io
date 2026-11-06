import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin before 150s kill
    
    n = 16
    m_target = 65
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def calc_score(adj, deg):
        """Calculate the maximum degree sum over all triangles."""
        # Convert adjacency to bitmask representation for speed
        masks = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i, j]:
                    mask |= (1 << j)
            masks[i] = mask
        
        max_sum = 0
        # Iterate over all pairs (u, v) that form an edge
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    # Common neighbors of u and v
                    common = masks[u] & masks[v]
                    # Only consider w > v to avoid duplicates (u < v < w)
                    common &= ~((1 << (v+1)) - 1)
                    if common:
                        w = common
                        while w:
                            low = w & (-w)
                            wi = low.bit_length() - 1
                            s = deg[u] + deg[v] + deg[wi]
                            if s > max_sum:
                                max_sum = s
                            w &= w - 1
        return max_sum

    def get_edges_non_edges(adj):
        edges = set()
        non_edges = set()
        for u, v in all_pairs:
            if adj[u, v]:
                edges.add((u, v))
            else:
                non_edges.add((u, v))
        return edges, non_edges

    def steepest_descent(adj, deg, max_iters=20, time_limit=10):
        """
        Perform steepest descent local search.
        Tries all possible edge swaps and picks the one that reduces the score the most.
        """
        current_score = calc_score(adj, deg)
        best_score = current_score
        best_adj = adj.copy()
        best_deg = deg.copy()
        
        edges, non_edges = get_edges_non_edges(adj)
        edges_list = list(edges)
        non_edges_list = list(non_edges)
        
        for _ in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            if not edges_list or not non_edges_list:
                break
                
            best_swap = None
            best_new_score = current_score
            
            # Try all possible swaps
            for e in edges_list:
                u, v = e
                for ne in non_edges_list:
                    x, y = ne
                    # Skip if the swap doesn't change the graph structure in a way that could help?
                    # No, just check all.
                    
                    # Apply swap
                    adj[u, v] = 0
                    adj[v, u] = 0
                    adj[x, y] = 1
                    adj[y, x] = 1
                    
                    deg[u] -= 1
                    deg[v] -= 1
                    deg[x] += 1
                    deg[y] += 1
                    
                    new_score = calc_score(adj, deg)
                    
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_swap = (e, ne)
                    
                    # Revert swap
                    adj[u, v] = 1
                    adj[v, u] = 1
                    adj[x, y] = 0
                    adj[y, x] = 0
                    
                    deg[u] += 1
                    deg[v] += 1
                    deg[x] -= 1
                    deg[y] -= 1
            
            if best_swap is None:
                break
            
            # Apply the best swap
            e, ne = best_swap
            u, v = e
            x, y = ne
            
            adj[u, v] = 0
            adj[v, u] = 0
            adj[x, y] = 1
            adj[y, x] = 1
            
            deg[u] -= 1
            deg[v] -= 1
            deg[x] += 1
            deg[y] += 1
            
            current_score = best_new_score
            if current_score < best_score:
                best_score = current_score
                best_adj = adj.copy()
                best_deg = deg.copy()
            
            # Update sets
            edges.remove(e)
            non_edges.add(e)
            non_edges.remove(ne)
            edges.add(ne)
            
            edges_list = list(edges)
            non_edges_list = list(non_edges)
            
            if current_score <= 23:
                break
                
        return best_adj, best_deg, best_score

    def simulated_annealing(adj, deg, time_limit=10):
        """
        Simulated annealing to escape local minima.
        """
        current_score = calc_score(adj, deg)
        best_score = current_score
        best_adj = adj.copy()
        best_deg = deg.copy()
        
        edges, non_edges = get_edges_non_edges(adj)
        edges_list = list(edges)
        non_edges_list = list(non_edges)
        
        T = 1.0
        cooling = 0.999
        no_improve = 0
        
        while time.time() - start_time < time_limit:
            if not edges_list or not non_edges_list:
                break
                
            u, v = random.choice(edges_list)
            x, y = random.choice(non_edges_list)
            
            adj[u, v] = 0
            adj[v, u] = 0
            adj[x, y] = 1
            adj[y, x] = 1
            
            deg[u] -= 1
            deg[v] -= 1
            deg[x] += 1
            deg[y] += 1
            
            new_score = calc_score(adj, deg)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                no_improve = 0
                
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj.copy()
                    best_deg = deg.copy()
                    
                    if best_score <= 23:
                        break
                else:
                    no_improve += 1
                
                edges.remove((u, v))
                non_edges.add((u, v))
                non_edges.remove((x, y))
                edges.add((x, y))
                
                edges_list = list(edges)
                non_edges_list = list(non_edges)
            else:
                adj[u, v] = 1
                adj[v, u] = 1
                adj[x, y] = 0
                adj[y, x] = 0
                
                deg[u] += 1
                deg[v] += 1
                deg[x] -= 1
                deg[y] -= 1
                no_improve += 1
            
            T *= cooling
            if T < 0.01:
                T = 0.01
            
            if no_improve > 200:
                break
                
        return best_adj, best_deg, best_score

    def generate_initial_structures():
        """Generate diverse initial structures."""
        structures = []
        
        # 1. K_{6,10} + 5 matching edges
        adj = np.zeros((n, n), dtype=int)
        vertices = list(range(n))
        random.shuffle(vertices)
        A = vertices[:6]
        B = vertices[6:]
        for u in A:
            for v in B:
                adj[u, v] = 1
                adj[v, u] = 1
        random.shuffle(B)
        for i in range(5):
            u, v = B[2*i], B[2*i+1]
            adj[u, v] = 1
            adj[v, u] = 1
        deg = np.array([np.sum(adj[i]) for i in range(n)])
        structures.append((adj, deg))
        
        # 2. K_{8,8} + 1 edge
        adj = np.zeros((n, n), dtype=int)
        vertices = list(range(n))
        random.shuffle(vertices)
        A = vertices[:8]
        B = vertices[8:]
        for u in A:
            for v in B:
                adj[u, v] = 1
                adj[v, u] = 1
        # Add one edge in A
        random.shuffle(A)
        adj[A[0], A[1]] = 1
        adj[A[1], A[0]] = 1
        deg = np.array([np.sum(adj[i]) for i in range(n)])
        structures.append((adj, deg))
        
        # 3. Random 8-regular graph + 1 edge
        # Generate a random 8-regular graph on 16 vertices
        # This is tricky to generate directly, so we start with K_{8,8} and do random swaps to make it regular?
        # K_{8,8} is 8-regular.
        # So K_{8,8} is already 8-regular.
        # We need 65 edges. K_{8,8} has 64.
        # So we add 1 edge.
        # This is the same as structure 2.
        
        # Let's try K_{7,9} + 2 edges
        adj = np.zeros((n, n), dtype=int)
        vertices = list(range(n))
        random.shuffle(vertices)
        A = vertices[:7]
        B = vertices[7:]
        for u in A:
            for v in B:
                adj[u, v] = 1
                adj[v, u] = 1
        # Need 2 more edges. Add them in B.
        random.shuffle(B)
        adj[B[0], B[1]] = 1
        adj[B[1], B[0]] = 1
        adj[B[2], B[3]] = 1
        adj[B[3], B[2]] = 1
        deg = np.array([np.sum(adj[i]) for i in range(n)])
        structures.append((adj, deg))
        
        # 4. Random graph with 65 edges
        adj = np.zeros((n, n), dtype=int)
        random.shuffle(all_pairs)
        for i in range(65):
            u, v = all_pairs[i]
            adj[u, v] = 1
            adj[v, u] = 1
        deg = np.array([np.sum(adj[i]) for i in range(n)])
        structures.append((adj, deg))
        
        return structures

    best_overall_score = float('inf')
    best_overall_adj = None
    best_overall_deg = None
    
    structures = generate_initial_structures()
    
    # Shuffle structures to ensure diversity in search order
    random.shuffle(structures)
    
    # Time budget allocation
    # 4 structures.
    # For each: Steepest Descent (fast, ~1-2s) + Simulated Annealing (slower, ~20s)
    # Total ~ 4 * 22 = 88s. Safe.
    
    time_per_structure = (deadline - (time.time() - start_time) - 5) / len(structures)
    
    for adj, deg in structures:
        if time.time() > deadline - 5:
            break
            
        # Phase 1: Steepest Descent
        time_limit_sd = min(5, time_per_structure / 3)
        adj_sd, deg_sd, score_sd = steepest_descent(adj, deg, max_iters=10, time_limit=time_limit_sd)
        
        if score_sd < best_overall_score:
            best_overall_score = score_sd
            best_overall_adj = adj_sd
            best_overall_deg = deg_sd
            
        if best_overall_score <= 23:
            break
            
        # Phase 2: Simulated Annealing
        time_limit_sa = time_per_structure - time_limit_sd - 1
        if time_limit_sa > 1:
            adj_sa, deg_sa, score_sa = simulated_annealing(adj_sd, deg_sd, time_limit=time_limit_sa)
            
            if score_sa < best_overall_score:
                best_overall_score = score_sa
                best_overall_adj = adj_sa
                best_overall_deg = deg_sa
                
            if best_overall_score <= 23:
                break

    # Final polish: Steepest Descent on the best result
    if time.time() < deadline - 2:
        time_limit_final = deadline - 2 - (time.time() - start_time)
        if time_limit_final > 1:
            best_overall_adj, best_overall_deg, best_overall_score = steepest_descent(
                best_overall_adj, best_overall_deg, max_iters=20, time_limit=time_limit_final
            )

    # Reconstruct edge list
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_overall_adj[i, j]:
                final_edges.append([i, j])
                
    return {'n': n, 'edges': final_edges}
