import time
import random
import math

def _count_c4s(n, adj):
    """
    Count the number of C4s in the graph.
    A C4 is a set of 4 vertices {u,v,w,x} such that edges (u,v), (v,w), (w,x), (x,u) exist.
    In a simple graph, a C4 corresponds to two pairs of vertices having exactly 2 common neighbors.
    Specifically, for each pair (u, v) that is an edge, if they have k common neighbors, 
    the number of C4s involving edge (u,v) is C(k, 2) = k*(k-1)/2.
    However, each C4 has 4 edges, so if we sum C(common_neighbors(u,v), 2) over all edges (u,v),
    we count each C4 exactly 4 times.
    So total C4s = (1/4) * sum_{(u,v) in E} C(deg_common(u,v), 2).
    """
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                common = adj[u] & adj[v]
                k = common.bit_count()
                if k >= 2:
                    total += k * (k - 1) // 2
    return total // 4

def _min_degree(n, adj):
    min_deg = n
    for u in range(n):
        d = adj[u].bit_count()
        if d < min_deg:
            min_deg = d
    return min_deg

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    best_n = 39
    best_edges = []
    
    # Target n values to try, starting from the highest possible
    # R(C4, K1,39) is known to be 46 or 47. 
    # If we find n=46, score is 46/39 ~ 1.179.
    # If we find n=47, score is 47/39 ~ 1.205.
    # Current best is 1.1282 which corresponds to n=44 (44/39 = 1.128).
    # So we must aim for 46 or 47.
    
    for target_n in [47, 46, 45, 44]:
        if time.time() > deadline:
            break
            
        min_deg_needed = target_n - 39
        
        # Run Simulated Annealing
        found = False
        
        # Try a few different initializations/SA runs
        for run_id in range(10):
            if time.time() > deadline:
                break
            
            random.seed(12345 + run_id * 100 + target_n)
            
            n = target_n
            adj = [0] * n
            
            # Initialization: Random graph with density p
            # We want min degree ~ min_deg_needed.
            # For n=46, min_deg_needed=7.
            # For n=47, min_deg_needed=8.
            # Let's start with a random graph with p=0.2
            p = 0.2
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < p:
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
            
            # Simulated Annealing Parameters
            # Objective: Minimize (C4_count * 1000) + (max(0, min_deg_needed - min_deg) * 100)
            # Actually, we want to MAXIMIZE min_degree subject to C4_count == 0.
            # Let's define Energy E = C4_count * 10000 + max(0, min_deg_needed - min_deg) * 1000
            # We want to minimize E.
            
            def get_energy(adj):
                c4s = _count_c4s(n, adj)
                md = _min_degree(n, adj)
                penalty = 0
                if md < min_deg_needed:
                    penalty = (min_deg_needed - md) * 1000
                return c4s * 10000 + penalty
            
            current_energy = get_energy(adj)
            best_energy = current_energy
            best_adj = [a for a in adj]
            
            T = 1000.0
            cooling_rate = 0.99
            min_T = 1.0
            
            # Move operator: Edge Swap
            # Pick two non-edges (u,v) and (x,y) and swap? No, that's complex.
            # Better: Pick an edge (u,v) to remove, and a non-edge (x,y) to add.
            # Or simpler: Pick a vertex u with low degree, pick a vertex v not connected to u.
            # Try to add (u,v). If it creates C4, maybe remove an edge (w,z) that breaks the C4?
            
            # Let's use a more standard SA move: 
            # 1. Pick a random edge (u,v) and remove it.
            # 2. Pick a random non-edge (x,y) and add it.
            # This keeps edge count constant, which helps maintain density.
            
            # However, we need to ensure we don't drop below min_deg_needed if we are already above it.
            # Let's use a mixed move set:
            # - Add edge (u,v)
            # - Remove edge (u,v)
            # - Swap: Remove (u,v), Add (x,y)
            
            for step in range(50000): # Fixed number of steps per run, controlled by time
                if time.time() > deadline:
                    break
                
                # Generate a move
                move_type = random.random()
                
                new_adj = [a for a in adj]
                
                if move_type < 0.4:
                    # Add a random non-edge
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v:
                        continue
                    if (adj[u] >> v) & 1:
                        continue
                    new_adj[u] |= 1 << v
                    new_adj[v] |= 1 << u
                elif move_type < 0.8:
                    # Remove a random edge
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v:
                        continue
                    if not ((adj[u] >> v) & 1):
                        continue
                    new_adj[u] &= ~(1 << v)
                    new_adj[v] &= ~(1 << u)
                else:
                    # Swap: Remove (u,v), Add (x,y)
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v or not ((adj[u] >> v) & 1):
                        continue
                    x = random.randint(0, n - 1)
                    y = random.randint(0, n - 1)
                    if x == y or ((adj[x] >> y) & 1):
                        continue
                    # Ensure we don't add an edge that is already being removed or vice versa
                    if (x == u and y == v) or (x == v and y == u):
                        continue
                    new_adj[u] &= ~(1 << v)
                    new_adj[v] &= ~(1 << u)
                    new_adj[x] |= 1 << y
                    new_adj[y] |= 1 << x
                
                new_energy = get_energy(new_adj)
                delta_e = new_energy - current_energy
                
                if delta_e < 0:
                    # Accept
                    adj = new_adj
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_adj = [a for a in adj]
                        # Check if we found a valid solution
                        if best_energy == 0:
                            found = True
                            # Extract edges
                            edges = []
                            for u in range(n):
                                for v in range(u + 1, n):
                                    if (best_adj[u] >> v) & 1:
                                        edges.append([u, v])
                            if target_n > best_n:
                                best_n = target_n
                                best_edges = edges
                            break
                else:
                    # Metropolis criterion
                    if T > min_T:
                        prob = math.exp(-delta_e / T)
                        if random.random() < prob:
                            adj = new_adj
                            current_energy = new_energy
                            
                T *= cooling_rate
                
            if found:
                break
                
        if found:
            break
            
    if best_n >= 40:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback to a known valid small graph if SA fails completely
        # n=39 is always valid (min deg 0)
        return {'n': 39, 'edges': []}
