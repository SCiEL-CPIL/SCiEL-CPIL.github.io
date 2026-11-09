"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency and edge set
    adj = {i: set() for i in range(n)}
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    
    def is_edge(u, v, allowed_edges):
        return (min(u,v), max(u,v)) in allowed_edges

    def cost(tour, allowed_edges):
        """Number of edges in tour that are NOT in allowed_edges."""
        c = 0
        for i in range(len(tour)):
            u = tour[i]
            v = tour[(i+1) % len(tour)]
            if not is_edge(u, v, allowed_edges):
                c += 1
        return c

    def sa_find_hamiltonian_cycle(vertices, allowed_edges, max_iters=50000, temp_start=10.0, temp_end=0.01):
        """
        Uses Simulated Annealing to find a Hamiltonian cycle in the subgraph defined by allowed_edges.
        Returns the cycle as a list of vertices or None.
        """
        m = len(vertices)
        if m < 3:
            return None
        
        # Initialize random permutation
        current_tour = vertices[:]
        random.shuffle(current_tour)
        current_cost = cost(current_tour, allowed_edges)
        
        best_tour = current_tour[:]
        best_cost = current_cost
        
        if best_cost == 0:
            return best_tour
        
        T = temp_start
        cooling_rate = (temp_end / temp_start) ** (1.0 / max_iters)
        
        for i in range(max_iters):
            if current_cost == 0:
                return current_tour
            
            # Generate neighbor: 2-opt move
            if m > 2:
                i1 = random.randint(0, m-1)
                i2 = random.randint(0, m-1)
                if i1 == i2:
                    continue
                if i1 > i2:
                    i1, i2 = i2, i1
                
                # Calculate delta cost for 2-opt
                # Edges involved: (tour[i1-1], tour[i1]), (tour[i2], tour[i2+1]) removed
                # Edges added: (tour[i1-1], tour[i2]), (tour[i1], tour[i2+1])
                # Note: indices are circular
                
                u1 = current_tour[i1-1]
                v1 = current_tour[i1]
                u2 = current_tour[i2]
                v2 = current_tour[(i2+1) % m]
                
                # If i1 and i2 are adjacent, 2-opt might degenerate or need care.
                # Simple swap is safer for Hamiltonian path/cycle in terms of implementation complexity,
                # but 2-opt is better for convergence. Let's stick to swap for robustness and simplicity, 
                # or implement 2-opt carefully.
                # Actually, for Hamiltonian Cycle, swapping two nodes is a valid move.
                # Let's use swap for simplicity and reliability.
                
                # Swap i1 and i2
                current_tour[i1], current_tour[i2] = current_tour[i2], current_tour[i1]
                new_cost = cost(current_tour, allowed_edges)
                delta = new_cost - current_cost
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_tour = current_tour[:]
                else:
                    # Revert
                    current_tour[i1], current_tour[i2] = current_tour[i2], current_tour[i1]
            
            T *= cooling_rate
            
            # Early stop if we found a solution
            if best_cost == 0:
                return best_tour
        
        if best_cost == 0:
            return best_tour
        return None

    # Main logic: Iteratively find edge-disjoint Hamiltonian cycles
    residual_edges = set(edge_set)
    cycles = []
    vertices = list(range(n))
    
    # We try to find up to 10 cycles
    max_cycles = 10
    
    for _ in range(max_cycles):
        if len(residual_edges) < n: # Need at least n edges to form a cycle of length n
            break
            
        # Try multiple times with different random seeds for SA if it fails
        found_cycle = None
        for attempt in range(5):
            random.seed() # Randomize
            found_cycle = sa_find_hamiltonian_cycle(vertices, residual_edges, max_iters=20000)
            if found_cycle:
                break
        
        if found_cycle:
            cycles.append(found_cycle)
            # Remove edges of this cycle from residual graph
            for i in range(len(found_cycle)):
                u = found_cycle[i]
                v = found_cycle[(i+1) % len(found_cycle)]
                e = (min(u,v), max(u,v))
                residual_edges.discard(e)
        else:
            break
    
    # Ensure types are correct
    result = [list(c) for c in cycles]
    return result
# EVOLVE-BLOCK-END
