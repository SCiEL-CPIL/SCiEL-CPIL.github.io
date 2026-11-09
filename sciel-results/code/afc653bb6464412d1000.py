"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}
    n = P['n']
    edges = [tuple(sorted(e)) for e in P['edges']]
    edge_set = set(edges)
    
    # Adjacency list for original graph
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find a Hamiltonian cycle as the initial solution
    def find_hamiltonian():
        path = [0]
        visited = {0}
        
        def backtrack(u):
            if len(path) == n:
                if 0 in adj[u]:
                    return True
                return False
            for v in sorted(adj[u] - visited):
                path.append(v)
                visited.add(v)
                if backtrack(v):
                    return True
                visited.remove(v)
                path.pop()
            return False
        
        if backtrack(0):
            return path
        return None
    
    seed_cycle = find_hamiltonian()
    best_sol = [seed_cycle] if seed_cycle else []
    
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    def find_cycle(rem_edge_set, used_cycles):
        """Find a Hamiltonian cycle in the remaining graph that doesn't conflict with used cycles."""
        if not rem_edge_set:
            return None
            
        # Build adjacency for remaining edges
        rem_adj = {i: set() for i in range(n)}
        for u, v in rem_edge_set:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
            
        # Check connectivity
        visited = {0}
        stack = [0]
        while stack:
            u = stack.pop()
            for v in rem_adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        if len(visited) < n:
            return None
        
        # Check degree: every vertex must have degree >= 2 in remaining graph
        for i in range(n):
            if len(rem_adj[i]) < 2:
                return None
        
        # Randomized backtracking with multiple restarts
        for attempt in range(50):
            random.seed(random.randint(0, 2**31))
            path = [0]
            visited = {0}
            
            def backtrack(u):
                if len(path) == n:
                    if 0 in rem_adj[u]:
                        return True
                    return False
                
                # Get unvisited neighbors
                candidates = [v for v in rem_adj[u] if v not in visited]
                if not candidates:
                    return False
                
                # Warnsdorff's rule: prefer vertices with fewer onward moves
                candidates.sort(key=lambda x: len(rem_adj[x] - visited))
                
                # Randomize slightly to escape local optima
                if random.random() < 0.3 and len(candidates) > 1:
                    random.shuffle(candidates)
                
                for v in candidates:
                    path.append(v)
                    visited.add(v)
                    if backtrack(v):
                        return True
                    visited.remove(v)
                    path.pop()
                return False
            
            if backtrack(0):
                # Verify the cycle is valid and edge-disjoint from used cycles
                cycle_edges = set()
                for i in range(n):
                    u = path[i]
                    v = path[(i + 1) % n]
                    e = tuple(sorted((u, v)))
                    if e not in rem_edge_set:
                        return None
                    cycle_edges.add(e)
                
                # Check no edge conflicts with previous cycles
                for c in used_cycles:
                    for i in range(len(c)):
                        u = c[i]
                        v = c[(i + 1) % len(c)]
                        e = tuple(sorted((u, v)))
                        if e in cycle_edges:
                            return None
                
                return path
        
        # Simulated annealing fallback for harder instances
        if time.time() < deadline:
            def hamiltonian_sa():
                """Simulated annealing on permutations to find Hamiltonian cycle."""
                import math
                
                # Initialize with a random valid permutation
                def random_perm():
                    perm = list(range(1, n))
                    random.shuffle(perm)
                    perm.insert(0, 0)
                    return perm
                
                def is_valid_cycle(perm):
                    """Check if perm forms a valid Hamiltonian cycle in rem_adj."""
                    for i in range(n):
                        u = perm[i]
                        v = perm[(i + 1) % n]
                        if v not in rem_adj[u]:
                            return False
                    return True
                
                def edge_cost(perm):
                    """Count number of invalid edges in the cycle."""
                    cost = 0
                    for i in range(n):
                        u = perm[i]
                        v = perm[(i + 1) % n]
                        if v not in rem_adj[u]:
                            cost += 1
                    return cost
                
                # SA parameters
                t0 = 1.0
                t_end = 0.001
                cooling = 0.999
                max_iters = 5000
                
                best_perm = random_perm()
                best_cost = edge_cost(best_perm)
                current_perm = best_perm[:]
                current_cost = best_cost
                t = t0
                
                for _ in range(max_iters):
                    if current_cost == 0:
                        break
                    
                    # 2-opt move: reverse a segment
                    if n > 3:
                        i, j = sorted(random.sample(range(n), 2))
                        new_perm = current_perm[:]
                        new_perm[i:j+1] = reversed(new_perm[i:j+1])
                    else:
                        # Swap two positions
                        i, j = sorted(random.sample(range(n), 2))
                        new_perm = current_perm[:]
                        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                    
                    new_cost = edge_cost(new_perm)
                    delta = new_cost - current_cost
                    
                    if delta <= 0 or random.random() < math.exp(-delta / max(t, 1e-10)):
                        current_perm = new_perm
                        current_cost = new_cost
                        
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_perm = current_perm[:]
                    
                    t *= cooling
                    if t < t_end:
                        break
                
                if best_cost == 0:
                    # Verify edge-disjointness with used cycles
                    cycle_edges = set()
                    for i in range(n):
                        u = best_perm[i]
                        v = best_perm[(i + 1) % n]
                        e = tuple(sorted((u, v)))
                        if e not in rem_edge_set:
                            return None
                        cycle_edges.add(e)
                    
                    for c in used_cycles:
                        for i in range(len(c)):
                            u = c[i]
                            v = c[(i + 1) % len(c)]
                            e = tuple(sorted((u, v)))
                            if e in cycle_edges:
                                return None
                    
                    return best_perm
                return None
            
            sa_result = hamiltonian_sa()
            if sa_result is not None:
                return sa_result
        
        return None
    
    # Search for maximum number of edge-disjoint cycles
    # Try multiple random orderings of edge removal to find different cycle sets
    num_trials = 20
    for trial in range(num_trials):
        if time.time() >= deadline:
            break
        
        rem_edges = set(edge_set)
        cycles = []
        
        while time.time() < deadline:
            c = find_cycle(rem_edges, cycles)
            if c is None:
                break
            cycles.append(c)
            
            # Remove edges of this cycle from remaining edges
            for i in range(n):
                u = c[i]
                v = c[(i + 1) % n]
                e = tuple(sorted((u, v)))
                if e in rem_edges:
                    rem_edges.remove(e)
            
            # Update best solution
            if len(cycles) > len(best_sol):
                best_sol = [list(c) for c in cycles]
            
            # If we've reached the theoretical maximum (3 for 6-regular graph), stop
            if len(best_sol) == 3:
                return best_sol
    
    # Return best solution found, ensuring proper types
    result = []
    for c in best_sol:
        result.append([int(v) for v in c])
    
    if not result:
        # Fallback: return the seed cycle if we found one
        if seed_cycle:
            result = [[int(v) for v in seed_cycle]]
        else:
            # Last resort: construct a trivial valid cycle if possible
            # Find any 3-cycle
            for e1 in edges:
                for e2 in edges:
                    if e1[1] == e2[0] and e1 != e2:
                        if (e1[0], e2[1]) in edge_set or (e2[1], e1[0]) in edge_set:
                            result = [[e1[0], e1[1], e2[1]]]
                            break
                if result:
                    break
            if not result:
                # Return the Hamiltonian cycle as fallback
                if seed_cycle:
                    result = [[int(v) for v in seed_cycle]]
                else:
                    result = [[0, 1, 2]]  # Should not happen
    
    return result
# EVOLVE-BLOCK-END
