"""Model-and-optimize approach for few cycle lengths under degree/girth constraints."""
# EVOLVE-BLOCK-START

import time
import random
import math

def construct():
    """
    Constructs a graph on 12 vertices (0..11) that is triangle-free, has minimum degree >= 3,
    and minimizes the number of distinct simple cycle lengths.
    
    Strategy:
    1. The theoretical minimum is 1 (a graph where all cycles have the same length).
    2. We search for such a graph using a randomized local search (Simulated Annealing).
    3. We specifically target graphs that might have only 4-cycles, 5-cycles, or 6-cycles.
    4. If no graph with 1 distinct cycle length is found within the time budget, we fall back 
       to a known valid graph with 2 distinct cycle lengths (two disjoint K_{3,3} components).
    
    Deadline guard: 85s.
    """
    start_time = time.time()
    timeout = 85.0  # Leave 5s margin for safety

    N = 12
    
    # Precompute all possible edges
    all_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            all_edges.append((u, v))
    
    num_edges = len(all_edges)
    
    # Map edge to index for fast lookup in adjacency sets
    edge_to_idx = {}
    for i, (u, v) in enumerate(all_edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i

    def get_cycle_lengths(edges):
        """
        Returns the set of distinct simple cycle lengths in the graph.
        """
        adj = [set() for _ in range(N)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cycle_lengths = set()
        
        # For each start node, find cycles where start is the smallest node
        for start in range(N):
            # DFS state: (current, visited_mask, length)
            # visited_mask is an integer bitmask
            
            def dfs(current, visited_mask, length):
                if length >= 3 and current == start:
                    cycle_lengths.add(length)
                    return
                
                if length > N:
                    return
                
                for neighbor in adj[current]:
                    if neighbor == start:
                        if length >= 3:
                            cycle_lengths.add(length)
                        continue
                    
                    # Check if neighbor is already visited
                    if visited_mask & (1 << neighbor):
                        continue
                    
                    # Pruning: Only visit neighbors > start (since start is min)
                    if neighbor < start:
                        continue
                    
                    dfs(neighbor, visited_mask | (1 << neighbor), length + 1)
            
            dfs(start, 1 << start, 0)
            
        return cycle_lengths

    def is_valid(edges):
        """Check min degree >= 3 and triangle-free."""
        adj = [set() for _ in range(N)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Triangle-free check
        for u in range(N):
            for v in adj[u]:
                if u < v:
                    if adj[u].intersection(adj[v]):
                        return False
        
        # Min degree check
        for u in range(N):
            if len(adj[u]) < 3:
                return False
                
        return True

    def get_cost(edges):
        """
        Returns a cost value.
        Lower is better.
        Invalid graphs have high cost.
        """
        if not is_valid(edges):
            return 1000
        lengths = get_cycle_lengths(edges)
        if not lengths:
            return 500 # Valid but no cycles (tree/forest) - though min deg 3 prevents this for N>2
        return len(lengths)

    def random_graph():
        """Generate a random graph with min degree >= 3 and triangle-free."""
        # Start with a random bipartite graph with high density
        # Split 6-6
        left = list(range(6))
        right = list(range(6, 12))
        edges = []
        for u in left:
            for v in right:
                if random.random() < 0.7:
                    edges.append([u, v])
        
        # Ensure min degree 3
        # This might fail, in which case we return a known valid graph
        # For simplicity, we just return the random edges and let the search fix it
        return edges

    def known_valid_graph():
        """Returns two disjoint K_{3,3}."""
        edges = []
        part1_left = [0, 1, 2]
        part1_right = [3, 4, 5]
        for u in part1_left:
            for v in part1_right:
                edges.append([u, v])
                
        part2_left = [6, 7, 8]
        part2_right = [9, 10, 11]
        for u in part2_left:
            for v in part2_right:
                edges.append([u, v])
        return edges

    # Initialize best
    best_edges = known_valid_graph()
    best_cost = get_cost(best_edges) # Should be 2
    
    # Local Search
    current_edges = random_graph()
    current_cost = get_cost(current_edges)
    
    # If initial is invalid, use known valid
    if current_cost > 100:
        current_edges = known_valid_graph()
        current_cost = get_cost(current_edges)
        
    T = 1.0
    cooling = 0.995
    
    while time.time() - start_time < timeout:
        # Generate neighbor
        idx = random.randint(0, num_edges - 1)
        u, v = all_edges[idx]
        
        new_edges = current_edges[:]
        # Flip edge
        # Check if edge exists
        edge_exists = False
        for i, e in enumerate(new_edges):
            if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                new_edges.pop(i)
                edge_exists = True
                break
        
        if not edge_exists:
            new_edges.append([u, v])
            
        new_cost = get_cost(new_edges)
        
        delta = new_cost - current_cost
        
        if delta < 0:
            # Improvement
            current_edges = new_edges
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_edges = new_edges
                
                if best_cost == 1:
                    # Found optimal
                    break
        else:
            # Accept with probability
            if delta == 0:
                # Sideways move
                if random.random() < 0.5:
                    current_edges = new_edges
            else:
                # Worse
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_edges = new_edges
                    current_cost = new_cost
        
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Return best found
    return best_edges

# EVOLVE-BLOCK-END
