"""Model-and-optimize approach for few cycle lengths under degree/girth constraints."""
# EVOLVE-BLOCK-START

import time
import itertools

def construct():
    """
    Constructs a graph on 12 vertices (0..11) that is triangle-free, has minimum degree >= 3,
    and minimizes the number of distinct simple cycle lengths.
    
    Strategy: 
    1. The theoretical minimum is 1 (a graph where all cycles have the same length).
    2. We search for such a graph using a randomized local search / random sampling approach.
    3. We specifically target graphs that might have only 5-cycles (like the Petersen graph structure) 
       or only 4-cycles.
    4. If no graph with 1 distinct cycle length is found within the time budget, we fall back 
       to a known valid graph with 2 distinct cycle lengths (two disjoint K_{3,3} components).
    
    Deadline guard: 85s.
    """
    start_time = time.time()
    timeout = 85.0  # Leave 5s margin for safety

    N = 12

    def get_cycle_lengths(graph):
        """
        Returns the set of distinct simple cycle lengths in the graph.
        Uses DFS with pruning for small graphs.
        """
        adj = {i: set() for i in range(N)}
        for u, v in graph:
            adj[u].add(v)
            adj[v].add(u)
        
        cycle_lengths = set()
        
        # For each start node, find cycles where start is the smallest node
        for start in range(N):
            # We only explore nodes >= start to avoid duplicates and ensure start is min
            # But we must be careful: the cycle must consist of nodes >= start?
            # Yes, if start is the minimum, all other nodes in the cycle must be > start.
            
            # DFS state: (current, visited_set, length)
            # We use recursion for simplicity as N is small
            
            def dfs(current, visited, length):
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
                    
                    if neighbor in visited:
                        continue
                    
                    # Pruning: Only visit neighbors > start (since start is min)
                    # Wait, if neighbor == start, handled above.
                    # If neighbor < start, we skip because start is supposed to be the min.
                    if neighbor < start:
                        continue
                    
                    visited.add(neighbor)
                    dfs(neighbor, visited, length + 1)
                    visited.remove(neighbor)
            
            visited = {start}
            dfs(start, visited, 0)
            
        return cycle_lengths

    def is_valid(graph):
        """Check min degree >= 3 and triangle-free."""
        adj = {i: set() for i in range(N)}
        for u, v in graph:
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

    def edge_to_index(u, v):
        if u > v:
            u, v = v, u
        return u * N + v

    def index_to_edge(idx):
        u = idx // N
        v = idx % N
        return u, v

    # All possible edges
    all_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            all_edges.append((u, v))
    
    num_edges = len(all_edges)
    
    # Precompute edge indices for fast lookup
    edge_set_to_idx = {edge_to_index(u, v): i for i, (u, v) in enumerate(all_edges)}
    
    def random_graph():
        """Generate a random graph with min degree >= 3 and triangle-free."""
        # This is hard to generate directly. 
        # Instead, we start with a random sparse graph and try to fix degrees, 
        # or just sample and check.
        # Given the constraints, sampling might be inefficient.
        
        # Let's try a different strategy:
        # Start with a known valid graph (e.g., Petersen-like or K_3,3 based) and mutate.
        # Or, just do random sampling of edge subsets and check validity.
        
        # For 12 vertices, C(12,2) = 66 edges.
        # We need at least 18 edges (12 * 3 / 2).
        
        # Let's try to construct a random bipartite graph first?
        # No, Petersen is not bipartite.
        
        # Let's just do random sampling of edges with a bias towards sparse graphs.
        # Probability p such that expected degree is around 3-4.
        # E[deg] = (N-1)p. If N=12, 11p = 3.5 => p = 0.32.
        
        p = 0.35
        edges = []
        for u, v in all_edges:
            import random
            if random.random() < p:
                edges.append([u, v])
        
        return edges

    def mutate(edges, adj):
        """Flip a random edge."""
        import random
        new_edges = edges[:]
        idx = random.randint(0, num_edges - 1)
        u, v = all_edges[idx]
        if edge_to_index(u, v) in adj:
            new_edges.remove([u, v])
        else:
            new_edges.append([u, v])
        return new_edges

    def build_adj(edges):
        adj = set()
        for u, v in edges:
            adj.add(edge_to_index(u, v))
        return adj

    # Initial best
    best_raw = 3
    best_edges = None
    
    # Try to find a graph with raw value 1
    # We will use a simple random restart local search.
    
    import random
    
    max_iterations = 500000
    for _ in range(max_iterations):
        if time.time() - start_time > timeout:
            break
            
        # Generate a random starting point
        edges = random_graph()
        
        if not is_valid(edges):
            continue
            
        lengths = get_cycle_lengths(edges)
        if not lengths:
            continue
            
        raw = len(lengths)
        
        if raw < best_raw:
            best_raw = raw
            best_edges = [e[:] for e in edges]
            if best_raw == 1:
                break
            
        # Local search: try to improve
        # Since we only care about raw value 1, if we are at 2, we try to get to 1.
        # If we are at 3, we try to get to 2 or 1.
        
        # We'll do a few steps of local search
        for step in range(20):
            if time.time() - start_time > timeout:
                break
                
            new_edges = mutate(edges, build_adj(edges))
            if not is_valid(new_edges):
                continue
                
            new_lengths = get_cycle_lengths(new_edges)
            if not new_lengths:
                continue
                
            new_raw = len(new_lengths)
            
            if new_raw < raw:
                raw = new_raw
                edges = new_edges
                if raw < best_raw:
                    best_raw = raw
                    best_edges = [e[:] for e in edges]
                    if best_raw == 1:
                        break
            elif new_raw == raw and random.random() < 0.5:
                # Accept sideways moves to escape local minima
                edges = new_edges
                
        if best_raw == 1:
            break

    # Fallback: Two disjoint K_{3,3}
    if best_edges is None or best_raw > 2:
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
    else:
        return best_edges

# EVOLVE-BLOCK-END
