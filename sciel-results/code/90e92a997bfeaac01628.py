"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    
    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def edge_key(u, v):
        return (min(u, v), max(u, v))
    
    def decompose_graph(edges, time_limit, seed):
        """
        Decompose edges into cycles and single edges using Eulerian tour approach.
        """
        rng = random.Random(seed)
        remaining = set()
        for e in edges:
            remaining.add(edge_key(e[0], e[1]))
        
        if not remaining:
            return []
        
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        
        def commit_cycle(cycle_vertices):
            for i in range(len(cycle_vertices)):
                u = cycle_vertices[i]
                v = cycle_vertices[(i + 1) % len(cycle_vertices)]
                ek = edge_key(u, v)
                remaining.discard(ek)
                adj[u].discard(v)
                adj[v].discard(u)
            pieces.append(cycle_vertices[:])
        
        def commit_single_edge(u, v):
            ek = edge_key(u, v)
            remaining.discard(ek)
            adj[u].discard(v)
            adj[v].discard(u)
            pieces.append([u, v])
        
        def find_eulerian_cycle(start):
            """
            Find an Eulerian cycle in the current subgraph if it exists.
            Uses Hierholzer's algorithm.
            """
            # Check if all vertices with non-zero degree have even degree
            for v in adj:
                if len(adj[v]) > 0 and len(adj[v]) % 2 != 0:
                    return None
            
            # Find a connected component with edges
            visited_edges = set()
            stack = [start]
            circuit = []
            
            while stack:
                current = stack[-1]
                neighbors = list(adj[current])
                # Remove visited edges from consideration
                unvisited = [n for n in neighbors if edge_key(current, n) not in visited_edges]
                
                if not unvisited:
                    circuit.append(stack.pop())
                else:
                    # Choose a neighbor (randomize for different trials)
                    rng.shuffle(unvisited)
                    next_node = unvisited[0]
                    ek = edge_key(current, next_node)
                    visited_edges.add(ek)
                    stack.append(next_node)
            
            # Check if we used all edges
            if len(visited_edges) != len(remaining):
                return None
            
            # Reverse to get the circuit in order
            circuit.reverse()
            # Remove duplicate end (start == end in circuit)
            if len(circuit) >= 2 and circuit[0] == circuit[-1]:
                circuit.pop()
            
            return circuit
        
        def find_long_cycle_greedy(adj, start, time_limit):
            """
            Find a long cycle using a greedy approach with backtracking.
            """
            best_cycle = None
            best_len = 0
            
            # Try multiple random restarts
            for attempt in range(10):
                if time.time() > time_limit:
                    break
                
                path = [start]
                visited = {start}
                current = start
                
                # Greedy: always go to unvisited neighbor with highest degree
                while True:
                    neighbors = [n for n in adj[current] if n not in visited]
                    
                    if not neighbors:
                        # Dead end
                        if len(path) >= 3 and start in adj[current]:
                            if len(path) > best_len:
                                best_len = len(path)
                                best_cycle = path[:]
                        break
                    
                    # Choose neighbor with highest degree (greedy)
                    neighbors.sort(key=lambda n: len(adj[n]), reverse=True)
                    # Add some randomness
                    if rng.random() < 0.3:
                        next_node = rng.choice(neighbors)
                    else:
                        next_node = neighbors[0]
                    
                    visited.add(next_node)
                    path.append(next_node)
                    current = next_node
                    
                    # Check if we can close the cycle
                    if current != start and start in adj[current] and len(path) >= 3:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                        break
                
                # Early stop if we found a very long cycle
                if best_len >= 15:
                    break
            
            return best_cycle
        
        # Repeatedly find and remove cycles
        while remaining:
            if time.time() > time_limit:
                break
            
            # Find vertices with degree >= 2
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                # No cycles left, commit all remaining as singles
                for u, v in list(remaining):
                    commit_single_edge(u, v)
                break
            
            # First, check if the whole remaining graph is Eulerian
            eulerian_cycle = find_eulerian_cycle(candidates[0])
            if eulerian_cycle is not None:
                commit_cycle(eulerian_cycle)
                continue
            
            # Try to find a long cycle
            best_cycle = None
            best_len = 0
            
            # Shuffle candidates to randomize search
            rng.shuffle(candidates)
            
            # Limit the number of attempts to avoid getting stuck
            max_attempts = min(20, len(candidates))
            
            for start in candidates[:max_attempts]:
                if time.time() > time_limit:
                    break
                
                # Randomized greedy to find a cycle
                cycle = find_long_cycle_greedy(adj, start, time_limit)
                if cycle is not None:
                    if len(cycle) > best_len:
                        best_cycle = cycle
                        best_len = len(cycle)
                        # If we found a very long cycle, stop early
                        if best_len >= 12:
                            break
            
            if best_cycle:
                commit_cycle(best_cycle)
            else:
                # Fallback: take a single edge to ensure progress
                if remaining:
                    u, v = list(remaining)[0]
                    commit_single_edge(u, v)
        
        # Any remaining edges become single-edge pieces
        for u, v in list(remaining):
            commit_single_edge(u, v)
            
        return pieces
    
    # Main: decompose each instance
    results = []
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    
    for spec in specs:
        if time.time() > deadline - 10:
            # Not enough time, use trivial solution
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        
        # Try multiple random seeds and keep the best
        best_pieces = None
        best_count = float('inf')
        
        # Allocate time for this instance
        time_per_instance = min(45, deadline - time.time() - 10)
        if time_per_instance < 5:
            time_per_instance = 5
            
        # Run multiple trials with different random seeds
        num_trials = 25
        trial_time = time_per_instance / num_trials
        
        for t in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Time limit for this specific trial
            trial_deadline = time.time() + trial_time
            
            pieces = decompose_graph(edges, trial_deadline, 2000 + t)
            count = len(pieces)
            
            if count < best_count:
                best_count = count
                best_pieces = pieces
        
        if best_pieces is None:
            best_pieces = [list(e) for e in edges]
        
        results.append(best_pieces)
    
    return results
# EVOLVE-BLOCK-END
