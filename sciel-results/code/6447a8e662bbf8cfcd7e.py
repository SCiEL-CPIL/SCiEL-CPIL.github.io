"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys
import math

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((min(u, v), max(u, v)))
        
    # Precompute edge keys for fast lookup
    def get_edge_key(u, v):
        return (min(u, v), max(u, v))

    def find_random_cycle(used_edges):
        """
        Finds a random simple cycle in the graph avoiding used_edges.
        Returns list of vertices or None.
        """
        # Pick a random start vertex
        start = random.randint(0, n - 1)
        
        # Randomize neighbor order for diversity
        adj_copy = [list(nei) for nei in adj]
        for i in range(n):
            random.shuffle(adj_copy[i])
            
        path = [start]
        visited = [False] * n
        visited[start] = True
        
        def dfs():
            if len(path) >= 3:
                last = path[-1]
                # Check if we can close the cycle back to start
                if start in adj_copy[last]:
                    ek = get_edge_key(last, start)
                    if ek not in used_edges:
                        return True
            
            last = path[-1]
            for nxt in adj_copy[last]:
                if nxt == start:
                    # Only allow closing if path length >= 3 (i.e., len(path) >= 3 means at least 3 vertices)
                    # Actually, cycle of 3 vertices: A-B-C-A. Path [A,B,C]. len=3.
                    # If len(path) < 3, we can't close.
                    if len(path) < 3:
                        continue
                    ek = get_edge_key(last, start)
                    if ek not in used_edges:
                        return True
                    continue
                
                if not visited[nxt]:
                    ek = get_edge_key(last, nxt)
                    if ek not in used_edges:
                        visited[nxt] = True
                        path.append(nxt)
                        
                        if dfs():
                            return True
                            
                        path.pop()
                        visited[nxt] = False
                        
            return False
            
        if dfs():
            return path[:]
        return None

    def greedy_fill(used_edges, cycles):
        """
        Greedily add as many cycles as possible.
        """
        attempts = 0
        while attempts < 100: # Limit attempts to avoid infinite loops if no cycle found
            cycle = find_random_cycle(used_edges)
            if cycle is None:
                break
            
            # Mark edges as used
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                used_edges.add(get_edge_key(u, v))
            
            cycles.append(cycle)
            attempts += 1
        return cycles

    # 1. Initial Greedy Solution
    used_edges = set()
    cycles = []
    greedy_fill(used_edges, cycles)
    
    best_cycles = [c[:] for c in cycles]
    best_count = len(best_cycles)
    
    # 2. Simulated Annealing
    current_cycles = [c[:] for c in cycles]
    current_used = set(used_edges)
    current_count = len(current_cycles)
    
    T = 10.0
    cooling_rate = 0.99
    iterations = 5000
    
    for _ in range(iterations):
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to Add a cycle
            new_cycle = find_random_cycle(current_used)
            if new_cycle is not None:
                # Always accept improvement
                for i in range(len(new_cycle)):
                    u = new_cycle[i]
                    v = new_cycle[(i + 1) % len(new_cycle)]
                    current_used.add(get_edge_key(u, v))
                
                current_cycles.append(new_cycle)
                current_count += 1
                
                if current_count > best_count:
                    best_count = current_count
                    best_cycles = [c[:] for c in current_cycles]
        else:
            # Try to Swap a cycle
            if current_count == 0:
                continue
                
            idx = random.randint(0, current_count - 1)
            old_cycle = current_cycles[idx]
            
            # Remove old cycle edges
            for i in range(len(old_cycle)):
                u = old_cycle[i]
                v = old_cycle[(i + 1) % len(old_cycle)]
                current_used.remove(get_edge_key(u, v))
            
            # Try to find a NEW cycle different from old_cycle
            # We might find the same cycle again, which is a wasted move but acceptable.
            # To encourage change, we could bias, but random search is fine.
            new_cycle = find_random_cycle(current_used)
            
            if new_cycle is not None:
                # Check if it's effectively different (to avoid useless swaps)
                # Simple check: set of edges
                old_edges = set()
                for i in range(len(old_cycle)):
                    u = old_cycle[i]
                    v = old_cycle[(i + 1) % len(old_cycle)]
                    old_edges.add(get_edge_key(u, v))
                    
                new_edges = set()
                for i in range(len(new_cycle)):
                    u = new_cycle[i]
                    v = new_cycle[(i + 1) % len(new_cycle)]
                    new_edges.add(get_edge_key(u, v))
                    
                if new_edges != old_edges:
                    # Accept the swap (count remains same)
                    # This explores the space of possible cycle sets
                    for i in range(len(new_cycle)):
                        u = new_cycle[i]
                        v = new_cycle[(i + 1) % len(new_cycle)]
                        current_used.add(get_edge_key(u, v))
                        
                    current_cycles[idx] = new_cycle
                    # current_count remains same
                else:
                    # Revert if we found the exact same cycle
                    for i in range(len(old_cycle)):
                        u = old_cycle[i]
                        v = old_cycle[(i + 1) % len(old_cycle)]
                        current_used.add(get_edge_key(u, v))
                    # current_cycles[idx] is still old_cycle
            else:
                # Failed to find new cycle, revert
                for i in range(len(old_cycle)):
                    u = old_cycle[i]
                    v = old_cycle[(i + 1) % len(old_cycle)]
                    current_used.add(get_edge_key(u, v))
                # current_cycles[idx] is still old_cycle

        T *= cooling_rate
        if T < 0.01:
            break

    return best_cycles
# EVOLVE-BLOCK-END
