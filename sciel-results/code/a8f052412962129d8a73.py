import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 55s limit to be safe
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Shuffle neighbors for randomization
    for i in range(n):
        random.shuffle(adj[i])
        
    # Set of used edges (frozenset to be hashable)
    used_edges = set()
    found_cycles = []
    
    def find_cycle(start, current, path, visited):
        # Time check
        if time.time() > deadline:
            return None
            
        if len(path) == n:
            # Check if last connects to start
            if start in adj[current]:
                # Check if edge (current, start) is not used
                e = frozenset((current, start))
                if e not in used_edges:
                    return list(path)
            return None
            
        for neighbor in adj[current]:
            if neighbor in visited:
                continue
            # Check if edge (current, neighbor) is used
            e = frozenset((current, neighbor))
            if e in used_edges:
                continue
                
            # Simple pruning: if neighbor is not start and we have visited all others, 
            # we must go to start. But visited check handles that mostly.
            
            # Connectivity pruning: 
            # If we are at step k, the remaining unvisited vertices must be connected 
            # to each other and to 'neighbor' (and eventually 'start' if last step).
            # For n=21, this might be too expensive to check at every step.
            # We'll rely on randomization and speed.
            
            path.append(neighbor)
            visited.add(neighbor)
            
            res = find_cycle(start, neighbor, path, visited)
            if res:
                return res
                
            path.pop()
            visited.remove(neighbor)
            
        return None

    # Try to find up to 3 cycles
    # We can restart the search if it fails or just try once with randomization.
    # Given the graph density, a single randomized DFS often finds one quickly.
    # We might need to reset the random seed or shuffle order if it gets stuck.
    
    for _ in range(3):
        # Reset randomization for a fresh attempt if previous failed or to vary path
        # Actually, we just continue with the current state of used_edges.
        # If find_cycle returns None, we might be stuck. 
        # For now, we just try once per "outer" iteration, but since adj is static, 
        # we should re-shuffle or use a different start node if it fails.
        
        # Let's try a few different start nodes if the first one fails
        found = False
        for start_node in range(n):
            if time.time() > deadline:
                break
            # Re-shuffle neighbors to get a different path order
            for i in range(n):
                random.shuffle(adj[i])
                
            res = find_cycle(start_node, start_node, [start_node], {start_node})
            if res:
                # Verify cycle
                valid = True
                cycle_edges = set()
                for i in range(len(res)):
                    u = res[i]
                    v = res[(i+1) % len(res)]
                    e = frozenset((u, v))
                    if e in used_edges:
                        valid = False
                        break
                    if e in cycle_edges:
                        valid = False
                        break
                    cycle_edges.add(e)
                
                if valid:
                    found_cycles.append(res)
                    for e in cycle_edges:
                        used_edges.add(e)
                    found = True
                    break # Found one, move to next outer iteration
        
        if not found:
            break # Couldn't find another cycle
            
    # Return the list of cycles
    # Ensure types are correct
    result = []
    for c in found_cycles:
        result.append([int(v) for v in c])
        
    if not result:
        # Fallback: return a single valid cycle if possible, else empty? 
        # Task says "nonempty list". 
        # If we found 0, we should try harder or return a known valid one?
        # The seed returns list(range(n)) which is a cycle? 
        # 0-1-2...20-0. 
        # Is (20,0) an edge? Yes, [0,20] is in edges.
        # Is (i, i+1) an edge? 
        # 0-1 yes, 1-2 yes, 2-3 yes, 3-4 yes, 4-5 yes, 5-6 yes, 6-7 yes, 7-8 yes, 8-9 yes, 9-10 yes, 10-11 yes, 11-12 yes, 12-13 yes, 13-14 yes, 14-15 yes, 15-16 yes, 16-17 yes, 17-18 yes, 18-19 yes, 19-20 yes.
        # So list(range(21)) is a valid Hamiltonian cycle.
        # If our search failed, we can return this one to ensure validity and score 1.0.
        return [list(range(n))]
        
    return result

# EVOLVE-BLOCK-END
