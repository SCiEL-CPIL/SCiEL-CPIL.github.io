"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    vertices = list(range(n))
    
    # Build adjacency list
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        
    # To make edge-disjointness easy to track, we treat edges as frozensets
    available_edges = set(frozenset(e) for e in P['edges'])
    
    def find_hamiltonian_cycle():
        # We need to find a cycle visiting all vertices using only available edges.
        # We fix the start vertex to 0 to reduce symmetry (all cycles can be rotated to start at 0).
        start = 0
        
        def backtrack(path, visited):
            if len(path) == n:
                # Check if the last vertex connects back to the start via an available edge
                last = path[-1]
                edge_key = frozenset((last, start))
                if edge_key in available_edges:
                    return path[:]
                return None
            
            current = path[-1]
            # Get neighbors of current that are connected via available edges
            neighbors = adj[current]
            
            # Heuristic: Sort neighbors by the number of available edges they have (Warnsdorff's rule)
            # This helps guide the search towards vertices that are becoming "trapped"
            def neighbor_score(nb):
                if nb in visited:
                    return float('inf')
                # Count available edges from nb to unvisited vertices
                count = 0
                for nn in adj[nb]:
                    if nn not in visited:
                        if frozenset((nb, nn)) in available_edges:
                            count += 1
                return count
            
            # Filter valid neighbors
            valid_neighbors = [nb for nb in neighbors if nb not in visited and frozenset((current, nb)) in available_edges]
            
            # Shuffle for randomization if we want variety, but sorting by score is usually better for speed
            # We can mix: sort by score, then randomize ties or just take the best.
            # For robustness, let's sort by score.
            valid_neighbors.sort(key=neighbor_score)
            
            # If we want to explore more, we could randomize the order of neighbors with similar scores
            # But for finding ANY cycle quickly, greedy is often good.
            # However, pure greedy can get stuck. Let's try a few random restarts if this fails.
            
            for nb in valid_neighbors:
                # Use edge
                edge_key = frozenset((current, nb))
                available_edges.remove(edge_key)
                visited.add(nb)
                path.append(nb)
                
                result = backtrack(path, visited)
                if result is not None:
                    return result
                
                # Backtrack
                path.pop()
                visited.remove(nb)
                available_edges.add(edge_key)
                
            return None

        # Try multiple random restarts if the first attempt fails? 
        # Actually, the graph is small (21 nodes). Backtracking with good heuristics should be very fast.
        # Let's just run it once. If it fails, we return None.
        # But wait, if we remove edges, the graph changes. 
        # The function `find_hamiltonian_cycle` relies on the global `available_edges`.
        # This is fine as long as we restore edges if we backtrack within a single search.
        # But if we find a cycle, we keep those edges removed for the next cycle.
        
        # One issue: `backtrack` modifies `available_edges`. If it returns None, it has restored all edges it removed.
        # If it returns a path, the edges used in that path are REMOVED from `available_edges` and NOT restored.
        # This is exactly what we want: the found cycle's edges are consumed.
        
        return backtrack([start], {start})

    cycles = []
    deadline = time.time() + 50  # Leave 10s margin for safety
    
    # We know max is 3.
    for _ in range(3):
        if time.time() > deadline:
            break
        # We need to ensure the graph still allows a Hamiltonian cycle.
        # If not, find_hamiltonian_cycle will return None.
        cycle = find_hamiltonian_cycle()
        if cycle is None:
            break
        cycles.append(cycle)
        
    # If we found 0 cycles, we must return a valid non-empty list.
    # The seed program returned a single cycle using all vertices.
    # If our search failed (which shouldn't happen for the first cycle on the full graph), 
    # we should fallback to a simpler heuristic or the seed's behavior.
    if not cycles:
        # Fallback: try to find any cycle. The seed used list(range(n)).
        # Let's check if list(range(n)) is a valid cycle in the ORIGINAL graph.
        # The seed program implies it is, or at least it's a valid structure for the evaluator.
        # Actually, the seed program `return [list(range(P['n']))]` suggests [0..20] is a cycle.
        # Let's verify: 0-1-2-...-20-0.
        # Edges: (0,1), (1,2), ..., (19,20), (20,0).
        # Are these in P['edges']?
        # (0,1) Yes. (1,2) Yes. ... (19,20) Yes. (20,0) Yes.
        # So [0, 1, ..., 20] is a valid Hamiltonian cycle.
        # If our complex search failed, we can just return this one.
        # But wait, if our search failed on the first iteration, it means the graph didn't support it?
        # No, the graph DOES support it. Why would it fail?
        # Maybe backtracking got stuck? Unlikely with Warnsdorff.
        # Let's just use the fallback if cycles is empty.
        cycles = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
        # Note: We don't need to remove edges for the fallback if we only return 1 cycle.
        
    return cycles

# EVOLVE-BLOCK-END
