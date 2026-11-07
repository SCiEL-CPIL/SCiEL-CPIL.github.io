"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(edges):
    """Build adjacency list from edge list."""
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle(adj, used_edges, start=0, deadline=None):
    """
    Find a Hamiltonian cycle in the graph defined by adj,
    avoiding edges in used_edges.
    Returns a list of vertices (cycle) or None.
    """
    n = P['n']
    sys.setrecursionlimit(10000)
    
    def dfs(path, visited, current, edges_used):
        if deadline is not None and time.time() > deadline:
            return None
        if len(path) == n:
            # Check if current can connect back to start
            if (current, start) not in edges_used and (start, current) not in edges_used:
                if start in adj[current]:
                    return list(path)
            return None
        
        # Warnsdorff's heuristic: order neighbors by fewest onward moves
        neighbors = []
        for nb in adj[current]:
            if nb not in visited and (current, nb) not in edges_used and (nb, current) not in edges_used:
                # Count onward moves for this neighbor
                onward = 0
                for nn in adj[nb]:
                    if nn not in visited and nn != current and (nb, nn) not in edges_used and (nn, nb) not in edges_used:
                        if nn == start and len(path) == n - 1:
                            onward += 1
                        elif nn != start:
                            onward += 1
                neighbors.append((onward, nb))
        
        neighbors.sort()
        
        for _, nb in neighbors:
            new_path = path + [nb]
            new_visited = visited | {nb}
            new_edges_used = edges_used | {(current, nb), (nb, current)}
            result = dfs(new_path, new_visited, nb, new_edges_used)
            if result is not None:
                return result
        return None
    
    return dfs([start], {start}, start, set())

def construct():
    deadline = time.time() + 55  # 55 second safety margin
    
    adj = _build_adj(P['edges'])
    used_edges = set()
    cycles = []
    
    # Try to find up to 3 Hamiltonian cycles (max possible)
    for attempt in range(3):
        if time.time() > deadline - 5:
            break
        
        cycle = _find_ham_cycle(adj, used_edges, deadline=deadline)
        if cycle is None:
            break
        
        # Validate the cycle
        n = len(cycle)
        if n != P['n']:
            break
        if len(set(cycle)) != n:
            break
        
        # Check all edges exist in the original graph
        valid = True
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            edge = (min(u, v), max(u, v))
            if edge not in [(min(e[0], e[1]), max(e[0], e[1])) for e in P['edges']]:
                valid = False
                break
            if edge in used_edges:
                valid = False
                break
        if not valid:
            break
        
        # Mark edges as used
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            edge = (min(u, v), max(u, v))
            used_edges.add(edge)
        
        cycles.append([int(v) for v in cycle])
    
    # Ensure we have at least 1 cycle (the seed behavior)
    if not cycles:
        # Fallback: return a single Hamiltonian cycle
        cycle = _find_ham_cycle(adj, set(), deadline=deadline)
        if cycle is not None:
            cycles = [[int(v) for v in cycle]]
        else:
            # Last resort: return a trivial valid cycle if possible
            cycles = [list(range(P['n']))]
    
    # Convert to proper types
    result = []
    for c in cycles:
        result.append(tuple(int(v) for v in c))
    
    return result
# EVOLVE-BLOCK-END
