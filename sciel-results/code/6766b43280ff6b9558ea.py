import time
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

N = P['n']
# Precompute adjacency lists
ADJ = [[] for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].append(v)
    ADJ[v].append(u)

# Precompute edge keys for fast lookup
EDGE_SET = set()
for u, v in P['edges']:
    if u < v:
        EDGE_SET.add((u, v))
    else:
        EDGE_SET.add((v, u))

def _edge_key(u, v):
    return (u, v) if u < v else (v, u)

def _find_hamiltonian_cycle(remaining_edges, deadline):
    """
    Finds a Hamiltonian cycle in the graph defined by remaining_edges.
    Uses Warnsdorff's rule with backtracking.
    """
    n = N
    # Build local adjacency from remaining edges
    adj = [set() for _ in range(n)]
    for u, v in remaining_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check degree constraints
    for i in range(n):
        if len(adj[i]) < 2:
            return None

    # Start from vertex 0 to reduce symmetry
    start = 0
    path = [start]
    visited = {start}
    
    # Warnsdorff's rule: prioritize vertices with fewer onward moves
    def backtrack():
        if time.time() > deadline:
            return None
        
        if len(path) == n:
            # Check if we can close the cycle
            if start in adj[path[-1]]:
                return path[:]
            return None
        
        current = path[-1]
        candidates = []
        for nb in adj[current]:
            if nb not in visited:
                # Count onward moves from nb
                onward = sum(1 for x in adj[nb] if x not in visited and x != current)
                candidates.append((onward, nb))
        
        if not candidates:
            return None
        
        # Sort by onward moves (Warnsdorff)
        candidates.sort(key=lambda x: x[0])
        
        for _, nb in candidates:
            path.append(nb)
            visited.add(nb)
            result = backtrack()
            if result is not None:
                return result
            path.pop()
            visited.discard(nb)
            
        return None

    return backtrack()

def _verify_packing(cycles):
    """Verify that the packing is valid."""
    if not cycles:
        return False
    
    used_edges = set()
    for cycle in cycles:
        if len(cycle) != N:
            return False
        if len(set(cycle)) != N:
            return False
        for i in range(N):
            u = cycle[i]
            v = cycle[(i + 1) % N]
            ek = _edge_key(u, v)
            if ek not in EDGE_SET:
                return False
            if ek in used_edges:
                return False
            used_edges.add(ek)
            
    # Check that all cycles use the same vertex set (which is range(N))
    # This is implicitly true if len(set(cycle))==N and vertices are 0..N-1
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    remaining = set(EDGE_SET)
    cycles = []
    
    # Try to find 3 cycles
    for _ in range(3):
        if time.time() > deadline:
            break
            
        # Find a Hamiltonian cycle in the remaining graph
        cycle = _find_hamiltonian_cycle(remaining, deadline)
        
        if cycle is None:
            break
        
        # Verify the cycle uses only remaining edges
        valid = True
        for i in range(N):
            u = cycle[i]
            v = cycle[(i + 1) % N]
            ek = _edge_key(u, v)
            if ek not in remaining:
                valid = False
                break
        
        if not valid:
            break
            
        cycles.append(cycle)
        
        # Remove used edges
        for i in range(N):
            u = cycle[i]
            v = cycle[(i + 1) % N]
            remaining.discard(_edge_key(u, v))
            
        if len(cycles) == 3:
            break
            
    # Final verification
    if cycles and _verify_packing(cycles):
        return [list(c) for c in cycles]
    
    # Fallback: If we didn't find 3, try to find at least 1 or 2
    # Since the graph is 6-regular, 3 should be possible.
    # If we failed, return the best valid subset.
    if cycles:
        # Check if the current cycles are valid
        if _verify_packing(cycles):
             return [list(c) for c in cycles]
             
    # Last resort: Return a single valid cycle if possible
    remaining = set(EDGE_SET)
    cycle = _find_hamiltonian_cycle(remaining, deadline)
    if cycle and _verify_packing([cycle]):
        return [list(cycle)]
        
    return []

# EVOLVE-BLOCK-END
