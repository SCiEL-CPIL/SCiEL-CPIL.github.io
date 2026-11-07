"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _verify_cycles(cycles, edges):
    """Verify all cycles are valid and edge-disjoint."""
    n = P['n']
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
    
    used_edges = set()
    for cyc in cycles:
        if len(cyc) < 3:
            return False
        # Check all vertices distinct
        if len(set(cyc)) != len(cyc):
            return False
        # Check all vertices are in range
        for v in cyc:
            if not isinstance(v, int) or isinstance(v, bool) or v < 0 or v >= n:
                return False
        # Check consecutive edges exist in host
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i + 1) % len(cyc)]
            key = (min(a, b), max(a, b))
            if key not in edge_set:
                return False
            if key in used_edges:
                return False
            used_edges.add(key)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 60.0  # safety margin before 90s kill
    
    # The graph has 63 edges. Each Hamiltonian cycle on 21 vertices uses 21 edges.
    # Maximum possible: floor(63/21) = 3. These three cycles achieve the bound.
    # This is a complete edge-decomposition into 3 Hamiltonian cycles.
    cycle1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    cycle2 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    cycle3 = [0, 5, 10, 15, 20, 4, 9, 14, 19, 3, 8, 13, 18, 2, 7, 12, 17, 1, 6, 11, 16]
    
    result = [cycle1, cycle2, cycle3]
    
    # Verify before returning
    if _verify_cycles(result, P['edges']):
        # Ensure types are correct (plain Python int, list)
        verified = []
        for cyc in result:
            verified.append([v for v in cyc if isinstance(v, int) and not isinstance(v, bool)])
        return verified
    
    # Fallback: return what we have (should not happen)
    return result
# EVOLVE-BLOCK-END
