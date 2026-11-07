import time

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _verify_packing(cycles, all_edges_set):
    """
    Verify that all cycles are valid, edge-disjoint, and use the same vertex set.
    Returns True if valid, False otherwise.
    """
    if not cycles:
        return False
    
    n = P['n']
    used_edges = set()
    target_vertex_set = None
    
    for cycle in cycles:
        # Check it's a list or tuple
        if not isinstance(cycle, (list, tuple)):
            return False
        if len(cycle) < 3:
            return False
        
        # Check all vertices are valid Python ints (not bool, not numpy int)
        for v in cycle:
            if not isinstance(v, int) or isinstance(v, bool):
                return False
            if v < 0 or v >= n:
                return False
        
        # Check distinct vertices
        if len(set(cycle)) != len(cycle):
            return False
        
        # Check vertex set consistency
        current_vset = set(cycle)
        if target_vertex_set is None:
            target_vertex_set = current_vset
        else:
            if current_vset != target_vertex_set:
                return False
        
        # Check edges exist and are disjoint
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i+1) % len(cycle)]
            ek = _edge_key(u, v)
            if ek not in all_edges_set:
                return False
            if ek in used_edges:
                return False
            used_edges.add(ek)
    
    return True

def construct():
    n = P['n']
    all_edges = P['edges']
    all_edges_set = set()
    for u, v in all_edges:
        all_edges_set.add(_edge_key(u, v))
    
    # The graph is a circulant graph C_21(1, 2, 5).
    # It is 6-regular with 63 edges.
    # Max edge-disjoint Hamiltonian cycles = 63 / 21 = 3.
    # We construct 3 Hamiltonian cycles using steps 1, 5, and 2 (or 19).
    # Step 1: edges (i, i+1)
    # Step 5: edges (i, i+5)
    # Step 2: edges (i, i+2) -> equivalent to step 19 for cycle generation order if we want specific direction, 
    # but any step s with gcd(s, 21)=1 works. 
    # The sets of edges used by steps 1, 2, 5 are disjoint because the offsets {1,20}, {2,19}, {5,16} are distinct.
    
    def make_cycle(step):
        # Generate vertices [0, step, 2*step, ...] mod n
        # This creates a cycle 0 -> step -> 2*step -> ... -> 0
        return [ (i * step) % n for i in range(n) ]
    
    # Construct the 3 cycles
    # Steps 1, 2, 5 are pairwise coprime to 21 and their edge sets are disjoint.
    cycles = [
        make_cycle(1),
        make_cycle(2),
        make_cycle(5)
    ]
    
    # Ensure types are strictly Python int
    validated_cycles = []
    for c in cycles:
        validated_cycles.append([int(v) for v in c])
    
    # Verify the packing before returning
    if _verify_packing(validated_cycles, all_edges_set):
        return validated_cycles
    
    # Fallback: If verification fails (should not happen for this specific graph), 
    # return an empty list to avoid invalid output, though this branch is unreachable 
    # for the given P.
    return []

# EVOLVE-BLOCK-END
