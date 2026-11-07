"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    deadline = time.time() + 85.0
    n = P['n']
    
    # Build adjacency set for O(1) edge lookup
    adj = set()
    for u, v in P['edges']:
        adj.add((min(u, v), max(u, v)))
        adj.add((max(u, v), min(u, v)))
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def make_jump_cycle(step):
        """Create a Hamiltonian cycle using step jumps mod n."""
        if gcd(step, n) != 1:
            return None
        cycle = []
        cur = 0
        for _ in range(n):
            cycle.append(cur)
            cur = (cur + step) % n
        # Verify all edges exist in the host graph
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            if (min(u, v), max(u, v)) not in adj:
                return None
        return cycle
    
    def get_cycle_edges(cycle):
        """Get set of edges (as frozensets) used by a cycle."""
        edges = set()
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edges.add((min(u, v), max(u, v)))
        return edges
    
    def verify_cycles(cycles):
        """Verify all cycles are valid and edge-disjoint."""
        used_edges = set()
        for cyc in cycles:
            if len(cyc) != n or len(set(cyc)) != n:
                return False
            for i in range(n):
                u = cyc[i]
                v = cyc[(i + 1) % n]
                ek = (min(u, v), max(u, v))
                if ek not in adj:
                    return False
                if ek in used_edges:
                    return False
                used_edges.add(ek)
        return True
    
    # Find all valid jump steps
    valid_steps = []
    for s in range(1, n):
        if gcd(s, n) == 1:
            c = make_jump_cycle(s)
            if c is not None:
                valid_steps.append(s)
    
    # Pre-compute cycles and their edges
    step_cycles = {}
    step_edges = {}
    for s in valid_steps:
        c = make_jump_cycle(s)
        if c is not None:
            step_cycles[s] = c
            step_edges[s] = get_cycle_edges(c)
    
    # Try all orderings of valid steps (backtracking search)
    # Since there are few valid steps, we can try permutations
    best_result = []
    
    # First, quick greedy in natural order
    used_edges = set()
    cycles = []
    for s in valid_steps:
        c = step_cycles[s]
        edges = step_edges[s]
        if edges.isdisjoint(used_edges):
            used_edges.update(edges)
            cycles.append(c)
    if cycles:
        best_result = cycles
    
    # Try to improve with backtracking over permutations
    if len(valid_steps) <= 8 and time.time() < deadline - 5.0:
        found_better = False
        for perm in itertools.permutations(valid_steps):
            if time.time() > deadline - 5.0:
                break
            used_e = set()
            cyc_list = []
            for s in perm:
                edges = step_edges[s]
                if edges.isdisjoint(used_e):
                    used_e.update(edges)
                    cyc_list.append(step_cycles[s])
            if len(cyc_list) > len(best_result):
                best_result = cyc_list
                found_better = True
                if len(best_result) >= 3:  # Theoretical max
                    break
    
    # Also try: generate additional non-jump Hamiltonian cycles
    # and do a combined search
    if time.time() < deadline - 10.0 and len(best_result) < 3:
        # Try to find additional cycles via different constructions
        # Use step cycles with different starting points (rotation doesn't change edges)
        # Try cycles built from different base patterns
        extra_cycles = []
        # Try all possible step-based cycles with different offsets
        # A cycle starting at vertex k with step s: k, k+s, k+2s, ...
        # But this is just a rotation of the same cycle, so same edges
        # Instead, try non-uniform patterns
        # For now, stick with what we have
        
    # Final verification
    if best_result and verify_cycles(best_result):
        return [[int(v) for v in c] for c in best_result]
    
    # Fallback: return a single known valid cycle
    fallback = list(range(n))
    for i in range(n):
        u = fallback[i]
        v = fallback[(i + 1) % n]
        if (min(u, v), max(u, v)) not in adj:
            return [[0, 1, 2]]
    return [fallback]
# EVOLVE-BLOCK-END
