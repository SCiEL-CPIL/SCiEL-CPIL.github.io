import math
import time

_START_TIME = None

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = _START_TIME + 87.0  # safety margin for 90s limit
    
    # Moduli from 2 to 128
    moduli = list(range(2, 129))
    n_mod = len(moduli)  # 127 moduli
    
    # Precompute gcd for all modulus pairs
    gcd_table = {}
    for i, m1 in enumerate(moduli):
        for j, m2 in enumerate(moduli):
            if i != j:
                gcd_table[(m1, m2)] = _gcd(m1, m2)
    
    # Build conflict graph: vertices are (r, m) for each modulus m and residue r in [0, m)
    # Vertex index: for modulus at position mi, residue r has index offset[mi] + r
    offsets = []
    offset = 0
    for m in moduli:
        offsets.append(offset)
        offset += m
    total_vertices = offset
    
    # Build adjacency as bitsets (Python ints as bitmasks)
    # adj[v] is a bitmask of all vertices that conflict with v
    adj = [0] * total_vertices
    
    # For each pair of vertices, check if they conflict
    # Two vertices (r1, m1) and (r2, m2) conflict if they intersect
    # They intersect iff gcd(m1, m2) | (r1 - r2)
    # i.e., (r1 - r2) % gcd(m1, m2) == 0
    
    # Process by modulus pairs for efficiency
    for i in range(n_mod):
        m1 = moduli[i]
        off1 = offsets[i]
        for j in range(i + 1, n_mod):
            m2 = moduli[j]
            off2 = offsets[j]
            g = gcd_table[(m1, m2)]
            if g == 1:
                # All pairs conflict (always intersect)
                for r1 in range(m1):
                    v1 = off1 + r1
                    for r2 in range(m2):
                        v2 = off2 + r2
                        adj[v1] |= (1 << v2)
                        adj[v2] |= (1 << v1)
            else:
                # Only some pairs conflict
                # (r1 - r2) % g == 0 means r1 ≡ r2 (mod g)
                for r1 in range(m1):
                    v1 = off1 + r1
                    r1_mod_g = r1 % g
                    for r2 in range(m2):
                        if r2 % g == r1_mod_g:
                            v2 = off2 + r2
                            adj[v1] |= (1 << v2)
                            adj[v2] |= (1 << v1)
    
    # Maximum independent set via branch and bound
    # We want the largest set of vertices with no edges between them
    # This is equivalent to maximum clique in the complement graph
    
    # Use a recursive branch-and-bound
    # State: candidates (bitmask of remaining vertices to consider), current set size, best found
    
    best_size = 0
    best_set = set()
    
    # Greedy upper bound: color the induced subgraph, number of colors is upper bound on independence
    # Actually for independence, we need a coloring of the conflict graph
    # The clique number of the conflict graph gives an upper bound on independence number
    
    # Simpler upper bound: greedy coloring of the conflict graph induced by candidates
    def greedy_color_bound(candidates_mask):
        """Upper bound on independent set size via greedy coloring of conflict graph."""
        # Greedy coloring: repeatedly find a maximal independent set, remove it
        # The number of colors needed is an upper bound on the independence number
        # Wait, that's not right. For maximum independent set, we need:
        # The independence number <= number of colors in any proper coloring
        # So we need to color the conflict graph
        
        # Actually, a simpler bound: repeatedly find a maximal independent set
        # and count how many we need. But that's the independence number itself.
        
        # Better: use a greedy upper bound based on the structure
        # For each vertex, at most 1 from each conflict class can be chosen
        # Simple bound: count vertices, subtract forced conflicts
        
        # Let's use a different approach: 
        # Upper bound = min over all vertices v in candidates of (1 + bound(candidates \ N[v]))
        # where N[v] is the closed neighborhood of v
        
        # Actually, let's just use a simple bound: the number of vertices
        # That's too loose. Let me use a greedy coloring approach.
        
        # Greedy coloring of the conflict graph:
        # Assign colors such that adjacent vertices get different colors
        # The number of colors is an upper bound on the independence number
        # Wait no - the independence number is the size of the largest set with NO edges
        # A proper coloring partitions vertices into independent sets
        # So independence number <= max size of any color class
        # That's not helpful directly.
        
        # Let me think differently. For maximum independent set:
        # Upper bound via: for any vertex v, alpha(G) <= 1 + alpha(G - N[v])
        # where N[v] is the closed neighborhood
        # This gives a recursive upper bound but is expensive.
        
        # Simple practical bound: use the fact that in our problem,
        # vertices from coprime moduli all conflict with each other.
        # So if we have k moduli that are pairwise coprime among the candidates,
        # we can pick at most 1 from each "group" of coprime moduli.
        
        # For now, just return the count of vertices as a loose bound
        return bin(candidates_mask).count('1')
    
    # Better approach: use a heuristic upper bound
    # Group vertices by their modulus. Within a modulus, no two vertices conflict
    # (since same modulus means gcd = m, and (r1-r2) % m == 0 only if r1 == r2).
    # Wait, that's wrong. Two vertices with the same modulus m:
    # gcd(m, m) = m, and (r1 - r2) % m == 0 iff r1 == r2.
    # So distinct residues with the same modulus do NOT conflict.
    # But we can only pick one vertex per modulus anyway (distinct moduli requirement).
    
    # So the problem reduces to: pick one vertex from each selected modulus,
    # such that the selected vertices are pairwise non-conflicting.
    
    # This is a constraint satisfaction / optimization problem.
    # Let's use a different formulation.
    
    # Since we can pick at most one residue per modulus, and there are 127 moduli,
    # the maximum possible is 127. But most pairs of moduli are coprime,
    # so we can't pick from both.
    
    # Let me reformulate: we need to select a subset of moduli and assign residues
    # such that for every pair of selected moduli (m_i, m_j), the assigned residues
    # r_i, r_j satisfy (r_i - r_j) % gcd(m_i, m_j) != 0.
    
    # This is a hard combinatorial optimization problem.
    # Let me try a different approach: local search / simulated annealing.
    
    # Start with a greedy solution, then try to improve by swapping residues.
    
    # Greedy initialization:
    # Process moduli in some order, for each modulus try residues in some order
    # and pick the first one that doesn't conflict with already selected.
    
    import random
    random.seed(42)
    
    best_solution = None
    best_count = 0
    
    def evaluate_solution(selected):
        """Check if solution is valid and return count."""
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                r1, m1 = selected[i]
                r2, m2 = selected[j]
                g = gcd_table[(m1, m2)]
                if (r1 - r2) % g == 0:
                    return 0  # invalid
        return len(selected)
    
    # Try multiple random greedy orderings
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Random ordering of moduli
        perm = moduli[:]
        random.shuffle(perm)
        
        selected = []  # list of (residue, modulus)
        
        for m in perm:
            # Try residues in random order
            residues = list(range(m))
            random.shuffle(residues)
            
            found = False
            for r in residues:
                # Check against all selected
                ok = True
                for (sr, sm) in selected:
                    g = gcd_table[(m, sm)]
                    if (r - sr) % g == 0:
                        ok = False
                        break
                if ok:
                    selected.append((r, m))
                    found = True
                    break
        
        count = len(selected)
        if count > best_count:
            best_count = count
            best_solution = selected[:]
        
        if best_count >= 50:
            break
    
    # Local search: try to improve by changing residues
    if best_solution is not None:
        # Try to improve by swapping residues for selected moduli
        for iteration in range(2000):
            if time.time() > deadline:
                break
            
            improved = False
            # Try changing one residue at a time
            for idx in range(len(best_solution)):
                r, m = best_solution[idx]
                # Try all other residues
                for new_r in range(m):
                    if new_r == r:
                        continue
                    # Check if this change is valid
                    ok = True
                    for jdx in range(len(best_solution)):
                        if jdx == idx:
                            continue
                        sr, sm = best_solution[jdx]
                        g = gcd_table[(m, sm)]
                        if (new_r - sr) % g == 0:
                            ok = False
                            break
                    if ok:
                        best_solution[idx] = (new_r, m)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
    
    # Try to add more moduli
    if best_solution is not None:
        selected_mods = set(m for _, m in best_solution)
        for m in moduli:
            if m in selected_mods:
                continue
            if time.time() > deadline:
                break
            for r in range(m):
                ok = True
                for (sr, sm) in best_solution:
                    g = gcd_table[(m, sm)]
                    if (r - sr) % g == 0:
                        ok = False
                        break
                if ok:
                    best_solution.append((r, m))
                    best_count += 1
                    break
    
    if best_solution is None:
        best_solution = [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 128]]
    
    return [[r, m] for r, m in best_solution]
