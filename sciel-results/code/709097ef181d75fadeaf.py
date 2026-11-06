import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # We'll try to find a valid graph on n vertices
    # Start with n=35 (the known record) and try to find a valid one
    # If we can't find one quickly, try smaller n
    
    best_result = None
    best_n = 0
    
    # Try n from 35 down to 20
    for n in range(35, 19, -1):
        if time.time() > deadline - 5.0:
            break
        
        result = _anneal(n, deadline)
        if result is not None:
            if result['n'] > best_n:
                best_n = result['n']
                best_result = result
            break  # Found valid graph for this n, return it
    
    if best_result is None:
        # Fallback: return a small valid graph
        # Paley graph on 17 vertices
        squares = {x * x % 17 for x in range(1, 17)}
        return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
    
    return best_result


def _anneal(n, deadline):
    """Simulated annealing to find K4-free graph with no independent 6-set on n vertices."""
    
    # Represent graph as adjacency bitmask: adj[i] is a bitmask of neighbors of vertex i
    adj = [0] * n
    
    # Initialize: random graph with moderate edge density
    # For K4-free, we want relatively sparse; for no indep-6, we want relatively dense
    # A good starting point might be around 50% density
    p = 0.5
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Track best valid solution found
    best_adj = None
    best_n_found = 0
    
    # Count violations efficiently
    def count_violations(adj):
        """Count number of K4s and independent 6-sets (or just check existence)."""
        # Check for K4
        k4_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    for k in range(j + 1, n):
                        if (adj[i] & (1 << k)) and (adj[j] & (1 << k)):
                            for l in range(k + 1, n):
                                if (adj[i] & (1 << l)) and (adj[j] & (1 << l)) and (adj[k] & (1 << l)):
                                    k4_count += 1
                                    if k4_count > 0:
                                        return k4_count, 0  # Early exit for K4
        # Check for independent 6-set
        # An independent set of size 6 means 6 vertices with no edges between them
        # Equivalently, in the complement graph, a clique of size 6
        # For each subset of 6 vertices, check if they form an independent set
        # This is expensive, so use a heuristic: check if any 6 vertices are mutually non-adjacent
        indep6_count = 0
        # Use a smarter approach: for each vertex, look at its non-neighbors
        # If any vertex has 5+ non-neighbors that are also mutually non-adjacent, we have an indep 6
        for i in range(n):
            non_neighbors = []
            for j in range(n):
                if j != i and not (adj[i] & (1 << j)):
                    non_neighbors.append(j)
            if len(non_neighbors) < 5:
                continue
            # Check if any 5 of these non-neighbors form an independent set with i
            # i.e., among non_neighbors, find 5 vertices that are mutually non-adjacent
            # This is still expensive. Use a simpler check:
            # Just check if there exist 6 vertices that are all mutually non-adjacent
            pass
        
        # Better approach for independent set: use the complement
        # Build complement adjacency
        comp_adj = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and not (adj[i] & (1 << j)):
                    comp_adj[i] |= (1 << j)
        
        # Now look for a clique of size 6 in the complement
        # Use a simple backtracking or just check all 6-subsets (expensive for large n)
        # For n=35, C(35,6) is about 1.6 million, which is feasible if we do it once
        # But we need to do this repeatedly during annealing, so we need something faster
        
        # Alternative: use a greedy/approximate check
        # For the purpose of annealing, we can use a probabilistic check
        # Sample random 6-sets and check if they're independent
        # Or: for each vertex, count how many non-neighbor 5-cliques exist in its non-neighborhood
        
        # Let's use a different strategy: track the "independence number" approximately
        # Actually, let's just do a full check but optimize it
        # For n <= 35, we can enumerate 6-subsets using combinations
        from itertools import combinations
        for combo in combinations(range(n), 6):
            is_independent = True
            for idx in range(6):
                for idx2 in range(idx + 1, 6):
                    u = combo[idx]
                    v = combo[idx2]
                    if adj[u] & (1 << v):
                        is_independent = False
                        break
                if not is_independent:
                    break
            if is_independent:
                indep6_count += 1
                if indep6_count > 0:
                    return k4_count, indep6_count
        
        return k4_count, indep6_count
    
    # For speed during annealing, use a faster violation metric
    # Instead of counting all violations, just check if ANY exist
    def has_violation(adj):
        """Return True if graph has K4 or independent 6-set."""
        # Check K4
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    for k in range(j + 1, n):
                        if (adj[i] & (1 << k)) and (adj[j] & (1 << k)):
                            for l in range(k + 1, n):
                                if (adj[i] & (1 << l)) and (adj[j] & (1 << l)) and (adj[k] & (1 << l)):
                                    return True
        # Check independent 6-set using complement clique detection
        # Build complement
        comp_adj = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and not (adj[i] & (1 << j)):
                    comp_adj[i] |= (1 << j)
        
        # Look for clique of size 6 in complement using backtracking
        def find_clique(size, start):
            if size == 0:
                return True
            if start >= n:
                return False
            # Try adding vertex start
            # Check if start can be added to current clique
            # We need to pass the current clique, so use a different approach
            pass
        
        # Simpler: enumerate 6-subsets in complement and check for clique
        from itertools import combinations
        for combo in combinations(range(n), 6):
            is_clique = True
            for idx in range(6):
                for idx2 in range(idx + 1, 6):
                    u = combo[idx]
                    v = combo[idx2]
                    if not (comp_adj[u] & (1 << v)):
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                return True
        return False
    
    # Even faster: use a probabilistic violation check for the annealing loop
    # Only do exact check periodically
    def quick_violation_check(adj, sample_size=20):
        """Quick probabilistic check for violations."""
        # Check for K4: sample random 4-tuples
        for _ in range(sample_size):
            quad = random.sample(range(n), 4)
            is_clique = True
            for i in range(4):
                for j in range(i + 1, 4):
                    if not (adj[quad[i]] & (1 << quad[j])):
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                return True
        
        # Check for independent 6-set: sample random 6-tuples
        for _ in range(sample_size):
            sext = random.sample(range(n), 6)
            is_independent = True
            for i in range(6):
                for j in range(i + 1, 6):
                    if adj[sext[i]] & (1 << sext[j]):
                        is_independent = False
                        break
                if not is_independent:
                    break
            if is_independent:
                return True
        
        return False
    
    # Run simulated annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    steps_per_temp = 50
    
    current_adj = adj[:]
    current_violations = 1 if quick_violation_check(current_adj) else 0
    
    # Exact initial check
    if not has_violation(current_adj):
        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if current_adj[i] & (1 << j)]
        return {'n': n, 'edges': edges}
    
    for step in range(100000):
        if time.time() > deadline - 2.0:
            break
        
        # Choose a random edge to flip
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        # Compute new violations
        # Flip edge (i, j)
        old_state = bool(current_adj[i] & (1 << j))
        
        # Tentatively flip
        if old_state:
            current_adj[i] &= ~(1 << j)
            current_adj[j] &= ~(1 << i)
        else:
            current_adj[i] |= (1 << j)
            current_adj[j] |= (1 << i)
        
        new_violations = 1 if quick_violation_check(current_adj) else 0
        
        # Accept or reject
        if new_violations <= current_violations:
            current_violations = new_violations
        else:
            # Reject with some probability
            if random.random() > 0.1:
                # Undo flip
                if old_state:
                    current_adj[i] |= (1 << j)
                    current_adj[j] |= (1 << i)
                else:
                    current_adj[i] &= ~(1 << j)
                    current_adj[j] &= ~(1 << i)
        
        # Periodic exact check
        if step % 100 == 0:
            if not has_violation(current_adj):
                edges = [[i, j] for i in range(n) for j in range(i + 1, n) if current_adj[i] & (1 << j)]
                return {'n': n, 'edges': edges}
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final exact check
    if not has_violation(current_adj):
        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if current_adj[i] & (1 << j)]
        return {'n': n, 'edges': edges}
    
    return None
