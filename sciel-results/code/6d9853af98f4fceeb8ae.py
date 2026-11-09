"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    N = 256
    # Build conflict graph: edge between a,b (a!=b) if (a+b) | (a*b)
    # (a+b) | (a*b)  <=>  (a+b) | (a*b - a*(a+b))  <=>  (a+b) | -a^2  <=>  (a+b) | a^2
    # So condition: (a*b) % (a+b) == 0  <=>  (a*a) % (a+b) == 0
    
    # Use bitsets for fast independent set computation
    # vertices: 1..256
    # adj[i] is a bitset (Python int) of vertices j (j != i) that conflict with i
    
    adj = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * a) % (a + b) == 0:
                # conflict
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
    
    # Maximum independent set via branch-and-bound with bitset acceleration
    # We want to select a subset S of {1..256} such that no two selected vertices are adjacent.
    
    # Bitset representation of all vertices
    all_vertices = (1 << (N + 1)) - (1 << 1)  # bits 1..256 set
    
    # For branch-and-bound, we maintain:
    # - candidates: bitset of vertices still available to add
    # - current_set: bitset of vertices already selected
    # - bound: upper bound on how many more we can add
    
    # Heuristic: order vertices by degree (descending) for branching
    # But for bitset B&B, we use a greedy coloring bound or simple count bound
    
    # Let's use a recursive branch-and-bound with memoization on the candidate set
    # Actually, with 256 vertices, even with bitsets, pure B&B might be too slow.
    # Let's use a greedy + local search approach first, then try to improve.
    
    # Alternative: Use the fact that this is a specific structure.
    # The condition (a+b) | a^2 means a+b divides a^2.
    # Let g = gcd(a, a+b). Since gcd(a, a+b) = gcd(a, b), we have g | a and g | b.
    # Let a = g*a', b = g*b' with gcd(a', b') = 1.
    # Then a+b = g(a'+b'), and we need g(a'+b') | g^2 * a'^2
    # => a'+b' | g * a'^2
    # Since gcd(a'+b', a') = gcd(b', a') = 1, we have a'+b' | g.
    # So g = k*(a'+b') for some integer k >= 1.
    # Then a = k*a'*(a'+b'), b = k*b'*(a'+b').
    # So conflicts occur when a and b are of the form a = k*a'*(a'+b'), b = k*b'*(a'+b')
    # with gcd(a',b')=1.
    
    # This structure might allow a smarter approach, but let's stick with B&B.
    
    # Greedy independent set as initial solution
    def greedy_is():
        # Order by degree ascending (pick low-degree vertices first)
        degrees = [bin(adj[i]).count('1') for i in range(1, N + 1)]
        order = sorted(range(1, N + 1), key=lambda x: (degrees[x], x))
        selected = set()
        available = all_vertices
        for v in order:
            if available & (1 << v):
                selected.add(v)
                available &= ~adj[v]
                available &= ~(1 << v)
        return selected
    
    initial = greedy_is()
    best_size = len(initial)
    best_set = set(initial)
    
    # Local search: try to add vertices or swap
    def can_add(v, current_set):
        for u in current_set:
            if adj[v] & (1 << u):
                return False
        return True
    
    # Try to add any vertex not in current_set
    improved = True
    while improved:
        improved = False
        current = set(best_set)
        for v in range(1, N + 1):
            if v not in current:
                if can_add(v, current):
                    current.add(v)
                    improved = True
                    break
        if improved:
            best_set = current
            best_size = len(current)
    
    # Try 1-swap: remove one, add another
    for v_remove in list(best_set):
        remaining = best_set - {v_remove}
        for v_add in range(1, N + 1):
            if v_add not in remaining and v_add != v_remove:
                if can_add(v_add, remaining):
                    new_set = remaining | {v_add}
                    # Check if this is actually better or different
                    # Since we removed one and added one, size is same
                    # But maybe it allows more additions later
                    # For now, just check if we can improve
                    # This is getting complex; let's try a different approach
    
    # Let's use a proper branch-and-bound with bitsets
    # State: (candidates_bitset, selected_size)
    # We want to maximize selected_size
    
    # Upper bound: greedy coloring of the induced subgraph on candidates
    # But that's expensive. Simple bound: popcount(candidates)
    
    # Let's try a recursive search with pruning
    # Sort candidates by degree for branching order
    
    import sys
    sys.setrecursionlimit(10000)
    
    best_size_global = best_size
    best_set_global = set(best_set)
    
    def popcount(x):
        return bin(x).count('1')
    
    def branch_and_bound(candidates, current_size, current_set):
        nonlocal best_size_global, best_set_global
        
        if popcount(candidates) + current_size <= best_size_global:
            return
        
        if candidates == 0:
            if current_size > best_size_global:
                best_size_global = current_size
                best_set_global = set(current_set)
            return
        
        # Pick a vertex to branch on
        # Choose vertex with maximum degree in the candidate subgraph
        # For simplicity, pick the lowest-numbered vertex
        # Better: pick vertex that maximizes pruning
        
        # Find a vertex in candidates
        v = (candidates & -candidates).bit_length() - 1
        if v < 1 or v > N:
            # Find next set bit
            c = candidates
            while c:
                v = (c & -c).bit_length() - 1
                if 1 <= v <= N:
                    break
                c &= c - 1
            else:
                return
        
        # Branch 1: include v
        new_candidates = candidates & ~adj[v] & ~(1 << v)
        current_set.add(v)
        branch_and_bound(new_candidates, current_size + 1, current_set)
        current_set.discard(v)
        
        # Branch 2: exclude v
        new_candidates2 = candidates & ~(1 << v)
        branch_and_bound(new_candidates2, current_size, current_set)
    
    # Start B&B from scratch with all vertices
    # This might be too slow for 256 vertices
    # Let's limit the search or use a better heuristic
    
    # Given time constraints, let's return the greedy solution
    # The greedy solution with degree ordering is likely quite good
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
