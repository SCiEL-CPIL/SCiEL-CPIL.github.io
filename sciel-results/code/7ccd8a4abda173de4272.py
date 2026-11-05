import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    n = 20
    vertices = list(range(n))
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Precompute neighbors for each vertex to speed up move generation
    # adj[i] is a bitmask of neighbors of i
    # We will maintain adj as a list of ints
    
    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    def is_valid_fast(adj, changed_u, changed_v):
        """
        Check if adding edge (changed_u, changed_v) creates a K3,3.
        A K3,3 is formed if there exist vertices a, b such that:
        {changed_u, changed_v, a} and {changed_v, changed_u, b} ... wait.
        K3,3 requires two disjoint sets of 3 vertices {u1, u2, u3} and {v1, v2, v3}
        such that all edges between them exist.
        
        If we add edge (u, v), any new K3,3 must include this edge.
        So the edge (u, v) is part of the K3,3.
        This means u and v are in opposite partitions.
        Let Partition A contain u and Partition B contain v.
        We need to find 2 more vertices a, b for A (so A={u, a, b}) 
        and 2 more vertices c, d for B (so B={v, c, d})
        such that all edges between A and B exist.
        
        Edges required:
        (u, v) - Added
        (u, c), (u, d) - u must be connected to c, d
        (a, v), (b, v) - a, b must be connected to v
        (a, c), (a, d) - a must be connected to c, d
        (b, c), (b, d) - b must be connected to c, d
        
        So:
        Candidates for c, d must be in N(u) \ {v}.
        Candidates for a, b must be in N(v) \ {u}.
        
        Let C = N(u) & ~N(v) is not quite right.
        c, d must be neighbors of u.
        a, b must be neighbors of v.
        And c, d must be neighbors of a and b.
        
        Let's iterate over pairs (a, b) from N(v) \ {u}.
        For a fixed (a, b), we need to find 2 vertices c, d in N(u) \ {v}
        such that c, d are in N(a) & N(b).
        
        So:
        CommonNeighborsAB = N(a) & N(b)
        ValidCandidates = N(u) & CommonNeighborsAB & ~(1<<v) & ~(1<<a) & ~(1<<b)
        If bit_count(ValidCandidates) >= 2, then a K3,3 exists.
        
        Note: a and b are distinct. u and v are distinct.
        Also a, b cannot be u or v.
        c, d cannot be u, v, a, b.
        """
        # N(u) without v
        nu = adj[changed_u] & ~(1 << changed_v)
        # N(v) without u
        nv = adj[changed_v] & ~(1 << changed_u)
        
        # Iterate over pairs (a, b) from nv
        # To make it faster, we can just iterate a in nv, b in nv, a < b
        # n is small (20), so nv has at most 19 bits. Pairs ~ 171.
        # For each pair, bit ops are fast.
        
        bits_nv = nv
        while bits_nv:
            low_a = bits_nv & (-bits_nv)
            a = low_a.bit_length() - 1
            bits_nv ^= low_a
            
            # b must be in nv and b > a (or just b != a, we handle pairs)
            # Let's just iterate all b in nv where b != a
            bits_b = nv & ~(1 << a)
            while bits_b:
                low_b = bits_b & (-bits_b)
                b = low_b.bit_length() - 1
                bits_b ^= low_b
                
                # Check if a and b are distinct from u, v
                # They are from nv, which excludes u.
                # They might be v? No, nv excludes u, but does it exclude v?
                # adj[v] does not contain v (simple graph). So v is not in nv.
                # So a, b are distinct from u, v.
                
                # Common neighbors of a and b
                common_ab = adj[a] & adj[b]
                
                # Candidates for c, d must be in N(u) and in common_ab
                # And not u, v, a, b
                candidates = nu & common_ab
                candidates &= ~(1 << a)
                candidates &= ~(1 << b)
                
                if candidates.bit_count() >= 2:
                    return False
        
        return True

    def is_valid_full(adj):
        """
        Full check for K3,3.
        For every triple (u, v, w), check if they have 3 common neighbors.
        Actually, checking every triple of vertices if they have >=3 common neighbors is O(n^4) or O(n^3 * n).
        n=20. 20^3 = 8000 triples. For each, bit_count is fast.
        """
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j+1, n):
                    cijk = cij & adj[k]
                    if cijk.bit_count() >= 3:
                        return False
        return True

    def make_valid(adj):
        """Remove edges until valid."""
        # Use the full check to find violation, then remove an edge involved.
        # Since n is small, we can afford a few iterations.
        for _ in range(1000):
            if is_valid_full(adj):
                return True
            
            # Find a violating triple
            viol_found = False
            for i in range(n):
                ai = adj[i]
                for j in range(i+1, n):
                    cij = ai & adj[j]
                    if cij.bit_count() < 3: continue
                    for k in range(j+1, n):
                        cijk = cij & adj[k]
                        if cijk.bit_count() >= 3:
                            # Remove edge (i, x) where x is a common neighbor
                            x = (cijk & (-cijk)).bit_length() - 1
                            adj[i] &= ~(1 << x)
                            adj[x] &= ~(1 << i)
                            viol_found = True
                            break
                    if viol_found: break
                if viol_found: break
            if not viol_found:
                return True
        
        return is_valid_full(adj)

    def greedy_add(adj):
        added = 0
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Shuffle to avoid bias
            random.shuffle(all_pairs)
            for u, v in all_pairs:
                if adj[u] & (1 << v):
                    continue
                old_u, old_v = adj[u], adj[v]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_valid_fast(adj, u, v):
                    added += 1
                    improved = True
                    break
                else:
                    adj[u] = old_u
                    adj[v] = old_v
        return added

    def simulated_annealing(adj, time_limit):
        # Initial state
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        
        best_adj = current_adj[:]
        best_edges = current_edges
        
        # Temperature
        # Max possible edges is 190.
        # Energy = -edge_count
        # Delta E for removing edge = +1, adding = -1
        # We want to maximize edges.
        
        t_start = time.time()
        # Start with high temp, cool down
        # Time based cooling
        total_time = time_limit - t_start
        if total_time <= 0: return best_adj, best_edges
        
        # Estimate steps per second to set T0
        # Just pick a reasonable T0
        T = 5.0
        cooling_rate = 0.9999
        
        steps = 0
        while time.time() < t_start + total_time:
            # Generate move
            move_type = random.random()
            
            if move_type < 0.5:
                # Add or remove a random edge
                u, v = random.choice(all_pairs)
                if current_adj[u] & (1 << v):
                    # Remove
                    old_u, old_v = current_adj[u], current_adj[v]
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    # Validity is guaranteed if we remove an edge from a valid graph
                    new_edges = current_edges - 1
                    delta_e = 1 # Energy increases (worse)
                    # Always accept removal? No, use SA.
                    # But removing edge always keeps graph valid.
                    # We accept with prob 1 if delta_e < 0 (improvement, which is never for removal)
                    # Or with prob exp(-delta_e/T)
                    if random.random() < math.exp(-delta_e / T):
                        current_edges = new_edges
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = current_adj[:]
                    else:
                        current_adj[u] = old_u
                        current_adj[v] = old_v
                else:
                    # Add
                    old_u, old_v = current_adj[u], current_adj[v]
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    if is_valid_fast(current_adj, u, v):
                        new_edges = current_edges + 1
                        delta_e = -1
                        # Always accept improvement
                        current_edges = new_edges
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = current_adj[:]
                    else:
                        current_adj[u] = old_u
                        current_adj[v] = old_v
            
            elif move_type < 0.8:
                # Swap: remove (u1, v1), add (u2, v2)
                # Pick random existing edge to remove
                # To pick random existing edge, we can pick random u, then random neighbor v
                u1 = random.randint(0, n-1)
                if current_adj[u1] == 0:
                    T *= cooling_rate
                    continue
                v1 = (current_adj[u1] & (-current_adj[u1])).bit_length() - 1
                # Ensure u1 != v1 (should be true)
                
                # Pick random non-edge to add
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 == v2:
                    T *= cooling_rate
                    continue
                if current_adj[u2] & (1 << v2):
                    T *= cooling_rate
                    continue
                
                # Ensure (u1, v1) != (u2, v2)
                if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
                    T *= cooling_rate
                    continue
                
                # Apply move
                old_u1, old_v1, old_u2, old_v2 = current_adj[u1], current_adj[v1], current_adj[u2], current_adj[v2]
                
                # Remove (u1, v1)
                current_adj[u1] &= ~(1 << v1)
                current_adj[v1] &= ~(1 << u1)
                # Add (u2, v2)
                current_adj[u2] |= (1 << v2)
                current_adj[v2] |= (1 << u2)
                
                # Check validity
                # The graph changed at u1, v1, u2, v2.
                # We need to check if the new graph is valid.
                # Since we removed an edge, we only need to check if adding (u2, v2) creates a K3,3
                # in the context of the graph where (u1, v1) is removed.
                # is_valid_fast checks if adding (u2, v2) to the CURRENT state creates a K3,3.
                # The current state has (u1, v1) removed.
                # So this is correct.
                
                if is_valid_fast(current_adj, u2, v2):
                    # Valid
                    # Edge count change: -1 (removal) + 1 (addition) = 0
                    # So current_edges stays same
                    # We accept with prob 1? Or SA?
                    # Since edge count is same, it's a neutral move.
                    # Accept to explore.
                    # If we want to be strict, we could accept with prob 1.
                    pass
                else:
                    # Invalid, revert
                    current_adj[u1] = old_u1
                    current_adj[v1] = old_v1
                    current_adj[u2] = old_u2
                    current_adj[v2] = old_v2
                
            elif move_type < 1.0:
                # 2-Opt style: Remove 2 edges, Add 2 edges
                # Too complex for now, stick to single swap
                pass

            T *= cooling_rate
            if T < 0.01:
                T = 0.01
            
            steps += 1
            # Periodic greedy add?
            if steps % 1000 == 0:
                greedy_add(current_adj)
                current_edges = count_edges(current_adj)
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = current_adj[:]

        return best_adj, best_edges

    best_adj = [0] * n
    best_count = 0
    best_edges = []
    
    def update_best(adj):
        nonlocal best_adj, best_count, best_edges
        c = count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]
            best_edges = adj_to_edges(adj)

    random.seed(42)
    
    # Strategy:
    # 1. Start with a dense random graph, make it valid.
    # 2. Greedy add.
    # 3. Simulated Annealing.
    
    # We can try multiple restarts if time permits.
    # But SA is slow. Let's do 1 long SA run or a few shorter ones.
    
    # Initial graph
    # Try a few different starting densities
    for p in [0.7, 0.6, 0.5]:
        if time.time() > deadline - 20:
            break
        
        adj = [0] * n
        for i, j in all_pairs:
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        if not make_valid(adj):
            continue
        
        greedy_add(adj)
        update_best(adj)
        
        # SA
        remaining = deadline - 15 - time.time()
        if remaining > 5:
            # Use most of the remaining time for SA
            sa_time = min(remaining, 30) # Cap SA time per restart?
            # Actually, let's just use all remaining time for the first good start
            # or split it.
            # Let's do 1 SA run with the best start found so far.
            pass
        
    # Let's refine: 
    # 1. Generate a few valid seeds.
    # 2. Pick the best one.
    # 3. Run SA on it for the remaining time.
    
    candidates = []
    for p in [0.7, 0.6, 0.5, 0.8]:
        if time.time() > deadline - 60:
            break
        adj = [0] * n
        for i, j in all_pairs:
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        if not make_valid(adj):
            continue
        
        greedy_add(adj)
        c = count_edges(adj)
        candidates.append((c, adj))
    
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_start = candidates[0][1]
        update_best(best_start)
        
        # Run SA on best_start
        remaining = deadline - 10 - time.time()
        if remaining > 0:
            sa_adj, sa_count = simulated_annealing(best_start, remaining)
            update_best(sa_adj)
    
    # Fallback
    if not best_edges:
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if (j - i) % 2 == 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if not is_valid_full(adj):
            return [[i, i+1] for i in range(n-1)]
        return adj_to_edges(adj)
        
    return best_edges

# EVOLVE-BLOCK-END
