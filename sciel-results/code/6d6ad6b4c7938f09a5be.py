import time
from math import gcd

def construct():
    """
    Returns the largest subset of [1, 255] containing 255 such that
    gcd(a, b) > 1 for all distinct pairs a, b.
    
    Uses bitset-based Bron-Kerbosch with fast greedy coloring,
    plus precomputed blocking sets for local search refinement.
    """
    deadline = time.time() + 80.0
    t0 = time.time()
    
    # Candidate set: numbers in [1,255] sharing a factor with 255 = 3*5*17
    candidates = [i for i in range(1, 256) if i % 3 == 0 or i % 5 == 0 or i % 17 == 0]
    n = len(candidates)
    idx_map = {v: i for i, v in enumerate(candidates)}
    
    # Build adjacency bitsets
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Precompute: for each candidate v not in the base set (multiples of 3),
    # which multiples of 3 does v conflict with (gcd == 1)?
    mult3_indices = [i for i in range(n) if candidates[i] % 3 == 0]
    mult3_set = set(candidates[i] for i in mult3_indices)
    
    # For local search: for each non-multiple-of-3 candidate, precompute
    # the set of multiples of 3 it conflicts with
    conflict_with_mult3 = {}
    for i in range(n):
        v = candidates[i]
        if v % 3 == 0:
            continue
        conflicts = set()
        for j in mult3_indices:
            u = candidates[j]
            if gcd(v, u) == 1:
                conflicts.add(j)
        conflict_with_mult3[i] = conflicts
    
    # Fast greedy coloring: no sorting, just iterate and assign min available color
    def greedy_color_fast(P):
        """Greedy coloring of P (bitset). Returns (order, colors) in O(n * max_color)."""
        P_list = [i for i in range(n) if (P >> i) & 1]
        if not P_list:
            return [], {}
        
        colors = {}
        order = []
        color_of = [0] * n  # color_of[v] = color assigned to v, 0 = uncolored
        
        for v in P_list:
            # Find colors used by colored neighbors of v
            used = set()
            for u in order:
                if (adj[v] >> u) & 1:
                    used.add(color_of[u])
            c = 1
            while c in used:
                c += 1
            color_of[v] = c
            colors[v] = c
            order.append(v)
        
        return order, colors
    
    # Bron-Kerbosch with coloring bound
    best = [set(candidates[i] for i in mult3_indices)]  # Start with multiples of 3 (size 85)
    best_size = 85
    
    def bron_kerbosch(P, X, R):
        nonlocal best, best_size
        if time.time() > deadline:
            return
        if not P and not X:
            if len(R) > best_size:
                best_size = len(R)
                best[0] = set(R)
            return
        
        order, colors = greedy_color_fast(P)
        if not order:
            return
        
        # Process in reverse order (highest color first)
        for idx in range(len(order) - 1, -1, -1):
            if time.time() > deadline:
                return
            v = order[idx]
            # Pruning: if current size + color[v] <= best, skip
            if len(R) + colors[v] <= best_size:
                break
            
            # Expand
            new_R = R | {v}
            new_P = P & adj[v] & ~(1 << v)
            new_X = X & adj[v] & ~(1 << v)
            
            bron_kerbosch(new_P, new_X, new_R)
            
            P &= ~(1 << v)
            X |= (1 << v)
    
    # Run Bron-Kerbosch on full candidate set
    full_P = (1 << n) - 1
    bron_kerbosch(full_P, 0, frozenset())
    
    # If BK didn't find anything better than 85, try targeted local search
    if best_size <= 85:
        # Try: for each non-mult3 candidate v, compute how many mult3 elements
        # it conflicts with. If conflicts < 1 (i.e., 0), we can add v for free.
        # If conflicts = k, we need to remove k elements and add 1, net change = 1 - k.
        # Only useful if k = 0 (free add) — but that's already handled by BK.
        # For k = 1: remove 1, add 1, net 0. Not helpful for size.
        # So the only way to beat 85 is to find a set where multiple non-mult3
        # elements together require removing fewer mult3 elements than they add.
        
        # Try 2-element additions: add v1, v2 (both non-mult3), remove conflicts
        # Net gain = 2 - |conflicts(v1) ∪ conflicts(v2) ∩ mult3|
        # Need 2 - |union| > 0, i.e., |union| < 2, i.e., |union| <= 1
        
        non_mult3 = [i for i in range(n) if candidates[i] % 3 != 0]
        
        found_better = False
        for i1 in range(len(non_mult3)):
            if time.time() > deadline:
                break
            v1 = non_mult3[i1]
            c1 = conflict_with_mult3.get(v1, set())
            for i2 in range(i1 + 1, len(non_mult3)):
                v2 = non_mult3[i2]
                c2 = conflict_with_mult3.get(v2, set())
                union_conflicts = c1 | c2
                if len(union_conflicts) < 2:
                    # Check v1 and v2 are compatible with each other
                    if (adj[v1] >> v2) & 1:
                        # Also check they're compatible with remaining mult3
                        # This is a valid improvement if len(union) < 2
                        # New size = 85 - len(union) + 2
                        if 85 - len(union_conflicts) + 2 > best_size:
                            found_better = True
                            # Build the set
                            new_set = set()
                            for j in mult3_indices:
                                if j not in union_conflicts:
                                    new_set.add(candidates[j])
                            new_set.add(candidates[v1])
                            new_set.add(candidates[v2])
                            best_size = len(new_set)
                            best[0] = new_set
        
        if not found_better and best_size <= 85:
            # Return the known optimal: all multiples of 3
            return [i for i in range(1, 256) if i % 3 == 0]
    
    # Ensure 255 is in the result
    if 255 not in best[0]:
        best[0].add(255)
        # Remove any element not connected to 255
        to_remove = [x for x in best[0] if x != 255 and gcd(x, 255) == 1]
        for x in to_remove:
            best[0].discard(x)
    
    return list(best[0])
