import time
import random
import sys

# EVOLVE-BLOCK-START
_DEADLINE = 118.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() >= _DEADLINE

def _build_polarity_graph(q):
    """Construct the polarity graph of the projective plane PG(2, q)."""
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, q-2, q)
                    key = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q-2, q)
                    key = (0, 1, (z * inv_y) % q)
                else:
                    key = (0, 0, 1)
                
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    n_pts = len(points)
    adj = [set() for _ in range(n_pts)]
    
    for i in range(n_pts):
        x1, x2, x3 = points[i]
        for j in range(i + 1, n_pts):
            y1, y2, y3 = points[j]
            if (x1 * y1 + x2 * y2 + x3 * y3) % q == 0:
                adj[i].add(j)
                adj[j].add(i)
                
    return adj

def _has_indep_11(adj):
    """Check if there is an independent set of size 11 using bitmask backtracking."""
    n = len(adj)
    if n < 11:
        return False
        
    masks = []
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        masks.append(m)
    
    sys.setrecursionlimit(10000)
    
    # Use memoization with a dictionary for visited states (remaining_mask, current_size)
    # But remaining_mask can be large. Let's just use recursion with good pruning.
    
    def branch(remaining_mask, current_size):
        if current_size >= 11:
            return True
        if remaining_mask == 0:
            return False
        if current_size + remaining_mask.bit_count() < 11:
            return False
        
        # Pick vertex with highest degree in remaining
        rem = remaining_mask
        best_v = -1
        best_deg = -1
        while rem:
            lsb = rem & (-rem)
            v = lsb.bit_length() - 1
            rem ^= lsb
            deg = (remaining_mask & masks[v]).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        if best_v == -1:
            return False

        # Branch include
        new_rem = remaining_mask & ~(masks[best_v] | (1 << best_v))
        if branch(new_rem, current_size + 1):
            return True
        
        # Branch exclude
        new_rem2 = remaining_mask & ~(1 << best_v)
        if branch(new_rem2, current_size):
            return True
        
        return False
    
    full = (1 << n) - 1
    return branch(full, 0)

def _get_conflict_masks(adj, n):
    """
    Returns a list of bitmasks for the conflict graph.
    conflict_mask[i] has bit j set if i and j share a common neighbor in adj.
    """
    conflict_masks = [0] * n
    for k in range(n):
        neighbors = list(adj[k])
        for i in range(len(neighbors)):
            u = neighbors[i]
            for j in range(i+1, len(neighbors)):
                v = neighbors[j]
                conflict_masks[u] |= (1 << v)
                conflict_masks[v] |= (1 << u)
                
    return conflict_masks

def _check_c4(adj):
    """adj: list of sets. Return True if no C4."""
    n = len(adj)
    for i in range(n):
        for j in range(i+1, n):
            if len(adj[i] & adj[j]) >= 2:
                return False
    return True

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = 118.0
    
    # Start with PG(2,5)
    adj = _build_polarity_graph(5)
    n = len(adj)
    
    # Verify initial graph
    if _has_indep_11(adj):
        # Fallback to a known valid smaller graph if PG(2,5) fails (unlikely)
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
        
    best_n = n
    best_adj = [set(s) for s in adj]
    
    # Main expansion loop
    # We try to add vertices one by one.
    # A new vertex can be connected to a set S of existing vertices if S is an independent set in the conflict graph.
    # The conflict graph has an edge (u,v) if N(u) and N(v) share a common neighbor.
    # If we add a new vertex w connected to S, then for any u, v in S, N(u) and N(v) must not share a common neighbor in the NEW graph.
    # In the old graph, they didn't share a common neighbor (since S is independent in conflict graph).
    # In the new graph, w is a neighbor of both u and v. So w is a common neighbor of u and v.
    # Wait. The condition is "No two vertices may have two common neighbours".
    # If u, v in S, they now have w as a common neighbor. Did they have any common neighbors before?
    # No, because S is independent in the conflict graph, which means no pair in S shares a common neighbor in the OLD graph.
    # So in the NEW graph, u and v have EXACTLY ONE common neighbor: w.
    # This is allowed! The condition is "No two vertices may have TWO common neighbours".
    # Having exactly one is fine.
    
    # So the strategy is correct.
    
    # To maximize n, we want to add vertices as long as alpha < 11.
    # Adding a vertex connected to S reduces alpha.
    # Specifically, any independent set in the new graph that includes w cannot include any vertex in S (since w is connected to all in S).
    # So it must be a subset of V \ (S ∪ {w}) that is independent in the OLD graph.
    # Any independent set in the new graph that does NOT include w is an independent set in the OLD graph.
    # So alpha_new = max(alpha_old, max_size_indep_set_in_V\S).
    # Since alpha_old < 11, we just need max_size_indep_set_in_V\S < 11.
    # If S is large, V\S is small, so this is likely true.
    
    # We want to add as many vertices as possible.
    # Greedy: find a large independent set S in the conflict graph, add w connected to S.
    # Repeat.
    
    # Optimization:
    # Instead of just greedy, we can try to find the MAXIMAL independent set in the conflict graph.
    # Or even better, use a heuristic to pick S such that the remaining graph still has alpha < 11.
    
    # Let's refine the loop.
    stuck_count = 0
    while _elapsed() < 100.0:
        if n >= 43:
            break
            
        conflict_masks = _get_conflict_masks(adj, n)
        
        # Find a large independent set in conflict graph
        # Greedy with randomization
        best_S = []
        best_alpha_remaining = 10**9
        
        for attempt in range(20):
            if _deadline_hit():
                break
                
            remaining = (1 << n) - 1
            S = []
            
            while remaining:
                # Pick vertex with min degree in remaining conflict graph
                # Tie-break randomly
                candidates = []
                min_deg = 10**9
                m = remaining
                while m:
                    lsb = m & (-m)
                    v = lsb.bit_length() - 1
                    m ^= lsb
                    deg = (conflict_masks[v] & remaining).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        candidates = [v]
                    elif deg == min_deg:
                        candidates.append(v)
                        
                if not candidates:
                    break
                    
                v = random.choice(candidates)
                S.append(v)
                remaining &= ~(conflict_masks[v] | (1 << v))
                
            if not S:
                continue
                
            # Check if adding a vertex connected to S keeps alpha < 11
            # We need to check if the graph induced by V \ S has an independent set of size 11.
            # Actually, we need to check the whole new graph.
            # But as analyzed, alpha_new = max(alpha_old, alpha(V\S)).
            # We know alpha_old < 11.
            # So we just need alpha(V\S) < 11.
            # V\S is a subgraph of the original graph.
            # We can check if V\S has an independent set of size 11.
            # This is faster than checking the whole new graph.
            
            # Let's implement a quick check for alpha(V\S) < 11.
            # We can use the same _has_indep_11 logic but on the subgraph induced by V\S.
            # Or simpler: just run _has_indep_11 on the new graph.
            # But that might be slow if n is large.
            # Let's try the subgraph check.
            
            # Build subgraph induced by V\S
            # Actually, just checking the new graph is safer and simpler.
            # The cost of _has_indep_11 is the bottleneck.
            # Let's see if we can prune.
            
            new_n = n + 1
            new_adj = [set(s) for s in adj]
            new_adj.append(set(S))
            for u in S:
                new_adj[u].add(new_n - 1)
                
            if not _has_indep_11(new_adj):
                # Valid
                # Prefer larger S?
                # Larger S means the new vertex has higher degree, which might help reduce alpha more?
                # Or does it?
                # If S is larger, V\S is smaller, so alpha(V\S) is smaller.
                # So yes, larger S is better for keeping alpha low.
                if len(S) > len(best_S):
                    best_S = S
                    best_adj_candidate = new_adj
                    best_n_candidate = new_n
                    
        if best_S:
            adj = best_adj_candidate
            n = best_n_candidate
            if n > best_n:
                best_n = n
                best_adj = [set(s) for s in adj]
            stuck_count = 0
        else:
            stuck_count += 1
            # If stuck, maybe try a different strategy?
            # For now, break.
            if stuck_count > 5:
                break
                
    # Final verification
    if _check_c4(best_adj) and not _has_indep_11(best_adj):
        n = len(best_adj)
        edges = []
        for i in range(n):
            for j in best_adj[i]:
                if i < j:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    else:
        # Fallback
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

# EVOLVE-BLOCK-END
