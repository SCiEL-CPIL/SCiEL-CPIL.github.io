"""K4-free graph without independent 6-sets: constructive search to maximize n."""
import time
import random
import sys

# Hard deadline guard
_DEADLINE = 150.0  # seconds wall-clock limit; use 140s safety margin
_MAX_N = 36  # target; 35 is published record

def _has_k4(adj, n):
    """Check if graph with adjacency matrix adj (list of int bitmasks) has a K4."""
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] >> j & 1:
                common = adj[i] & adj[j]
                # Check if any pair among common forms a K4 with i,j
                k = common & ~(1 << i) & ~(1 << j)
                while k:
                    kk = k & (-k)
                    k ^= kk
                    l = kk.bit_length() - 1
                    if l <= j:
                        continue
                    common2 = common & adj[l]
                    m = common2 & ~(1 << i) & ~(1 << j) & ~(1 << l)
                    while m:
                        mm = m & (-m)
                        m ^= mm
                        if mm & adj[l]:
                            return True
                    # simpler: check if l and m are adjacent
                    # Actually let me redo this more carefully
                # Redo: for each pair (i,j) that are adjacent, find common neighbors
                # and check if any two common neighbors are adjacent
        # This approach is getting complex; let me use a cleaner method
    return False

def _has_k4_simple(adj, n):
    """O(n^4) K4 check using bitmasks."""
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] >> j & 1):
                continue
            common = adj[i] & adj[j]
            common &= ~(1 << i) & ~(1 << j)
            # Iterate over pairs in common
            c_list = []
            c = common
            while c:
                bit = c & (-c)
                c ^= bit
                c_list.append(bit.bit_length() - 1)
            for a in range(len(c_list)):
                for b in range(a+1, len(c_list)):
                    if adj[c_list[a]] >> c_list[b] & 1:
                        return True
    return False

def _has_independent_6(adj, n):
    """Check if there's an independent set of size 6."""
    # Complement graph: non-edges
    # Use branch and bound
    def _find_iso(adj_comp, target, start):
        """Find independent set of size target in complement (i.e., clique in complement)."""
        if target <= 0:
            return True
        # Greedy lower bound
        remaining = list(range(start, n))
        if len(remaining) < target:
            return False
        # Try each vertex
        for v in remaining:
            # Can we extend?
            new_remaining = [u for u in remaining if u > v and not (adj_comp[v] >> u & 1)]
            if len(new_remaining) >= target - 1:
                if _find_iso(adj_comp, target - 1, v + 1):
                    return True
        return False
    
    # Build complement adjacency
    adj_comp = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] >> j & 1):
                adj_comp[i] |= (1 << j)
                adj_comp[j] |= (1 << i)
    
    return _find_iso(adj_comp, 6, 0)

def _is_valid(adj, n):
    """Check K4-free and no independent 6-set."""
    if _has_k4_simple(adj, n):
        return False
    if _has_independent_6(adj, n):
        return False
    return True

def _count_violations(adj, n):
    """Count K4s and independent 6-sets (for scoring)."""
    k4_count = 0
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] >> j & 1):
                continue
            common = adj[i] & adj[j] & ~(1 << i) & ~(1 << j)
            c_list = []
            c = common
            while c:
                bit = c & (-c)
                c ^= bit
                c_list.append(bit.bit_length() - 1)
            for a in range(len(c_list)):
                for b in range(a+1, len(c_list)):
                    if adj[c_list[a]] >> c_list[b] & 1:
                        k4_count += 1
    # Independent 6: just check existence
    iso6 = 1 if _has_independent_6(adj, n) else 0
    return k4_count + iso6

def _greedy_construct(n, rng):
    """Try to construct a valid graph on n vertices via greedy random method."""
    adj = [0] * n
    for v in range(1, n):
        # Choose neighbors among previous vertices
        # Constraint: no K4, no independent 6
        # Greedily: try to add edges that don't create K4
        # We'll use a randomized approach: for each previous vertex, decide edge with some probability
        # But we need to ensure no K4 and no independent 6
        
        # Strategy: start with no edges to v, then add edges one by one
        # Check constraints after each addition
        
        # Actually, let's use a different approach:
        # Assign v's neighborhood as a random subset of {0,...,v-1}
        # Then check if it creates a K4 or independent 6
        # If not, keep it; if yes, try another subset
        
        # For efficiency, use the fact that:
        # - Adding v with neighborhood N_v: creates K4 iff there exist i,j in N_v that are adjacent
        #   and both i,j have a common neighbor in N_v (forming K4 with v)
        #   More precisely: K4 containing v means v,i,j,k all pairwise adjacent
        #   So we need i,j,k in N_v forming a triangle
        # - Independent 6 containing v: v plus 5 vertices not in N_v that form an independent set
        #   in the subgraph induced by V \\ N_v \\ {v}
        
        # This is complex. Let's use a simpler iterative approach.
        
        # For each previous vertex u, decide if (v,u) is an edge
        # Use a heuristic: prefer to add edges to avoid large independent sets
        # but avoid creating K4s
        
        # Simple approach: random subset, validate, retry
        best_adj = None
        for attempt in range(50):
            # Random neighborhood
            N_v = set()
            for u in range(v):
                if rng.random() < 0.5:
                    N_v.add(u)
            
            # Check K4: does N_v contain a triangle?
            has_tri = False
            for i in N_v:
                for j in N_v:
                    if i < j and (adj[i] >> j & 1):
                        # Check if there's k in N_v adjacent to both i and j
                        common = adj[i] & adj[j] & ~(1 << i) & ~(1 << j)
                        for k in N_v:
                            if k != i and k != j and (common >> k & 1):
                                has_tri = True
                                break
                        if has_tri:
                            break
                if has_tri:
                    break
            if has_tri:
                continue
            
            # Check independent 6: v + 5 from V\N_v\{v}
            non_neighbors = [u for u in range(v) if u not in N_v]
            # Need independent set of size 5 in non_neighbors
            # Quick check: if len(non_neighbors) < 5, skip
            if len(non_neighbors) >= 5:
                # Check for independent 5-set in non_neighbors
                # This is expensive; use a quick heuristic
                # For now, just check if all pairs are non-adjacent (too strict)
                # Better: use a small search
                def _has_iso5(subset, adj_local):
                    """Check if subset has independent set of size 5."""
                    if len(subset) < 5:
                        return False
                    # Greedy: try to find 5 non-adjacent vertices
                    chosen = []
                    remaining = list(subset)
                    while remaining and len(chosen) < 5:
                        # Pick vertex with fewest neighbors in remaining
                        best_v = min(remaining, key=lambda x: sum(1 for y in remaining if y != x and (adj_local[x] >> y & 1)))
                        chosen.append(best_v)
                        remaining = [y for y in remaining if y != best_v and not (adj_local[best_v] >> y & 1)]
                    return len(chosen) >= 5
                
                if _has_iso5(non_neighbors, adj):
                    continue
            
            # Valid!
            best_adj = adj[:]
            for u in N_v:
                best_adj[v] |= (1 << u)
                best_adj[u] |= (1 << v)
            break
        
        if best_adj is None:
            return None
        adj = best_adj
    
    return adj

def _local_search(adj, n, rng, max_iters=10000):
    """Improve graph via local search: flip edges to reduce violations."""
    adj = adj[:]
    best_score = _count_violations(adj, n)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        # Try flipping a random edge
        u = rng.randint(0, n-1)
        v = rng.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        # Flip edge (u,v)
        if adj[u] >> v & 1:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        new_score = _count_violations(adj, n)
        if new_score < best_score:
            best_score = new_score
            if best_score == 0:
                return adj
        else:
            # Revert
            if adj[u] >> v & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    return adj

def construct():
    """Construct a K4-free graph with no independent 6-set, maximizing n."""
    start_time = time.time()
    deadline = start_time + _DEADLINE - 10  # 140s effective
    
    rng = random.Random(42)
    
    # Start from known good: Paley graph on 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj17 = [0] * 17
    for i in range(17):
        for j in range(i+1, 17):
            if (j - i) % 17 in squares:
                adj17[i] |= (1 << j)
                adj17[j] |= (1 << i)
    
    best_n = 17
    best_adj = adj17
    
    # Try to extend to larger n
    for target_n in range(18, _MAX_N + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Start from best_adj and add vertices
            adj = best_adj[:] + [0] * (target_n - best_n)
            
            # Add new vertices one by one
            success = True
            for v in range(best_n, target_n):
                # Try to find a valid neighborhood for v
                valid_found = False
                for inner_attempt in range(100):
                    N_v = set()
                    for u in range(v):
                        if rng.random() < 0.5:
                            N_v.add(u)
                    
                    # Check K4
                    has_tri = False
                    nv_list = list(N_v)
                    for a in range(len(nv_list)):
                        for b in range(a+1, len(nv_list)):
                            i, j = nv_list[a], nv_list[b]
                            if adj[i] >> j & 1:
                                common = adj[i] & adj[j] & ~(1 << i) & ~(1 << j)
                                for k in nv_list:
                                    if k != i and k != j and (common >> k & 1):
                                        has_tri = True
                                        break
                            if has_tri:
                                break
                        if has_tri:
                            break
                    if has_tri:
                        continue
                    
                    # Check independent 6
                    non_neighbors = [u for u in range(v) if u not in N_v]
                    if len(non_neighbors) >= 5:
                        # Quick check for independent 5
                        # Use a simple greedy
                        def _greedy_iso(subset, adj_mat, target=5):
                            chosen = []
                            remaining = list(subset)
                            while remaining and len(chosen) < target:
                                # Pick vertex with min degree in remaining
                                best_v = min(remaining, key=lambda x: sum(1 for y in remaining if y != x and (adj_mat[x] >> y & 1)))
                                chosen.append(best_v)
                                remaining = [y for y in remaining if y != best_v and not (adj_mat[best_v] >> y & 1)]
                            return len(chosen) >= target
                        
                        if _greedy_iso(non_neighbors, adj):
                            continue
                    
                    # Valid
                    valid_found = True
                    for u in N_v:
                        adj[v] |= (1 << u)
                        adj[u] |= (1 << v)
                    break
                
                if not valid_found:
                    success = False
                    break
            
            if success:
                # Verify
                if _is_valid(adj, target_n):
                    best_n = target_n
                    best_adj = adj
                    found = True
                    break
        
        if not found:
            # Try local search on a random graph
            if time.time() > deadline:
                break
            adj_try = [0] * target_n
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if rng.random() < 0.5:
                        adj_try[i] |= (1 << j)
                        adj_try[j] |= (1 << i)
            
            adj_try = _local_search(adj_try, target_n, rng, max_iters=5000)
            if _is_valid(adj_try, target_n):
                best_n = target_n
                best_adj = adj_try
    
    # Convert best_adj to edge list
    edges = []
    for i in range(best_n):
        for j in range(i+1, best_n):
            if best_adj[i] >> j & 1:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
