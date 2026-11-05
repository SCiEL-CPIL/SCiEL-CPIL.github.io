import time
import random
import itertools

# Q3 structure
_Q3_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
_Q3_ADJ = [0] * 8
for _a, _b in _Q3_EDGES:
    _Q3_ADJ[_a] |= (1 << _b)
    _Q3_ADJ[_b] |= (1 << _a)

# Precompute Q3 edge orientations with sorted unassigned vertices
# For each edge (qa,qb) in Q3, orient as (gu,gv) and order the other 6 vertices
# by how many of {gu,gv} they are adjacent to in Q3 (desc) to prune early.
_Q3_EDGE_ORIENTS = []
for qa, qb in _Q3_EDGES:
    for (gu_q, gv_q) in [(qa, qb), (qb, qa)]:
        unassigned = [i for i in range(8) if i != gu_q and i != gv_q]
        def _cc(x, a, b):
            c = 0
            if _Q3_ADJ[x] & (1 << a): c += 1
            if _Q3_ADJ[x] & (1 << b): c += 1
            return c
        unassigned.sort(key=lambda x: _cc(x, gu_q, gv_q), reverse=True)
        _Q3_EDGE_ORIENTS.append((gu_q, gv_q, tuple(unassigned)))

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin (150s limit)
    
    n = 16
    full_mask = (1 << n) - 1
    all_edges = list(itertools.combinations(range(n), 2))
    
    best_count = 0
    best_adj = [0] * n
    
    def has_q3_after_add(adj, u, v, dl):
        """Check if adding edge (u,v) to current adj creates a Q3 subgraph.
        Uses bitmask candidate intersection for fast pruning."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        for (qa, qb, unassigned) in _Q3_EDGE_ORIENTS:
            for (gu, gv) in [(u, v), (v, u)]:
                assignment = [None] * 8
                assignment[qa] = gu
                assignment[qb] = gv
                used_mask = (1 << gu) | (1 << gv)
                
                def bt(idx, used_mask):
                    if time.time() > dl - 2:
                        return False
                    if idx == 6:
                        # All 6 unassigned are assigned; verify all Q3 edges
                        for a, b in _Q3_EDGES:
                            ga = assignment[a]
                            gb = assignment[b]
                            if not (adj[ga] & (1 << gb)):
                                return False
                        return True
                    
                    x = unassigned[idx]
                    # Candidates: not used, adjacent to all assigned Q3-neighbors of x
                    candidates = full_mask & ~used_mask
                    for y in range(8):
                        if assignment[y] is not None and (_Q3_ADJ[x] & (1 << y)):
                            hy = assignment[y]
                            candidates &= adj[hy]
                    
                    while candidates:
                        bit = candidates & (-candidates)
                        g = bit.bit_length() - 1
                        candidates ^= bit
                        
                        assignment[x] = g
                        new_used = used_mask | (1 << g)
                        if bt(idx + 1, new_used):
                            return True
                        assignment[x] = None
                    return False
                
                if bt(0, used_mask):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    return True
        
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return False
    
    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2
    
    def edge_list(adj):
        es = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    es.append([u, v])
        return es
    
    def greedy_add_edges(adj, missing, dl):
        """Greedily add edges from missing list that don't create Q3."""
        cnt = count_edges(adj)
        for u, v in missing:
            if time.time() > dl - 5:
                break
            if not (adj[u] & (1 << v)):
                if not has_q3_after_add(adj, u, v, dl):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    cnt += 1
        return adj, cnt
    
    def local_search(adj, dl, max_rounds=100):
        """Try to add edges that don't create Q3 in multiple passes."""
        current_count = count_edges(adj)
        
        for iteration in range(max_rounds):
            if time.time() > dl - 5:
                break
            
            improved = False
            
            # Try adding each missing edge
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] & (1 << v)):
                        if not has_q3_after_add(adj, u, v, dl):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            current_count += 1
                            improved = True
            
            if not improved:
                break
        
        return adj, current_count
    
    def find_blocking_q3s(adj, u, v, dl):
        """Find Q3 embeddings that would be created by adding edge (u,v).
        Returns list of edge sets (each a set of 12 edges) that form Q3s."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        blocking_sets = []
        
        for (qa, qb, unassigned) in _Q3_EDGE_ORIENTS:
            for (gu, gv) in [(u, v), (v, u)]:
                assignment = [None] * 8
                assignment[qa] = gu
                assignment[qb] = gv
                used_mask = (1 << gu) | (1 << gv)
                
                def bt(idx, used_mask):
                    if time.time() > dl - 2:
                        return False
                    if idx == 6:
                        # Verify all Q3 edges exist
                        for a, b in _Q3_EDGES:
                            ga = assignment[a]
                            gb = assignment[b]
                            if not (adj[ga] & (1 << gb)):
                                return False
                        # Found a Q3 - record the edges
                        edge_set = frozenset((min(assignment[a], assignment[b]), max(assignment[a], assignment[b])) for a, b in _Q3_EDGES)
                        blocking_sets.append(edge_set)
                        return True  # Stop after finding one
                    
                    x = unassigned[idx]
                    candidates = full_mask & ~used_mask
                    for y in range(8):
                        if assignment[y] is not None and (_Q3_ADJ[x] & (1 << y)):
                            hy = assignment[y]
                            candidates &= adj[hy]
                    
                    while candidates:
                        bit = candidates & (-candidates)
                        g = bit.bit_length() - 1
                        candidates ^= bit
                        
                        assignment[x] = g
                        new_used = used_mask | (1 << g)
                        if bt(idx + 1, new_used):
                            return True
                        assignment[x] = None
                    return False
                
                bt(0, used_mask)
        
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        return blocking_sets
    
    def hitting_set_removal(adj, current_count, dl):
        """Greedy hitting-set strategy: identify Q3s blocking missing edges,
        remove minimal set of edges to unblock multiple edges at once."""
        best_adj = adj[:]
        best_count = current_count
        
        # Find all missing edges
        missing = []
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)):
                    missing.append((u, v))
        
        if not missing:
            return best_adj, best_count
        
        # For each missing edge, find blocking Q3s
        edge_blockers = {}
        for u, v in missing:
            if time.time() > dl - 10:
                break
            blockers = find_blocking_q3s(adj, u, v, dl)
            if blockers:
                edge_blockers[(u, v)] = blockers
        
        if not edge_blockers:
            return best_adj, best_count
        
        # Build edge-to-blocking-Q3 count
        # For each existing edge, count how many missing edges it blocks
        edge_block_count = {}
        for (u, v), blockers in edge_blockers.items():
            for q3_edges_set in blockers:
                for e in q3_edges_set:
                    if e == (u, v) or e == (v, u):
                        continue  # This is the missing edge itself
                    key = (min(e[0], e[1]), max(e[0], e[1]))
                    edge_block_count[key] = edge_block_count.get(key, 0) + 1
        
        # Greedy: repeatedly remove the edge that blocks the most missing edges
        trial_adj = adj[:]
        trial_count = current_count
        
        for iteration in range(20):  # Limit removals
            if time.time() > dl - 10:
                break
            
            # Find edge with highest block count
            if not edge_block_count:
                break
            
            max_edge = max(edge_block_count.keys(), key=lambda e: edge_block_count[e])
            max_count = edge_block_count[max_edge]
            
            if max_count < 2:
                break  # Not worth removing if it only blocks 1 missing edge
            
            # Remove this edge
            u, v = max_edge
            if trial_adj[u] & (1 << v):
                trial_adj[u] &= ~(1 << v)
                trial_adj[v] &= ~(1 << u)
                trial_count -= 1
            
            # Try to add back all missing edges
            added = 0
            for u2 in range(n):
                for v2 in range(u2+1, n):
                    if time.time() > dl - 10:
                        break
                    if not (trial_adj[u2] & (1 << v2)):
                        if not has_q3_after_add(trial_adj, u2, v2, dl):
                            trial_adj[u2] |= (1 << v2)
                            trial_adj[v2] |= (1 << u2)
                            added += 1
                            trial_count += 1
            
            if trial_count > best_count:
                best_count = trial_count
                best_adj = trial_adj[:]
            
            # Update block counts (recompute for remaining missing edges)
            edge_block_count = {}
            for u2 in range(n):
                for v2 in range(u2+1, n):
                    if time.time() > dl - 10:
                        break
                    if not (trial_adj[u2] & (1 << v2)):
                        blockers = find_blocking_q3s(trial_adj, u2, v2, dl)
                        for q3_edges_set in blockers:
                            for e in q3_edges_set:
                                if e == (u2, v2) or e == (v2, u2):
                                    continue
                                key = (min(e[0], e[1]), max(e[0], e[1]))
                                edge_block_count[key] = edge_block_count.get(key, 0) + 1
        
        return best_adj, best_count
    
    # Run multiple random trials
    trials = 20
    for t in range(trials):
        if time.time() > deadline - 15:
            break
        
        random.seed(t * 7919 + 42)
        
        # Use bitmask adjacency
        adj = [0] * n
        
        # Start with K3,13
        for u in range(3):
            for v in range(3, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Build list of remaining edges
        existing = set()
        for u in range(3):
            for v in range(3, 16):
                existing.add((u, v))
        
        remaining_edges = [e for e in all_edges if e not in existing]
        random.shuffle(remaining_edges)
        
        # Greedy add
        adj, count = greedy_add_edges(adj, remaining_edges, deadline)
        
        # Local search: try to add more edges
        adj, count = local_search(adj, deadline)
        
        # Hitting-set removal strategy
        adj, count = hitting_set_removal(adj, count, deadline)
        
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Convert best_adj to edge list
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    
    return result
