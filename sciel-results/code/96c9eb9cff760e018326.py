"""Rewritten program: branch-and-bound over maximal cliques and maximal independent sets."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

import time
import sys

def construct():
    n = P['n']
    edge_set = set()
    for e in P['edges']:
        edge_set.add((e[0], e[1]))
        edge_set.add((e[1], e[0]))
    
    # Build adjacency bitmask
    adj = [0] * n
    for (u, v) in edge_set:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Find all maximal cliques using Bron–Kerbosch with pivot
    def bron_kerbosch(R, P_set, X, max_cliques):
        if not P_set and not X:
            if R:
                max_cliques.append(R)
            return
        # Choose pivot u from P_set ∪ X to minimize branching
        union = P_set | X
        if union:
            # Pick pivot with max degree in P_set
            best_u = -1
            best_deg = -1
            u = union
            while u:
                lsb = u & (-u)
                v = lsb.bit_length() - 1
                deg = (P_set & adj[v]).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_u = v
                u ^= lsb
            candidates = P_set & ~adj[best_u]
        else:
            candidates = P_set
        
        u = candidates
        while u:
            lsb = u & (-u)
            v = lsb.bit_length() - 1
            u ^= lsb
            bron_kerbosch(R | lsb, P_set & adj[v], X & adj[v], max_cliques)
            P_set ^= lsb
            X |= lsb
    
    max_cliques = []
    bron_kerbosch(0, full_mask, 0, max_cliques)
    
    # Find all maximal independent sets (maximal cliques in complement)
    comp_adj = [0] * n
    for v in range(n):
        comp_adj[v] = full_mask ^ (1 << v) ^ adj[v]
    
    max_indeps = []
    bron_kerbosch(0, full_mask, 0, max_indeps)
    # Note: bron_kerbosch uses adj[] internally, so we need a version that uses comp_adj
    # Let me redo this properly
    
    def bron_kerbosch_generic(R, P_set, X, adj_arr, max_sets):
        if not P_set and not X:
            if R:
                max_sets.append(R)
            return
        union = P_set | X
        if union:
            best_u = -1
            best_deg = -1
            u = union
            while u:
                lsb = u & (-u)
                v = lsb.bit_length() - 1
                deg = (P_set & adj_arr[v]).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_u = v
                u ^= lsb
            candidates = P_set & ~adj_arr[best_u]
        else:
            candidates = P_set
        
        u = candidates
        while u:
            lsb = u & (-u)
            v = lsb.bit_length() - 1
            u ^= lsb
            bron_kerbosch_generic(R | lsb, P_set & adj_arr[v], X & adj_arr[v], adj_arr, max_sets)
            P_set ^= lsb
            X |= lsb
    
    max_cliques = []
    bron_kerbosch_generic(0, full_mask, 0, adj, max_cliques)
    
    max_indeps = []
    bron_kerbosch_generic(0, full_mask, 0, comp_adj, max_indeps)
    
    # Combine all homogeneous sets (maximal cliques and maximal independent sets)
    # Each set is represented as a bitmask
    all_sets = list(set(max_cliques + max_indeps))
    
    # Filter: only keep sets that are subsets of full_mask (they all are)
    # Sort by size descending for better pruning
    all_sets.sort(key=lambda m: m.bit_count(), reverse=True)
    
    # Also include singleton vertices as sets (in case no maximal set contains a vertex... but maximal sets cover all vertices)
    # Actually every vertex is in at least one maximal clique and one maximal independent set, so we're fine.
    
    # Remove duplicate sets
    all_sets = list(dict.fromkeys(all_sets))  # preserve order, remove dups
    all_sets.sort(key=lambda m: m.bit_count(), reverse=True)
    
    # Now do branch-and-bound: find minimum number of disjoint sets from all_sets that cover full_mask
    # Since sets must be disjoint (partition), we need to pick a subset of all_sets that are pairwise disjoint and union to full_mask
    
    # For k=1: check if full_mask itself is in all_sets
    if full_mask in all_sets:
        part = [v for v in range(n) if full_mask & (1 << v)]
        return [part]
    
    # For k=2: check if there exist two disjoint sets s1, s2 in all_sets with s1|s2 == full_mask
    set_list = all_sets
    for i, s1 in enumerate(set_list):
        complement = full_mask ^ s1
        if complement == 0:
            continue
        # Check if complement is in all_sets
        # We need s2 = complement and s2 must be in all_sets
        if complement in all_sets:
            part1 = [v for v in range(n) if s1 & (1 << v)]
            part2 = [v for v in range(n) if complement & (1 << v)]
            return [part1, part2]
    
    # For k >= 3: branch and bound
    # We'll try to find a partition into k parts using sets from all_sets
    # DFS: at each step, pick a set that contains the smallest uncovered vertex
    
    start_time = time.time()
    deadline = 80.0  # seconds, safety margin before 90s kill
    
    best_solution = None
    best_k = n + 1
    
    # Precompute for each vertex, which sets contain it (for ordering)
    # sets_containing[v] = list of set indices that contain vertex v
    sets_containing = [[] for _ in range(n)]
    for idx, s in enumerate(all_sets):
        for v in range(n):
            if s & (1 << v):
                sets_containing[v].append(idx)
    
    # Sort sets_containing[v] by set size descending
    for v in range(n):
        sets_containing[v].sort(key=lambda idx: all_sets[idx].bit_count(), reverse=True)
    
    # DFS to find partition
    def dfs(covered, current_sets, depth, max_depth):
        nonlocal best_solution, best_k
        if time.time() > deadline:
            return False
        if depth >= best_k:
            return False
        
        if covered == full_mask:
            if depth < best_k:
                best_k = depth
                best_solution = list(current_sets)
            return True
        
        # Find the smallest uncovered vertex
        uncovered = full_mask ^ covered
        lsb = uncovered & (-uncovered)
        v = lsb.bit_length() - 1
        
        # Try all sets that contain v and are disjoint from covered
        for idx in sets_containing[v]:
            s = all_sets[idx]
            if s & covered:
                continue
            # Pruning: if depth + 1 >= best_k, skip
            if depth + 1 >= best_k:
                break
            # Lower bound: at least ceil(remaining_vertices / max_set_size) more parts needed
            remaining = full_mask ^ (covered | s)
            if remaining:
                # Simple lower bound: number of vertices not yet covered / max possible set size
                # But sets have varying sizes, so use a crude bound
                pass
            current_sets.append(idx)
            if dfs(covered | s, current_sets, depth + 1, max_depth):
                current_sets.pop()
                return True
            current_sets.pop()
        
        return False
    
    # Try increasing k
    for k in range(3, n + 1):
        if time.time() > deadline:
            break
        # Try to find a partition into exactly k parts
        # Use DFS with max_depth = k
        # We need to be smarter: the number of parts is exactly k, and they must be disjoint and cover all
        # But our DFS finds any number up to best_k. Let's just do DFS and see what it finds.
        
        # Reset for this k
        found = dfs(0, [], 0, k)
        if found:
            break
    
    if best_solution is not None:
        parts = []
        for idx in best_solution:
            s = all_sets[idx]
            part = [v for v in range(n) if s & (1 << v)]
            parts.append(part)
        # Verify
        all_verts = set()
        for p in parts:
            for v in p:
                all_verts.add(v)
        if all_verts == set(range(n)):
            # Verify each part is clique or independent
            valid = True
            for p in parts:
                if len(p) <= 1:
                    continue
                is_cliq = True
                is_ind = True
                for i in range(len(p)):
                    for j in range(i+1, len(p)):
                        if (p[i], p[j]) in edge_set:
                            is_ind = False
                        else:
                            is_cliq = False
                if not is_cliq and not is_ind:
                    valid = False
                    break
            if valid:
                return parts
    
    # Fallback: use the original DP approach
    size = 1 << n
    is_clique = [False] * size
    is_indep = [False] * size
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, size):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        if is_clique[rest]:
            if (rest & ~adj[v]) == 0:
                is_clique[mask] = True
        if is_indep[rest]:
            if (rest & adj[v]) == 0:
                is_indep[mask] = True
    
    valid = [is_clique[m] or is_indep[m] for m in range(size)]
    
    INF = 10**9
    dp = [INF] * size
    dp[0] = 0
    
    for mask in range(1, size):
        best = INF
        s = mask
        while s:
            if valid[s]:
                val = 1 + dp[mask ^ s]
                if val < best:
                    best = val
                    if best == 3:
                        break
            s = (s - 1) & mask
        dp[mask] = best
    
    min_parts = dp[full_mask]
    
    parts = []
    remaining = full_mask
    while remaining:
        s = remaining
        found = False
        while s:
            if valid[s] and dp[remaining ^ s] == dp[remaining] - 1:
                part = [v for v in range(n) if s & (1 << v)]
                parts.append(part)
                remaining ^= s
                found = True
                break
            s = (s - 1) & remaining
        if not found:
            break
    
    if not parts or len(parts) == 0:
        parts = [[v] for v in range(n)]
    
    return parts
# EVOLVE-BLOCK-END
