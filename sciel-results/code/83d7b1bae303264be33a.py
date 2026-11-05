"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict
from math import lcm

def construct():
    """
    Find a maximum subset of [1,128] with no three distinct elements
    whose three pairwise LCMs are all equal.
    
    Uses branch-and-bound with a tight upper bound derived from the 
    independence number of the conflict graph induced by the selected vertex.
    """
    n = 128
    # Precompute LCM table
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            lcm_table[i][j] = lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples
    pairs_by_lcm = defaultdict(list)
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            L = lcm_table[a][b]
            pairs_by_lcm[L].append((a, b))
            
    bad_triples = []
    for L, edges in pairs_by_lcm.items():
        adj = defaultdict(set)
        for a, b in edges:
            adj[a].add(b)
            adj[b].add(a)
        
        nodes = sorted(set(a for a, b in edges) | set(b for a, b in edges))
        for i in range(len(nodes)):
            a = nodes[i]
            for j in range(i+1, len(nodes)):
                b = nodes[j]
                if b in adj[a]:
                    common = adj[a] & adj[b]
                    for c in common:
                        if c > b:
                            bad_triples.append((a, b, c))
                            
    bad_triples = list(set(bad_triples))
    bad_triples.sort()
    
    if not bad_triples:
        return list(range(1, n+1))
        
    # Precompute conflict pairs for each vertex
    conflict_pairs_with = defaultdict(list)
    for a, b, c in bad_triples:
        conflict_pairs_with[a].append((b, c))
        conflict_pairs_with[b].append((a, c))
        conflict_pairs_with[c].append((a, b))
        
    # Precompute bad triple lookup
    bad_triple_set = defaultdict(lambda: defaultdict(set))
    for a, b, c in bad_triples:
        bad_triple_set[a][b].add(c)
        bad_triple_set[a][c].add(b)
        bad_triple_set[b][a].add(c)
        bad_triple_set[b][c].add(a)
        bad_triple_set[c][a].add(b)
        bad_triple_set[c][b].add(a)
        
    sys.setrecursionlimit(10000)
    best_size = 0
    best_set = []
    
    # Greedy initialization
    degrees = [0]*(n+1)
    for a, b, c in bad_triples:
        degrees[a] += 1
        degrees[b] += 1
        degrees[c] += 1
        
    def can_add_greedy(x, selected_set):
        sel_list = list(selected_set)
        for i in range(len(sel_list)):
            for j in range(i+1, len(sel_list)):
                a, b = sel_list[i], sel_list[j]
                if x in bad_triple_set[a][b]:
                    return False
        return True
        
    greedy_set = set()
    orders = [
        list(range(1, n+1)),
        list(range(n, 0, -1)),
        sorted(range(1, n+1), key=lambda x: -degrees[x])
    ]
    for order in orders:
        s = set()
        for x in order:
            if can_add_greedy(x, s):
                s.add(x)
        if len(s) > best_size:
            best_size = len(s)
            best_set = list(s)
            
    def greedy_vc_bound(edges, nodes):
        """
        Returns an upper bound on the size of the Maximum Independent Set
        by finding a greedy Vertex Cover.
        MIS <= N - VC
        """
        if not edges:
            return len(nodes)
            
        remaining_edges = set(edges)
        cover = set()
        
        while remaining_edges:
            u, v = next(iter(remaining_edges))
            
            # Pick vertex with higher degree in remaining_edges
            deg_u = 0
            deg_v = 0
            for e in remaining_edges:
                if e[0] == u or e[1] == u: deg_u += 1
                if e[0] == v or e[1] == v: deg_v += 1
                
            chosen = u if deg_u >= deg_v else v
            cover.add(chosen)
            
            new_remaining = set()
            for e in remaining_edges:
                if e[0] != chosen and e[1] != chosen:
                    new_remaining.add(e)
            remaining_edges = new_remaining
            
        return len(nodes) - len(cover)

    def solve(candidates, selected, forbidden_pairs, current_size):
        nonlocal best_size, best_set
        
        if current_size + len(candidates) <= best_size:
            return
            
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_set = list(selected)
            return
            
        v = candidates[0]
        rest = candidates[1:]
        
        # Branch 1: Exclude v
        solve(rest, selected, forbidden_pairs, current_size)
        
        # Branch 2: Include v
        valid = True
        sel_list = list(selected)
        for i in range(len(sel_list)):
            for j in range(i+1, len(sel_list)):
                a, b = sel_list[i], sel_list[j]
                if v in bad_triple_set[a][b]:
                    valid = False
                    break
            if not valid: break
            
        if valid:
            for u in sel_list:
                if (u, v) in forbidden_pairs or (v, u) in forbidden_pairs:
                    valid = False
                    break
            
        if valid:
            new_selected = selected | {v}
            new_forbidden = set(forbidden_pairs)
            
            for u, w in conflict_pairs_with[v]:
                if u in rest and w in rest:
                    new_forbidden.add((u, w))
                    new_forbidden.add((w, u))
                    
            # Calculate UB for Include branch
            rest_set = set(rest)
            valid_edges = [(u, w) for u, w in new_forbidden if u in rest_set and w in rest_set]
            # Remove duplicates? Set handles it.
            # But valid_edges is a list. Convert to set for greedy_vc_bound?
            # greedy_vc_bound takes a list or set. It converts to set internally.
            
            ub_add = greedy_vc_bound(valid_edges, rest)
            
            if current_size + 1 + ub_add > best_size:
                solve(rest, new_selected, new_forbidden, current_size + 1)
                
    initial_candidates = list(range(1, n+1))
    initial_candidates.sort(key=lambda x: -degrees[x])
    
    solve(initial_candidates, set(), set(), 0)
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
