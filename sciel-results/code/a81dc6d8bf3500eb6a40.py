import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        C = list(range(k))
        I = list(range(k, n))
        C_set = set(C)
        
        # Internal C edges
        internal_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                if v in adj[u]:
                    internal_edges.add((u, v))
        
        # Edges from I to C
        i_c_edges = {}
        for v in I:
            nbrs = adj[v] & C_set
            i_c_edges[v] = nbrs
        
        # Group I vertices by their exact neighborhood in C
        groups = defaultdict(list)
        for v in I:
            if i_c_edges[v]:
                groups[frozenset(i_c_edges[v])].append(v)
        
        # Sort groups by shadow size descending
        sorted_groups = sorted(groups.items(), key=lambda x: -len(x[0]))
        
        best_partition = None
        best_count = float('inf')
        
        # Strategy: 
        # 1. Greedily form large cliques covering I vertices and their shadows.
        # 2. For remaining C-C edges, use a "Maximal Clique" heuristic based on 
        #    common neighbors, trying to pack as many edges into each clique as possible.
        #    This is a structural change from simple greedy edge-picking to a 
        #    clique-expansion approach that respects the chordal nature better 
        #    by finding large independent sets in the complement (cliques in G).
        
        # We will run a few variations of the heuristic to find the best partition.
        
        for strategy_idx in range(3):
            if time.time() > deadline - 10.0:
                break
                
            partition = []
            used_c_edges = set()
            
            # Process I groups
            # Order can be varied, but for now we stick to descending shadow size 
            # as it tends to cover more edges per clique.
            
            group_order = sorted_groups
            if strategy_idx == 1:
                # Try ascending? Might be risky but worth a shot if time permits.
                # Actually, descending is usually better for "covering" because 
                # large shadows are harder to cover with small cliques.
                group_order = sorted(sorted_groups, key=lambda x: len(x[0]))
            elif strategy_idx == 2:
                # Randomized ordering based on a seed?
                # Let's just stick to deterministic for now to avoid noise, 
                # but maybe shuffle slightly? No, keep it simple.
                pass
            
            for shadow, i_vertices in group_order:
                if not i_vertices:
                    continue
                
                available = list(shadow)
                if not available:
                    continue
                
                def get_unused_degree(u, avail_set):
                    cnt = 0
                    for w in avail_set:
                        if u != w:
                            e = (min(u, w), max(u, w))
                            if e in internal_edges and e not in used_c_edges:
                                cnt += 1
                    return cnt
                
                # Try to find a large clique in the shadow using iterative expansion
                # Start with the vertex of highest unused degree, then iteratively add
                # the best candidate that connects to all current clique members.
                
                def greedy_clique_from(start_vertex, cand_set):
                    """Build a clique starting from start_vertex, expanding greedily."""
                    clq = [start_vertex]
                    clq_set = {start_vertex}
                    remaining = set(cand_set) - clq_set
                    while remaining:
                        # Find candidate with max unused degree among those connecting to all clique members
                        best_cand = None
                        best_deg = -1
                        for c in remaining:
                            # Check c connects to all clique members
                            ok = True
                            for w in clq:
                                e = (min(c, w), max(c, w))
                                if e not in internal_edges or e in used_c_edges:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            deg = 0
                            for w in remaining:
                                if w != c:
                                    e2 = (min(c, w), max(c, w))
                                    if e2 in internal_edges and e2 not in used_c_edges:
                                        deg += 1
                            if deg > best_deg:
                                best_deg = deg
                                best_cand = c
                        if best_cand is None:
                            break
                        clq.append(best_cand)
                        clq_set.add(best_cand)
                        remaining.discard(best_cand)
                    return clq
                
                # Try starting from each of the top few vertices by unused degree
                available.sort(key=lambda u: -get_unused_degree(u, set(available)))
                best_c_clique = []
                for start_idx in range(min(3, len(available))):
                    sv = available[start_idx]
                    # Candidates must connect to sv
                    cands = set()
                    for w in available:
                        if w == sv:
                            continue
                        e = (min(sv, w), max(sv, w))
                        if e in internal_edges and e not in used_c_edges:
                            cands.add(w)
                    cl = greedy_clique_from(sv, cands)
                    if len(cl) > len(best_c_clique):
                        best_c_clique = cl
                
                c_clique = best_c_clique
                if len(c_clique) >= 1:
                    new_clique = sorted(i_vertices + c_clique)
                    partition.append(new_clique)
                    for i in range(len(c_clique)):
                        for j in range(i + 1, len(c_clique)):
                            used_c_edges.add((min(c_clique[i], c_clique[j]), max(c_clique[i], c_clique[j])))
                
                rem_i = list(i_vertices)
                rem_c = list(shadow - set(c_clique))
                
                while rem_i and rem_c:
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Find a clique in rem_c using unused edges via iterative expansion
                    rem_c_set = set(rem_c)
                    rem_c.sort(key=lambda u: -get_unused_degree(u, rem_c_set))
                    
                    best_sub_clique = []
                    for start_idx in range(min(3, len(rem_c))):
                        sv = rem_c[start_idx]
                        cands = set()
                        for w in rem_c:
                            if w == sv:
                                continue
                            e = (min(sv, w), max(sv, w))
                            if e in internal_edges and e not in used_c_edges:
                                cands.add(w)
                        cl = greedy_clique_from(sv, cands)
                        if len(cl) > len(best_sub_clique):
                            best_sub_clique = cl
                    
                    if best_sub_clique:
                        sub_clique = best_sub_clique
                    else:
                        sub_clique = [rem_c[0]]
                    
                    new_clique = sorted(rem_i + sub_clique)
                    partition.append(new_clique)
                    
                    for i_idx in range(len(sub_clique)):
                        for j_idx in range(i_idx + 1, len(sub_clique)):
                            used_c_edges.add((min(sub_clique[i_idx], sub_clique[j_idx]), max(sub_clique[i_idx], sub_clique[j_idx])))
                    
                    sub_set = set(sub_clique)
                    rem_c = [c for c in rem_c if c not in sub_set]

            # Cover remaining C-C edges using iterative greedy expansion
            rem_set = internal_edges - used_c_edges
            
            # Precompute adjacency restricted to C vertices for speed
            c_adj = [set() for _ in range(k)]
            for u in range(k):
                for v in range(u + 1, k):
                    if (u, v) in internal_edges:
                        c_adj[u].add(v)
                        c_adj[v].add(u)
            
            while rem_set:
                if time.time() > deadline - 3.0:
                    break
                
                # Pick the edge with maximum number of valid common neighbors
                # in the remaining graph. Use a faster approach:
                # For each edge, count common neighbors that have both edges to a and b remaining.
                
                # Build a quick lookup: for each vertex, which of its C-neighbors still have edges to it
                # Actually, let's just iterate edges and compute common neighbors on the fly
                # but use the c_adj structure for candidate generation.
                
                best_edge = None
                max_common = -1
                # Sample edges if too many for speed
                edge_list = list(rem_set)
                if len(edge_list) > 500:
                    import random as _rnd
                    _rnd.seed(42)
                    edge_list = _rnd.sample(edge_list, 500)
                
                for (a, b) in edge_list:
                    # Common neighbors in C that connect to both a and b in remaining graph
                    common = c_adj[a] & c_adj[b]
                    valid_common = 0
                    for c in common:
                        if (min(a, c), max(a, c)) in rem_set and (min(b, c), max(b, c)) in rem_set:
                            valid_common += 1
                    if valid_common > max_common:
                        max_common = valid_common
                        best_edge = (a, b)
                
                if best_edge is None:
                    break
                
                a, b = best_edge
                clique = [a, b]
                clique_set = {a, b}
                
                # Iteratively expand: at each step, add the best candidate
                # that connects to all current clique members in the remaining graph
                while True:
                    # Find all candidates that connect to every member of clique
                    candidates = []
                    for c in range(k):
                        if c in clique_set:
                            continue
                        ok = True
                        for u in clique:
                            e = (min(u, c), max(u, c))
                            if e not in rem_set:
                                ok = False
                                break
                        if ok:
                            candidates.append(c)
                    
                    if not candidates:
                        break
                    
                    # Score each candidate by number of remaining edges to other candidates
                    # (this helps build larger cliques)
                    def cand_score(c):
                        cnt = 0
                        for w in candidates:
                            if w != c:
                                e = (min(c, w), max(c, w))
                                if e in rem_set:
                                    cnt += 1
                        return cnt
                    
                    candidates.sort(key=lambda u: -cand_score(u))
                    
                    # Add the best candidate
                    best_c = candidates[0]
                    clique.append(best_c)
                    clique_set.add(best_c)
                    # Remove edges covered by adding best_c
                    for u in clique[:-1]:
                        e = (min(u, best_c), max(u, best_c))
                        rem_set.discard(e)
                    
                    # Check if we can continue expanding
                    # (the next iteration will recompute candidates)
                    if time.time() > deadline - 2.0:
                        break
                
                if len(clique) >= 2:
                    partition.append(sorted(clique))
                    # Remove all edges within the clique
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            rem_set.discard((min(clique[i], clique[j]), max(clique[i], clique[j])))
            
            cnt = len(partition)
            if cnt < best_count:
                best_count = cnt
                best_partition = partition
        
        if best_partition is None:
            return []
        
        # Validate and clean: ensure each clique has >= 2 distinct int vertices
        cleaned = []
        for clique in best_partition:
            c = [int(x) for x in clique]
            if len(c) >= 2:
                cleaned.append(c)
        return cleaned

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        results.append(res)

    return results
