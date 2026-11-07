"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _validate(bicliques, adj, all_edges_set):
    """Check that bicliques form a valid partition."""
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        # Check disjointness of sides within a biclique
        if set(left).intersection(right):
            return False
        # Check cross pairs are all edges
        for u in left:
            for v in right:
                if v not in adj[u]:
                    return False
                e = (min(u, v), max(u, v))
                if e in covered:
                    return False
                covered.add(e)
    # Check all edges covered
    if covered != all_edges_set:
        return False
    return True

def _expand_biclique(start_left, start_right, adj, n, remaining_edges_set=None):
    """Expand a biclique to be maximal, optionally constrained to remaining edges."""
    left = set(start_left)
    right = set(start_right)
    changed = True
    while changed:
        changed = False
        # Add to left: vertices connected to all of right
        for u in range(n):
            if u not in left and u not in right:
                if right.issubset(adj[u]):
                    if remaining_edges_set is None:
                        left.add(u)
                        changed = True
                    else:
                        has_rem = False
                        for v in right:
                            if (min(u, v), max(u, v)) in remaining_edges_set:
                                has_rem = True
                                break
                        if has_rem:
                            left.add(u)
                            changed = True
        # Add to right: vertices connected to all of left
        for v in range(n):
            if v not in left and v not in right:
                if left.issubset(adj[v]):
                    if remaining_edges_set is None:
                        right.add(v)
                        changed = True
                    else:
                        has_rem = False
                        for u in left:
                            if (min(u, v), max(u, v)) in remaining_edges_set:
                                has_rem = True
                                break
                        if has_rem:
                            right.add(v)
                            changed = True
    right -= left
    left -= right
    if not left or not right:
        return None
    return (left, right)

def _get_biclique_edges(left, right, adj):
    edges = set()
    for u in left:
        for v in right:
            if v in adj[u]:
                edges.add((min(u, v), max(u, v)))
    return edges

def _greedy_max_biclique(adj, remaining_edges_set, n):
    """Find a large biclique among remaining edges using heuristic."""
    best_biclique = None
    best_size = 0
    
    # Try each vertex as anchor on left side
    for anchor in range(n):
        anchor_neigh = set()
        for (u, v) in remaining_edges_set:
            if u == anchor:
                anchor_neigh.add(v)
            elif v == anchor:
                anchor_neigh.add(u)
        
        if not anchor_neigh:
            continue
        
        left_set = {anchor}
        right_set = set(anchor_neigh)
        
        # Expand
        changed = True
        while changed:
            changed = False
            new_left = set()
            for u in range(n):
                if u in left_set or u in right_set:
                    continue
                if right_set.issubset(adj[u]):
                    has_remaining = False
                    for v in right_set:
                        e = (min(u, v), max(u, v))
                        if e in remaining_edges_set:
                            has_remaining = True
                            break
                    if has_remaining:
                        new_left.add(u)
                        changed = True
            left_set |= new_left
            
            new_right = set()
            for v in range(n):
                if v in left_set or v in right_set:
                    continue
                if left_set.issubset(adj[v]):
                    has_remaining = False
                    for u in left_set:
                        e = (min(u, v), max(u, v))
                        if e in remaining_edges_set:
                            has_remaining = True
                            break
                    if has_remaining:
                        new_right.add(v)
                        changed = True
            right_set |= new_right
        
        right_set -= left_set
        left_set -= right_set
        
        # Verify it's a valid biclique on remaining edges
        valid = True
        for u in left_set:
            for v in right_set:
                if v not in adj[u]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and left_set and right_set:
            size = len(left_set) * len(right_set)
            if size > best_size:
                best_size = size
                best_biclique = (set(left_set), set(right_set))
    
    return best_biclique

def _greedy_partition(adj, all_edges_set, deadline, n):
    """Greedy: repeatedly extract largest biclique from remaining edges."""
    remaining = set(all_edges_set)
    bicliques = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        biclique = _greedy_max_biclique(adj, remaining, n)
        if biclique is None:
            e = next(iter(remaining))
            biclique = ({e[0]}, {e[1]})
        
        left_set, right_set = biclique
        for u in left_set:
            for v in right_set:
                e = (min(u, v), max(u, v))
                remaining.discard(e)
        
        bicliques.append((left_set, right_set))
    
    return bicliques

def _biclique_edges_set(left, right):
    s = set()
    for u in left:
        for v in right:
            s.add((min(u, v), max(u, v)))
    return s

def _local_search(bicliques, adj, all_edges_set, deadline, n):
    """Try to merge or improve bicliques via exhaustive pair-merge candidates."""
    bicliques = [list(bc) for bc in bicliques]
    
    # Precompute edge sets for each biclique
    def compute_edge_sets():
        return [_biclique_edges_set(l, r) for l, r in bicliques]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        edge_sets = compute_edge_sets()
        for i in range(len(bicliques)):
            for j in range(i + 1, len(bicliques)):
                if time.time() > deadline:
                    break
                li, ri = bicliques[i]
                lj, rj = bicliques[j]
                Li, Ri, Lj, Rj = set(li), set(ri), set(lj), set(rj)
                
                # Generate all merge candidates: pick left from {Li, Lj, Li|Lj, Li&Lj}
                # and right from {Ri, Rj, Ri|Rj, Ri&Rj}, ensuring disjoint and valid
                left_opts = [Li, Lj, Li | Lj, Li & Lj]
                right_opts = [Ri, Rj, Ri | Rj, Ri & Rj]
                
                found = False
                for new_left in left_opts:
                    if not new_left:
                        continue
                    for new_right in right_opts:
                        if not new_right:
                            continue
                        if new_left & new_right:
                            continue
                        # Validity: all cross pairs are edges
                        valid = True
                        for u in new_left:
                            for v in new_right:
                                if v not in adj[u]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        merge_edges = _biclique_edges_set(new_left, new_right)
                        old_edges = edge_sets[i] | edge_sets[j]
                        if merge_edges != old_edges:
                            continue
                        # Check no overlap with other bicliques
                        overlap = False
                        for k in range(len(bicliques)):
                            if k == i or k == j:
                                continue
                            if merge_edges & edge_sets[k]:
                                overlap = True
                                break
                        if overlap:
                            continue
                        # Accept merge
                        bicliques[i] = [list(new_left), list(new_right)]
                        del bicliques[j]
                        improved = True
                        found = True
                        break
                    if found:
                        break
                if improved:
                    break
            if improved:
                break
    
    return bicliques

def _randomized_greedy(adj, all_edges_set, deadline, n, seed):
    """Randomized greedy partition."""
    random.seed(seed)
    remaining = set(all_edges_set)
    bicliques = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        e = random.choice(list(remaining))
        u0, v0 = e
        
        left_set = {u0}
        right_set = {v0}
        
        changed = True
        while changed:
            changed = False
            new_left = set()
            for u in range(n):
                if u in left_set or u in right_set:
                    continue
                if right_set.issubset(adj[u]):
                    has_rem = False
                    for v in right_set:
                        if (min(u, v), max(u, v)) in remaining:
                            has_rem = True
                            break
                    if has_rem:
                        new_left.add(u)
                        changed = True
            left_set |= new_left
            
            new_right = set()
            for v in range(n):
                if v in left_set or v in right_set:
                    continue
                if left_set.issubset(adj[v]):
                    has_rem = False
                    for u in left_set:
                        if (min(u, v), max(u, v)) in remaining:
                            has_rem = True
                            break
                    if has_rem:
                        new_right.add(v)
                        changed = True
            right_set |= new_right
        
        right_set -= left_set
        left_set -= right_set
        
        if not left_set or not right_set:
            left_set = {u0}
            right_set = {v0}
        
        for u in left_set:
            for v in right_set:
                remaining.discard((min(u, v), max(u, v)))
        
        bicliques.append((left_set, right_set))
    
    return bicliques

def _candidate_pool_partition(candidates, all_edges_set, deadline, adj, n):
    """Greedy set cover using pre-generated candidate bicliques."""
    remaining = set(all_edges_set)
    bicliques = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        best_cand = None
        best_overlap = 0
        for left, right, edges in candidates:
            overlap = len(edges & remaining)
            if overlap > best_overlap:
                best_overlap = overlap
                best_cand = (left, right, edges)
        
        if best_cand is None or best_overlap == 0:
            e = next(iter(remaining))
            bicliques.append(({e[0]}, {e[1]}))
            remaining.discard(e)
        else:
            left, right, edges = best_cand
            bicliques.append((left, right))
            remaining -= edges
    
    return bicliques

def _generate_candidates(adj, n, all_edges_set, num_random=300):
    """Generate a pool of candidate bicliques."""
    candidates = set()
    
    # 1. Maximal bicliques containing each edge (both orientations)
    for e in all_edges_set:
        u, v = e
        bc = _expand_biclique({u}, {v}, adj, n)
        if bc:
            candidates.add((frozenset(bc[0]), frozenset(bc[1])))
        bc = _expand_biclique({v}, {u}, adj, n)
        if bc:
            candidates.add((frozenset(bc[0]), frozenset(bc[1])))
    
    # 2. Random expansions with multiple seeds for diversity
    random.seed(42)
    for trial in range(num_random):
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v and v in adj[u]:
            bc = _expand_biclique({u}, {v}, adj, n)
            if bc:
                candidates.add((frozenset(bc[0]), frozenset(bc[1])))
        
        l = random.randint(0, n - 1)
        r = random.randint(0, n - 1)
        if l != r:
            bc = _expand_biclique({l}, {r}, adj, n)
            if bc:
                candidates.add((frozenset(bc[0]), frozenset(bc[1])))
    
    # 3. For each vertex, try expanding with each of its neighbors as the other side
    for u in range(n):
        for v in adj[u]:
            bc = _expand_biclique({u}, {v}, adj, n)
            if bc:
                candidates.add((frozenset(bc[0]), frozenset(bc[1])))
    
    # Convert to list of (left_set, right_set, edge_set)
    cand_list = []
    for l, r in candidates:
        edges = _get_biclique_edges(l, r, adj)
        if edges:
            cand_list.append((set(l), set(r), edges))
    
    return cand_list

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    n = P['n']
    adj = _build_adjacency()
    all_edges_set = set((min(u, v), max(u, v)) for u, v in P['edges'])
    
    best_result = None
    best_count = float('inf')
    
    # Strategy 1: Greedy partition
    bicliques = _greedy_partition(adj, all_edges_set, deadline, n)
    if bicliques:
        if _validate(bicliques, adj, all_edges_set):
            best_result = bicliques
            best_count = len(bicliques)
    
    # Strategy 2: Local search improvement
    if best_result and time.time() < deadline:
        improved = _local_search(best_result, adj, all_edges_set, deadline, n)
        if _validate(improved, adj, all_edges_set) and len(improved) < best_count:
            best_result = improved
            best_count = len(improved)
    
    # Strategy 3: Candidate pool based set cover
    if time.time() < deadline:
        candidates = _generate_candidates(adj, n, all_edges_set, num_random=200)
        if candidates:
            pool_result = _candidate_pool_partition(candidates, all_edges_set, deadline, adj, n)
            if pool_result and _validate(pool_result, adj, all_edges_set):
                if len(pool_result) < best_count:
                    best_count = len(pool_result)
                    best_result = pool_result
                if time.time() < deadline:
                    improved = _local_search(pool_result, adj, all_edges_set, deadline, n)
                    if _validate(improved, adj, all_edges_set) and len(improved) < best_count:
                        best_count = len(improved)
                        best_result = improved
    
    # Strategy 4: Multiple randomized greedy runs
    if time.time() < deadline:
        for trial in range(80):
            if time.time() > deadline:
                break
            
            bicliques = _randomized_greedy(adj, all_edges_set, deadline, n, seed=trial)
            if bicliques and _validate(bicliques, adj, all_edges_set):
                if len(bicliques) < best_count:
                    best_count = len(bicliques)
                    best_result = bicliques
                
                if time.time() < deadline:
                    improved = _local_search(bicliques, adj, all_edges_set, deadline, n)
                    if _validate(improved, adj, all_edges_set) and len(improved) < best_count:
                        best_count = len(improved)
                        best_result = improved
    
    # Final fallback: seed solution
    if best_result is None:
        best_result = [[[u], [v for a, v in P['edges'] if a == u]] for u in range(n) if any(a == u for a, v in P['edges'])]
    
    # Convert to proper types
    result = []
    for left, right in best_result:
        result.append([list(left), list(right)])
    
    return result
# EVOLVE-BLOCK-END
