"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _ek(e):
    a, b = e
    return (a, b) if a < b else (b, a)

def _adj(n, edges):
    A = [set() for _ in range(n)]
    for u, v in edges:
        A[u].add(v)
        A[v].add(u)
    return A

def _max_biclique_from_seed(seed_left, seed_right, edge_set, adj, n, deadline):
    """Grow a biclique from initial left/right sets, adding vertices that connect fully to opposite side."""
    left = set(seed_left)
    right = set(seed_right)
    if not left or not right or (left & right):
        return None
    # Verify all cross edges exist
    for u in left:
        for v in right:
            if _ek((u, v)) not in edge_set:
                return None
    # Use adjacency sets for fast "connects to all" checks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try to add to left: vertex w connects to all in right
        for w in range(n):
            if w in left or w in right:
                continue
            if right.issubset(adj[w]):
                left.add(w)
                changed = True
        # Try to add to right: vertex w connects to all in left
        for w in range(n):
            if w in left or w in right:
                continue
            if left.issubset(adj[w]):
                right.add(w)
                changed = True
    return (left, right)

def _cross_edges_of_biclique(left, right, edge_set):
    es = set()
    for u in left:
        for v in right:
            k = _ek((u, v))
            if k in edge_set:
                es.add(k)
    return es

def _greedy_partition(edge_set, adj, n, deadline, rng, max_seeds=200):
    """Greedy: repeatedly pick a maximal biclique, remove its edges."""
    remaining = set(edge_set)
    bicliques = []
    
    if not remaining:
        return bicliques
    
    # Precompute degrees for seed ordering
    deg = {v: 0 for v in range(n)}
    for u, v in edge_set:
        deg[u] += 1
        deg[v] += 1
    
    while remaining and time.time() < deadline:
        # Build list of remaining edges
        rem_edges = [(u, v) for u, v in remaining]
        rem_set = set(remaining)
        
        # Compute current degrees
        cur_deg = {v: 0 for v in range(n)}
        for u, v in rem_edges:
            cur_deg[u] += 1
            cur_deg[v] += 1
        
        # Candidate seeds: edges in remaining, sorted by combined degree
        candidates = []
        for u, v in rem_edges:
            candidates.append((cur_deg[u] + cur_deg[v], u, v))
        candidates.sort(key=lambda x: -x[0])
        
        # Try top seeds (with some randomness for diversity)
        best_bc = None
        best_size = 0
        top_k = min(max_seeds, len(candidates))
        
        for idx in range(top_k):
            if time.time() > deadline:
                break
            _, u, v = candidates[idx]
            # Try both orientations
            for left_seed, right_seed in [({u}, {v}), ({v}, {u})]:
                bc = _max_biclique_from_seed(left_seed, right_seed, rem_set, adj, n, deadline)
                if bc is None:
                    continue
                l, r = bc
                size = len(l) * len(r)
                if size > best_size:
                    best_size = size
                    best_bc = bc
        
        if best_bc is None:
            # Fallback: single edge
            u, v = rem_edges[0]
            best_bc = ({u}, {v})
        
        left, right = best_bc
        ce = _cross_edges_of_biclique(left, right, rem_set)
        if not ce:
            # Should not happen
            u, v = rem_edges[0]
            bicliques.append(({u}, {v}))
            remaining.discard(_ek((u, v)))
            continue
        
        bicliques.append((left, right))
        remaining -= ce
    
    return bicliques

def _try_merge(bicliques, edge_set, deadline):
    """Try to merge pairs of bicliques if compatible."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        bicliques = list(bicliques)
        n_bc = len(bicliques)
        for i in range(n_bc):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_bc):
                if time.time() > deadline:
                    break
                li, ri = bicliques[i]
                lj, rj = bicliques[j]
                
                # Try 4 combinations
                for combo in [(li | lj, ri | rj), (li | rj, ri | lj), (ri | lj, li | rj), (ri | rj, li | lj)]:
                    nl, nr = combo
                    if not nl or not nr or (nl & nr):
                        continue
                    # Check all cross edges exist
                    valid = True
                    for u in nl:
                        for v in nr:
                            if _ek((u, v)) not in edge_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    # Check cross edges of merged == union of cross edges
                    merged_cross = set()
                    for u in nl:
                        for v in nr:
                            merged_cross.add(_ek((u, v)))
                    cross_i = set()
                    for u in li:
                        for v in ri:
                            cross_i.add(_ek((u, v)))
                    cross_j = set()
                    for u in lj:
                        for v in rj:
                            cross_j.add(_ek((u, v)))
                    if merged_cross == cross_i | cross_j:
                        bicliques[i] = (nl, nr)
                        bicliques.pop(j)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return bicliques

def _validate(bicliques, all_edge_set):
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        if left & right:
            return False
        for u in left:
            for v in right:
                k = _ek((u, v))
                if k not in all_edge_set:
                    return False
                if k in covered:
                    return False
                covered.add(k)
    return covered == all_edge_set

def _local_improve(bicliques, edge_set, adj, n, deadline, max_iters=5000):
    """Local search: try to move or swap vertices between bicliques to reduce count."""
    bicliques = list(bicliques)
    improved = True
    iters = 0
    while improved and iters < max_iters and time.time() < deadline:
        improved = False
        iters += 1
        # First: try to absorb small bicliques into larger ones
        for i in range(len(bicliques)):
            if time.time() > deadline:
                return bicliques
            li, ri = bicliques[i]
            si = len(li) * len(ri)
            if si > 3:
                continue
            # Try to absorb biclique i into another biclique j
            for j in range(len(bicliques)):
                if i == j:
                    continue
                lj, rj = bicliques[j]
                for nl, nr in [(lj | li, rj), (lj, rj | ri), (lj | ri, rj | li), (lj | li, rj | ri)]:
                    if not nl or not nr or (nl & nr):
                        continue
                    valid = True
                    for u in nl:
                        for v in nr:
                            if _ek((u, v)) not in edge_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    new_cross = set(_ek((u, v)) for u in nl for v in nr)
                    # Check no overlap with other bicliques
                    overlap = False
                    for k in range(len(bicliques)):
                        if k == i or k == j:
                            continue
                        lk, rk = bicliques[k]
                        ck = set(_ek((u, v)) for u in lk for v in rk)
                        if new_cross & ck:
                            overlap = True
                            break
                    if overlap:
                        continue
                    # Must cover all edges of biclique i
                    old_cross = set(_ek((u, v)) for u in li for v in ri)
                    if not old_cross.issubset(new_cross):
                        continue
                    bicliques[j] = (nl, nr)
                    bicliques.pop(i)
                    improved = True
                    break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Try moving a vertex from left of i to right of j, or vice versa
        for i in range(len(bicliques)):
            if time.time() > deadline:
                return bicliques
            li, ri = bicliques[i]
            for j in range(len(bicliques)):
                if i == j:
                    continue
                lj, rj = bicliques[j]
                # Move v from li to rj
                for v in list(li):
                    new_li = li - {v}
                    new_rj = rj | {v}
                    if not new_li:
                        continue
                    if new_rj & new_li:
                        continue
                    if not lj.issubset(adj[v]):
                        continue
                    if not ri.issubset(adj[v]) or not all(adj[a] & ri == ri for a in new_li):
                        continue
                    if not all(adj[a] & new_rj == new_rj for a in lj):
                        continue
                    cross_i_new = set(_ek((a, b)) for a in new_li for b in ri)
                    cross_j_new = set(_ek((a, b)) for a in lj for b in new_rj)
                    if cross_i_new & cross_j_new:
                        continue
                    bicliques[i] = (new_li, ri)
                    bicliques[j] = (lj, new_rj)
                    if not new_li:
                        bicliques.pop(i)
                        if j > i:
                            j -= 1
                        if j >= len(bicliques):
                            break
                    improved = True
                    break
                if improved:
                    break
            if improved:
                break
        # Try swapping: v in li <-> w in rj (or other cross combos)
        if not improved:
            for i in range(len(bicliques)):
                if time.time() > deadline:
                    return bicliques
                li, ri = bicliques[i]
                for j in range(len(bicliques)):
                    if i == j:
                        continue
                    lj, rj = bicliques[j]
                    # Swap v from li with w from rj: v->rj, w->li
                    for v in list(li):
                        for w in list(rj):
                            new_li = (li - {v}) | {w}
                            new_rj = (rj - {w}) | {v}
                            if not new_li or not new_rj:
                                continue
                            if new_li & new_rj:
                                continue
                            if new_li & li == new_li and new_rj & rj == new_rj:
                                continue  # no actual change
                            # Check biclique i validity
                            if not all(adj[a] & ri == ri for a in new_li):
                                continue
                            # Check biclique j validity
                            if not all(adj[a] & new_rj == new_rj for a in lj):
                                continue
                            cross_i_new = set(_ek((a, b)) for a in new_li for b in ri)
                            cross_j_new = set(_ek((a, b)) for a in lj for b in new_rj)
                            if cross_i_new & cross_j_new:
                                continue
                            bicliques[i] = (new_li, ri)
                            bicliques[j] = (lj, new_rj)
                            improved = True
                            break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        # Also try merging
        bicliques = _try_merge(bicliques, edge_set, deadline)
    return bicliques

def _edge_reassign_polish(bicliques, edge_set, adj, n, deadline):
    """Try to eliminate small bicliques by reassigning their edges to other bicliques."""
    bicliques = list(bicliques)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Sort by size (smallest first) to try eliminating small ones
        sizes = [len(l) * len(r) for l, r in bicliques]
        order = sorted(range(len(bicliques)), key=lambda i: sizes[i])
        for si in order:
            if time.time() > deadline:
                break
            if si >= len(bicliques):
                continue
            li, ri = bicliques[si]
            # Get all edges in this biclique
            bc_edges = set(_ek((u, v)) for u in li for v in ri)
            # Try to find a single other biclique that can absorb all these edges
            absorbed = False
            for sj in range(len(bicliques)):
                if sj == si:
                    continue
                lj, rj = bicliques[sj]
                # Try 4 orientations: add li to lj, ri to rj; etc.
                for nl, nr in [(lj | li, rj), (lj, rj | ri), (lj | ri, rj | li), (lj | li, rj | ri)]:
                    if not nl or not nr or (nl & nr):
                        continue
                    # Check all cross edges exist in edge_set
                    valid = True
                    for u in nl:
                        for v in nr:
                            if _ek((u, v)) not in edge_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    # Check that the new cross edges cover bc_edges and don't overlap with other bicliques
                    new_cross = set(_ek((u, v)) for u in nl for v in nr)
                    # Must cover all edges of biclique si
                    if not bc_edges.issubset(new_cross):
                        continue
                    # Must not overlap with edges of other bicliques (except si which we're removing)
                    overlap = False
                    for sk in range(len(bicliques)):
                        if sk == si or sk == sj:
                            continue
                        lk, rk = bicliques[sk]
                        ck = set(_ek((u, v)) for u in lk for v in rk)
                        if new_cross & ck:
                            overlap = True
                            break
                    if overlap:
                        continue
                    # Valid: absorb si into sj
                    bicliques[sj] = (nl, nr)
                    bicliques.pop(si)
                    improved = True
                    absorbed = True
                    break
                if absorbed:
                    break
            if absorbed:
                break
        # Try splitting a biclique and redistributing edges to eliminate it
        if not improved:
            for si in range(len(bicliques)):
                if time.time() > deadline:
                    break
                li, ri = bicliques[si]
                si_size = len(li) * len(ri)
                if si_size < 2:
                    continue
                # Try removing one vertex from li and see if remaining + redistribution works
                for v in list(li):
                    new_li = li - {v}
                    if not new_li:
                        continue
                    # Edges that v was covering: (v, r) for r in ri
                    v_edges = set(_ek((v, r)) for r in ri)
                    # Remaining edges in biclique si
                    rem_edges = set(_ek((u, r)) for u in new_li for r in ri)
                    # Try to find where v_edges can go
                    all_other_cross = set()
                    for sj in range(len(bicliques)):
                        if sj == si:
                            continue
                        lj, rj = bicliques[sj]
                        all_other_cross |= set(_ek((u, v)) for u in lj for v in rj)
                    # Check if v_edges can be covered by expanding other bicliques
                    # This is complex; skip for now
                    pass
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    all_edge_set = set(_ek(e) for e in edges)
    adj = _adj(n, edges)
    
    rng = random.Random(42)
    
    # Multiple random restarts with different seeds
    best_bicliques = None
    best_count = float('inf')
    
    # First, a deterministic greedy with high max_seeds
    bc = _greedy_partition(all_edge_set, adj, n, deadline, rng, max_seeds=100)
    bc = _try_merge(bc, all_edge_set, deadline)
    if _validate(bc, all_edge_set):
        best_bicliques = bc
        best_count = len(bc)
    
    # More restarts with lower max_seeds but different random orderings
    remaining_time_budget = deadline - time.time()
    n_restarts = int(min(30, max(0, remaining_time_budget / 2)))
    
    for r in range(n_restarts):
        if time.time() > deadline - 5:
            break
        rng = random.Random(42 + r * 137)
        bc = _greedy_partition(all_edge_set, adj, n, deadline, rng, max_seeds=50 + r * 5)
        bc = _try_merge(bc, all_edge_set, deadline)
        if _validate(bc, all_edge_set):
            if len(bc) < best_count:
                best_bicliques = bc
                best_count = len(bc)
    
    # Local improvement on best
    if best_bicliques is not None:
        best_bicliques = _local_improve(best_bicliques, all_edge_set, adj, n, deadline)
        if not _validate(best_bicliques, all_edge_set):
            # Revert to pre-local-improvement if broken
            bc2 = _greedy_partition(all_edge_set, adj, n, deadline, random.Random(42), max_seeds=100)
            bc2 = _try_merge(bc2, all_edge_set, deadline)
            if _validate(bc2, all_edge_set):
                best_bicliques = bc2
            else:
                # Fallback: each edge as its own biclique
                best_bicliques = [({u}, {v}) for u, v in edges]
        # Additional local improvement pass
        if time.time() < deadline - 10:
            best_bicliques = _local_improve(best_bicliques, all_edge_set, adj, n, deadline)
            if not _validate(best_bicliques, all_edge_set):
                bc3 = _greedy_partition(all_edge_set, adj, n, deadline, random.Random(42), max_seeds=100)
                bc3 = _try_merge(bc3, all_edge_set, deadline)
                if _validate(bc3, all_edge_set):
                    best_bicliques = bc3
                else:
                    best_bicliques = [({u}, {v}) for u, v in edges]
    
    if best_bicliques is None:
        # Fallback: each edge as its own biclique
        best_bicliques = [({u}, {v}) for u, v in edges]
    
    # Post-hoc edge reassignment polish: try to steal edges from smaller bicliques
    # into larger ones, eliminating small bicliques entirely
    if time.time() < deadline - 5:
        best_bicliques = _edge_reassign_polish(best_bicliques, all_edge_set, adj, n, deadline)
    
    # Final validation
    if not _validate(best_bicliques, all_edge_set):
        # Ultimate fallback
        best_bicliques = [({u}, {v}) for u, v in edges]
    
    # Convert to proper format
    result = []
    for left, right in best_bicliques:
        result.append([list(left), list(right)])
    
    return result
# EVOLVE-BLOCK-END
