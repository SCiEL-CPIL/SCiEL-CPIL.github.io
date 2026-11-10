"""Combined program: hexagonal core + branch-and-bound + local search for 32 points on triangular lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math
import sys
from functools import lru_cache

def _dist2(a1, b1, a2, b2):
    """Squared distance on triangular lattice."""
    da = a1 - a2
    db = b1 - b2
    return da*da + da*db + db*db

def _count_edges(points):
    """Count unit-distance pairs among given (a,b) points."""
    point_set = set(tuple(p) for p in points)
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    edge_count = 0
    for a, b in point_set:
        for da, db in neighbors:
            if (a+da, b+db) in point_set:
                edge_count += 1
    return edge_count // 2

def _hexagon_points(r):
    """All points with hex distance <= r from origin: max(|a|,|b|,|a+b|) <= r."""
    pts = []
    for a in range(-r, r+1):
        for b in range(-r, r+1):
            if max(abs(a), abs(b), abs(a+b)) <= r:
                pts.append((a, b))
    return pts

NEIGHBORS = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]

def _precompute_adjacency(core, ring):
    """Precompute adjacency between core points, ring points, and core-ring."""
    core_set = set(core)
    ring_set = set(ring)
    all_points = core + ring
    
    # Build adjacency lists
    adj_core = {}
    adj_ring = {}
    adj_core_ring = {}
    
    for i, p in enumerate(core):
        adj_core[i] = set()
        for j, q in enumerate(core):
            if i != j and _dist2(p[0], p[1], q[0], q[1]) == 1:
                adj_core[i].add(j)
        for k, r in enumerate(ring):
            if _dist2(p[0], p[1], r[0], r[1]) == 1:
                adj_core_ring.setdefault(i, set()).add(k)
    
    for k, p in enumerate(ring):
        adj_ring[k] = set()
        for l, q in enumerate(ring):
            if k != l and _dist2(p[0], p[1], q[0], q[1]) == 1:
                adj_ring[k].add(l)
        for i, c in enumerate(core):
            if _dist2(p[0], p[1], c[0], c[1]) == 1:
                adj_core_ring.setdefault(i, set()).add(k)
    
    # Count edges within core
    core_edges = sum(len(s) for s in adj_core.values()) // 2
    
    # For each ring point, count its degree to core
    ring_to_core_degree = [0] * len(ring)
    for i in range(len(core)):
        for k in adj_core_ring.get(i, set()):
            ring_to_core_degree[k] += 1
    
    return adj_core, adj_ring, adj_core_ring, core_edges, ring_to_core_degree

def _upper_bound(selected_ring, adj_ring, ring_to_core_degree, remaining_indices, n_remaining):
    """Compute an upper bound on additional edges from selecting n_remaining from remaining_indices.
    
    The upper bound considers:
    1. Edges from new points to core (at most sum of top n_remaining ring_to_core degrees)
    2. Edges among new points (at most the max possible, bounded by n_remaining * 3 since each point has max 6 neighbors but we need to be conservative)
    3. Edges from new points to already-selected ring points
    """
    if n_remaining == 0:
        return 0
    
    # Sort remaining by their core degree (descending) for optimistic bound
    remaining_sorted = sorted(remaining_indices, key=lambda k: -ring_to_core_degree[k])
    take = remaining_sorted[:n_remaining]
    
    # Bound 1: edges to core
    core_bound = sum(ring_to_core_degree[k] for k in take)
    
    # Bound 2: edges among the selected new points
    # Each point can have at most 6 neighbors, but in practice on the ring,
    # max neighbors among ring points is limited. Use a conservative bound:
    # The maximum number of edges among m points where each has degree at most d
    # is at most m*d/2. But we can do better by checking actual adjacencies.
    # For a tighter bound, compute the actual max edges among any subset of size n_remaining
    # from the remaining set, but that's expensive. Use a simple bound:
    # Each new point can connect to at most min(n_remaining-1, 6) other new points
    # So max edges among new points <= n_remaining * min(n_remaining-1, 6) / 2
    ring_ring_bound = n_remaining * min(n_remaining - 1, 6) // 2
    
    # Bound 3: edges from new points to already-selected ring points
    # Each new point can connect to at most len(selected_ring) already-selected points
    # But actually, we should check how many of the already-selected are adjacent
    # For a conservative bound, assume each new point connects to all selected
    # But that's too optimistic. Let's use: for each candidate, count actual adjacencies
    # to selected, and take the top n_remaining
    if selected_ring:
        sel_set = set(selected_ring)
        candidate_scores = []
        for k in remaining_indices:
            # Count adjacencies to selected ring points
            adj_count = sum(1 for s in sel_set if s in adj_ring.get(k, set()))
            candidate_scores.append(adj_count)
        candidate_scores.sort(reverse=True)
        cross_bound = sum(candidate_scores[:n_remaining])
    else:
        cross_bound = 0
    
    return core_bound + ring_ring_bound + cross_bound

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill
    
    # Hexagon of radius 2 has 19 points
    core = _hexagon_points(2)
    # Hexagon of radius 3 has 37 points
    all_r3 = _hexagon_points(3)
    ring3 = [p for p in all_r3 if max(abs(p[0]), abs(p[1]), abs(p[0]+p[1])) == 3]
    
    n_need = 32 - len(core)  # 32 - 19 = 13
    n_ring = len(ring3)  # 18
    
    # Precompute adjacency
    adj_core, adj_ring, adj_core_ring, core_edges, ring_to_core_degree = _precompute_adjacency(core, ring3)
    
    # Greedy initial selection from ring3
    selected = set()
    core_set = set(core)
    for _ in range(n_need):
        best_gain = -1
        best_idx = -1
        for i, p in enumerate(ring3):
            if i in selected:
                continue
            gain = 0
            # Count adjacencies to core
            gain += ring_to_core_degree[i]
            # Count adjacencies to already selected ring points
            for j in selected:
                if j in adj_ring.get(i, set()):
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx >= 0:
            selected.add(best_idx)
    
    best_selection = sorted(selected)
    current_points = core + [ring3[i] for i in best_selection]
    best_edges = _count_edges(current_points)
    
    # Branch and bound search
    if time.time() < deadline - 10.0:
        # State: (current_selection_set, current_edge_count, remaining_indices_tuple, nodes_remaining)
        # We'll do a DFS with pruning
        
        best_found = best_edges
        best_sel_found = set(best_selection)
        
        # For efficiency, we'll use iterative deepening or just a bounded DFS
        # The search space is C(18, 13) = 8568, which is manageable
        
        # Precompute all combinations
        from itertools import combinations
        
        # We'll iterate through combinations in order of decreasing potential
        # First, score each combination by a heuristic
        all_combos = list(combinations(range(n_ring), n_need))
        
        # Score each combination for ordering
        def score_combo(combo):
            s = set(combo)
            edges = 0
            # Edges to core
            for k in combo:
                edges += ring_to_core_degree[k]
            # Edges among combo
            for i, k in enumerate(combo):
                for l in combo[i+1:]:
                    if l in adj_ring.get(k, set()):
                        edges += 1
            return edges
        
        # Sort by score descending
        all_combos.sort(key=score_combo, reverse=True)
        
        # Evaluate top combinations
        evaluated = 0
        for combo in all_combos:
            if time.time() > deadline - 10.0:
                break
            if evaluated > 5000:  # Limit to avoid timeout
                break
            
            s = set(combo)
            edges = 0
            # Edges to core
            for k in combo:
                edges += ring_to_core_degree[k]
            # Edges among combo
            for i, k in enumerate(combo):
                for l in combo[i+1:]:
                    if l in adj_ring.get(k, set()):
                        edges += 1
            
            total = core_edges + edges
            if total > best_found:
                best_found = total
                best_sel_found = s
                if best_found >= 76:
                    break
            evaluated += 1
        
        if best_found > best_edges:
            best_edges = best_found
            best_selection = sorted(best_sel_found)
    
    # Local search refinement if we haven't reached 76
    if best_edges < 76 and time.time() < deadline - 5.0:
        current_selection = set(best_selection)
        current_edges = best_edges
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for keep in list(current_selection):
                if time.time() >= deadline - 5.0:
                    break
                keep_pt = ring3[keep]
                for add in range(n_ring):
                    if add in current_selection:
                        continue
                    add_pt = ring3[add]
                    
                    # Compute delta
                    lost = 0
                    for da, db in NEIGHBORS:
                        np = (keep_pt[0]+da, keep_pt[1]+db)
                        # Check if np is in current point set (core + selected ring)
                        if np in core_set:
                            lost += 1
                        else:
                            # Check if np is a selected ring point
                            for idx, rp in enumerate(ring3):
                                if rp == np and idx in current_selection:
                                    lost += 1
                                    break
                    
                    gained = 0
                    for da, db in NEIGHBORS:
                        np = (add_pt[0]+da, add_pt[1]+db)
                        if np in core_set:
                            gained += 1
                        else:
                            for idx, rp in enumerate(ring3):
                                if rp == np:
                                    if idx in current_selection and idx != keep:
                                        gained += 1
                                    elif idx == keep:
                                        gained += 1  # Edge restored
                                    break
                    
                    delta = gained - lost
                    e = current_edges + delta
                    if e > current_edges:
                        current_selection = current_selection - {keep} | {add}
                        current_edges = e
                        improved = True
                        if current_edges >= 76:
                            break
                if current_edges >= 76:
                    break
            if current_edges >= 76:
                break
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_selection = sorted(current_selection)
    
    # 2-swap local search
    if best_edges < 76 and time.time() < deadline - 5.0:
        current_selection = set(best_selection)
        current_edges = best_edges
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            sel_list = sorted(current_selection)
            for i in range(len(sel_list)):
                for j in range(i+1, len(sel_list)):
                    if time.time() >= deadline - 5.0:
                        break
                    keep1, keep2 = sel_list[i], sel_list[j]
                    keep1_pt = ring3[keep1]
                    keep2_pt = ring3[keep2]
                    for add1 in range(n_ring):
                        if add1 in current_selection:
                            continue
                        add1_pt = ring3[add1]
                        for add2 in range(n_ring):
                            if add2 in current_selection or add2 == add1:
                                continue
                            add2_pt = ring3[add2]
                            
                            # Compute delta for removing keep1, keep2 and adding add1, add2
                            lost = 0
                            for da, db in NEIGHBORS:
                                np1 = (keep1_pt[0]+da, keep1_pt[1]+db)
                                np2 = (keep2_pt[0]+da, keep2_pt[1]+db)
                                if np1 in core_set:
                                    lost += 1
                                else:
                                    for idx, rp in enumerate(ring3):
                                        if rp == np1 and idx in current_selection and idx not in (keep1, keep2):
                                            lost += 1
                                            break
                                if np2 in core_set:
                                    lost += 1
                                else:
                                    for idx, rp in enumerate(ring3):
                                        if rp == np2 and idx in current_selection and idx not in (keep1, keep2):
                                            lost += 1
                                            break
                            
                            # Edge between keep1 and keep2
                            if _dist2(keep1_pt[0], keep1_pt[1], keep2_pt[0], keep2_pt[1]) == 1:
                                lost += 1
                            
                            gained = 0
                            for da, db in NEIGHBORS:
                                np1 = (add1_pt[0]+da, add1_pt[1]+db)
                                np2 = (add2_pt[0]+da, add2_pt[1]+db)
                                if np1 in core_set:
                                    gained += 1
                                else:
                                    for idx, rp in enumerate(ring3):
                                        if rp == np1:
                                            if idx in current_selection and idx not in (keep1, keep2):
                                                gained += 1
                                            elif idx in (keep1, keep2):
                                                gained += 1
                                            break
                                if np2 in core_set:
                                    gained += 1
                                else:
                                    for idx, rp in enumerate(ring3):
                                        if rp == np2:
                                            if idx in current_selection and idx not in (keep1, keep2):
                                                gained += 1
                                            elif idx in (keep1, keep2):
                                                gained += 1
                                            break
                            
                            # Edge between add1 and add2
                            if _dist2(add1_pt[0], add1_pt[1], add2_pt[0], add2_pt[1]) == 1:
                                gained += 1
                            
                            delta = gained - lost
                            e = current_edges + delta
                            if e > current_edges:
                                current_selection = current_selection - {keep1, keep2} | {add1, add2}
                                current_edges = e
                                improved = True
                                if current_edges >= 76:
                                    break
                        if current_edges >= 76:
                            break
                    if current_edges >= 76:
                        break
                if current_edges >= 76:
                    break
            if current_edges >= 76:
                break
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_selection = sorted(current_selection)
    
    # Build final result
    final_points = core + [ring3[i] for i in best_selection]
    
    # Validation
    assert len(final_points) == 32, f"Got {len(final_points)} points"
    assert len(set(tuple(p) for p in final_points)) == 32, "Points not distinct"
    for a, b in final_points:
        assert abs(a) <= 64 and abs(b) <= 64, f"Point ({a},{b}) out of bounds"
    
    result = [[a, b] for a, b in final_points]
    return result
# EVOLVE-BLOCK-END
