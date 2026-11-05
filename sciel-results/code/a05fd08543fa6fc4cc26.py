import time
import random
import sys

sys.setrecursionlimit(10000)

_N = 20
_COLORS = 20
_ALL_EDGES = [(i, j) for i in range(_N) for j in range(i + 1, _N)]
_EDGE_IDX = {(i, j): idx for idx, (i, j) in enumerate(_ALL_EDGES)}
_N_EDGES = len(_ALL_EDGES)

_CYCLES = []
_seen_cycles = set()

for i in range(_N):
    for j in range(i + 1, _N):
        for k in range(j + 1, _N):
            for l in range(k + 1, _N):
                e1 = _EDGE_IDX[(i, j)]
                e2 = _EDGE_IDX[(j, k)]
                e3 = _EDGE_IDX[(k, l)]
                e4 = _EDGE_IDX[(l, i)]
                cycle1 = tuple(sorted([e1, e2, e3, e4]))
                if cycle1 not in _seen_cycles:
                    _seen_cycles.add(cycle1)
                    _CYCLES.append(cycle1)
                
                e1 = _EDGE_IDX[(i, j)]
                e2 = _EDGE_IDX[(j, l)]
                e3 = _EDGE_IDX[(l, k)]
                e4 = _EDGE_IDX[(k, i)]
                cycle2 = tuple(sorted([e1, e2, e3, e4]))
                if cycle2 not in _seen_cycles:
                    _seen_cycles.add(cycle2)
                    _CYCLES.append(cycle2)
                
                e1 = _EDGE_IDX[(i, k)]
                e2 = _EDGE_IDX[(k, j)]
                e3 = _EDGE_IDX[(j, l)]
                e4 = _EDGE_IDX[(l, i)]
                cycle3 = tuple(sorted([e1, e2, e3, e4]))
                if cycle3 not in _seen_cycles:
                    _seen_cycles.add(cycle3)
                    _CYCLES.append(cycle3)

_EDGE_TO_CYCLES = [[] for _ in range(_N_EDGES)]
for c_idx, cycle in enumerate(_CYCLES):
    for e_idx in cycle:
        _EDGE_TO_CYCLES[e_idx].append(c_idx)

# Precompute for each edge, the other 3 edges in each cycle it belongs to
# This allows fast conflict checking
_EDGE_CYCLE_NEIGHBORS = [None] * _N_EDGES
for e_idx in range(_N_EDGES):
    neighbors = []
    for c_idx in _EDGE_TO_CYCLES[e_idx]:
        cycle = _CYCLES[c_idx]
        others = [e for e in cycle if e != e_idx]
        neighbors.append(others)
    _EDGE_CYCLE_NEIGHBORS[e_idx] = neighbors

def _is_cycle_conflict(others, colors):
    c1 = colors[others[0]]
    c2 = colors[others[1]]
    c3 = colors[others[2]]
    # The 4th color is the one being changed, but for checking if the *rest* allows a conflict,
    # we just need to know if the existing 3 are distinct?
    # No, the cycle is conflict if ALL 4 are not distinct.
    # If we change edge e to color C, the cycle is conflict if
    # C is in {c1, c2, c3} OR (c1==c2 or c2==c3 or c1==c3).
    # Wait, if c1==c2, the cycle is ALREADY a conflict regardless of C?
    # No, "Every 4-cycle must receive four distinct colours".
    # If c1==c2, then the set of colors has size < 4, so it IS a conflict.
    # So, a cycle is a conflict if:
    # 1. The 3 other edges already have a repeated color.
    # 2. OR, the new color C matches one of the 3.
    
    # Let's precompute for each edge e and each cycle C containing e:
    # Does the cycle C have a conflict among the OTHER 3 edges?
    
    return len({c1, c2, c3}) < 3

# Precompute "static_conflict" for each edge-cycle pair
# static_conflict[e_idx][c_idx_in_list] = True if the other 3 edges in that cycle already have duplicate colors
_STATIC_CONFLICTS = [None] * _N_EDGES
for e_idx in range(_N_EDGES):
    statics = []
    for others in _EDGE_CYCLE_NEIGHBORS[e_idx]:
        statics.append(_is_cycle_conflict(others, [0]*_N_EDGES)) # Colors don't matter for this check? 
        # Wait, _is_cycle_conflict checks if len({c1,c2,c3}) < 3.
        # This depends on the actual colors of the other edges.
        # So we cannot precompute this statically. It changes as we change colors of other edges.
        pass

# So we must check dynamically.
# But we can optimize the check.

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    rng = random.Random(42)
    
    best_edges_count = 0
    best_solution = []
    
    # Local Search on K_20
    while time.time() < deadline:
        colors = [rng.randint(0, _COLORS - 1) for _ in range(_N_EDGES)]
        
        # Calculate initial conflicts
        current_conflicts = 0
        for cycle in _CYCLES:
            if len({colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]}) < 4:
                current_conflicts += 1
        
        if current_conflicts == 0:
            return [[u, v, colors[idx]] for idx, (u, v) in enumerate(_ALL_EDGES)]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            e_idx = rng.randint(0, _N_EDGES - 1)
            old_color = colors[e_idx]
            
            best_new_color = old_color
            best_conflicts = current_conflicts
            
            for c in range(_COLORS):
                if c == old_color:
                    continue
                
                # Calculate delta in conflicts
                delta = 0
                for others in _EDGE_CYCLE_NEIGHBORS[e_idx]:
                    c1 = colors[others[0]]
                    c2 = colors[others[1]]
                    c3 = colors[others[2]]
                    
                    # Was it a conflict before?
                    # Conflict if len({old_color, c1, c2, c3}) < 4
                    was_conflict = len({old_color, c1, c2, c3}) < 4
                    
                    # Is it a conflict after?
                    # Conflict if len({c, c1, c2, c3}) < 4
                    is_conflict = len({c, c1, c2, c3}) < 4
                    
                    if was_conflict and not is_conflict:
                        delta -= 1
                    elif not was_conflict and is_conflict:
                        delta += 1
                
                new_conflicts = current_conflicts + delta
                if new_conflicts < best_conflicts:
                    best_conflicts = new_conflicts
                    best_new_color = c
                    if new_conflicts == 0:
                        colors[e_idx] = c
                        return [[u, v, colors[idx]] for idx, (u, v) in enumerate(_ALL_EDGES)]
            
            colors[e_idx] = best_new_color
            if best_conflicts < current_conflicts:
                current_conflicts = best_conflicts
                improved = True
            
            if not improved:
                # Random restart or noise
                # Let's do a random noise step to escape local minima
                e_idx = rng.randint(0, _N_EDGES - 1)
                colors[e_idx] = rng.randint(0, _COLORS - 1)
                # Recalculate conflicts for this edge change?
                # For simplicity, just recalculate total if we do noise?
                # Or just mark improved=True to continue loop, and rely on next step to fix.
                # But we need accurate current_conflicts.
                # Let's just restart the whole local search if stuck.
                break
        
        # If stuck, restart
        # We don't save anything here because K_20 is invalid.
        
    # Fallback to greedy if K_20 failed
    present_edges = set()
    colors_map = {}
    order = list(range(_N_EDGES))
    rng.shuffle(order)
    
    for e_idx in order:
        if time.time() > deadline:
            break
        color_order = list(range(_COLORS))
        rng.shuffle(color_order)
        for c in color_order:
            if _is_valid_greedy(e_idx, c, present_edges, colors_map):
                present_edges.add(e_idx)
                colors_map[e_idx] = c
                break
    
    if len(present_edges) > best_edges_count:
        best_edges_count = len(present_edges)
        best_solution = [[_ALL_EDGES[e][0], _ALL_EDGES[e][1], colors_map[e]] for e in present_edges]
        
    return best_solution

def _is_valid_greedy(edge_idx, color, present_edges, colors):
    for c_idx in _EDGE_TO_CYCLES[edge_idx]:
        cycle = _CYCLES[c_idx]
        other_edges = [e for e in cycle if e != edge_idx]
        if all(e in present_edges for e in other_edges):
            c1 = colors[other_edges[0]]
            c2 = colors[other_edges[1]]
            c3 = colors[other_edges[2]]
            if len({c1, c2, c3, color}) < 4:
                return False
    return True
