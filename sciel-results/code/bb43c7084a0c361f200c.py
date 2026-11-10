"""Improved program for edge-colourings of K16 with every K4 seeing five colours."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _edge_index(i, j):
    """Return linear index for edge (i,j) in lexicographic order."""
    return i * 15 - i * (i - 1) // 2 + (j - i - 1)

def _build_k4_edges():
    """Precompute all K4 edge index sets."""
    k4_edges = []
    for quad in combinations(range(16), 4):
        a, b, c, d = quad
        k4_edges.append((
            _edge_index(a, b), _edge_index(a, c), _edge_index(a, d),
            _edge_index(b, c), _edge_index(b, d), _edge_index(c, d)
        ))
    return k4_edges

def _is_valid(coloring, k4_edges):
    """Check if every K4 has at least 5 distinct colours."""
    for edges in k4_edges:
        c0 = coloring[edges[0]]
        c1 = coloring[edges[1]]
        c2 = coloring[edges[2]]
        c3 = coloring[edges[3]]
        c4 = coloring[edges[4]]
        c5 = coloring[edges[5]]
        # Count distinct colours without building a set
        distinct = 1
        if c1 != c0:
            distinct += 1
        if c2 != c0 and c2 != c1:
            distinct += 1
        if c3 != c0 and c3 != c1 and c3 != c2:
            distinct += 1
        if distinct >= 5:
            continue
        if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3:
            distinct += 1
        if distinct >= 5:
            continue
        if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4:
            distinct += 1
        if distinct < 5:
            return False
    return True

def _count_colors(coloring):
    return len(set(coloring))

def _try_merge(coloring, target, other, k4_edges):
    """Try to recolor all edges of 'target' to 'other'. Return new coloring or None."""
    new_coloring = coloring[:]
    changed = False
    for i in range(len(new_coloring)):
        if new_coloring[i] == target:
            new_coloring[i] = other
            changed = True
    if not changed:
        return None
    if _is_valid(new_coloring, k4_edges):
        return new_coloring
    return None

def _try_single_recolor(coloring, edge_idx, new_color, k4_edges):
    """Try to recolor a single edge to new_color. Return new coloring or None."""
    if coloring[edge_idx] == new_color:
        return None
    old_color = coloring[edge_idx]
    coloring[edge_idx] = new_color
    valid = _is_valid(coloring, k4_edges)
    coloring[edge_idx] = old_color
    if valid:
        return coloring[:]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s limit
    
    k4_edges = _build_k4_edges()
    
    # Initial coloring: (i+j) mod 17
    best = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    best_colors = _count_colors(best)
    
    # Verify initial coloring is valid
    if not _is_valid(best, k4_edges):
        # Fallback: use a simple valid coloring
        # Assign each edge a unique color based on (i, j) pair
        best = [_edge_index(i, j) for i in range(16) for j in range(i + 1, 16)]
        best_colors = _count_colors(best)
    
    # Phase 1: Systematic color merging
    # Try to merge colors by frequency (least frequent first)
    current = best[:]
    current_colors = best_colors
    
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        used_colors = sorted(set(current))
        if len(used_colors) <= current_colors:
            break
            
        # Sort by frequency (ascending) - try to eliminate rare colors first
        freq = {}
        for c in current:
            freq[c] = freq.get(c, 0) + 1
        sorted_colors = sorted(used_colors, key=lambda c: freq.get(c, 0))
        
        improved = False
        for target in sorted_colors:
            if time.time() > deadline:
                break
            if target not in freq or freq[target] == 0:
                continue
                
            # Try to merge target into each other color
            for other in sorted_colors:
                if other == target or other not in freq or freq[other] == 0:
                    continue
                
                result = _try_merge(current, target, other, k4_edges)
                if result is not None:
                    new_count = _count_colors(result)
                    if new_count < current_colors:
                        current = result
                        current_colors = new_count
                        best = current[:]
                        best_colors = new_count
                        improved = True
                        break
            
            if improved:
                break
        
        if not improved:
            break
    
    # Phase 2: Single-edge recoloring to enable further merges
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        used_colors = sorted(set(current))
        if len(used_colors) <= current_colors:
            break
            
        freq = {}
        for c in current:
            freq[c] = freq.get(c, 0) + 1
        sorted_colors = sorted(used_colors, key=lambda c: freq.get(c, 0))
        
        improved = False
        for target in sorted_colors:
            if time.time() > deadline:
                break
            if target not in freq or freq[target] == 0:
                continue
            
            # Try to recolor individual edges of target to other colors
            edges_with_target = [i for i, c in enumerate(current) if c == target]
            random.shuffle(edges_with_target)
            
            for ei in edges_with_target:
                if time.time() > deadline:
                    break
                
                for other in sorted_colors:
                    if other == target or other not in freq or freq[other] == 0:
                        continue
                    
                    # Try recoloring this edge to other
                    old = current[ei]
                    current[ei] = other
                    if _is_valid(current, k4_edges):
                        # Check if this enables a merge
                        new_freq = {}
                        for c in current:
                            new_freq[c] = new_freq.get(c, 0) + 1
                        
                        # Try to merge target (now possibly with fewer edges)
                        if target in new_freq and new_freq[target] > 0:
                            for merge_target in sorted(new_freq.keys(), key=lambda x: new_freq[x]):
                                if merge_target == other or new_freq[merge_target] == 0:
                                    continue
                                result = _try_merge(current, merge_target, other, k4_edges)
                                if result is not None:
                                    new_count = _count_colors(result)
                                    if new_count < current_colors:
                                        current = result
                                        current_colors = new_count
                                        best = current[:]
                                        best_colors = new_count
                                        improved = True
                                        break
                        if improved:
                            break
                    else:
                        current[ei] = old
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    # Phase 3: Randomized restarts with perturbation
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Start from best or a perturbed version
        if restart == 0:
            current = best[:]
        else:
            current = best[:]
            # Perturb: randomly reassign a few edges
            for _ in range(random.randint(1, 5)):
                ei = random.randint(0, 119)
                new_c = random.randint(0, best_colors - 1)
                current[ei] = new_c
            if not _is_valid(current, k4_edges):
                current = best[:]
        
        current_colors = _count_colors(current)
        
        # Try merging colors
        improved = False
        for iteration in range(20):
            if time.time() > deadline:
                break
            
            used_colors = sorted(set(current))
            if len(used_colors) <= current_colors:
                break
                
            freq = {}
            for c in current:
                freq[c] = freq.get(c, 0) + 1
            sorted_colors = sorted(used_colors, key=lambda c: freq.get(c, 0))
            
            for target in sorted_colors:
                if time.time() > deadline:
                    break
                if target not in freq or freq[target] == 0:
                    continue
                    
                for other in sorted_colors:
                    if other == target or other not in freq or freq[other] == 0:
                        continue
                    
                    result = _try_merge(current, target, other, k4_edges)
                    if result is not None:
                        new_count = _count_colors(result)
                        if new_count < current_colors:
                            current = result
                            current_colors = new_count
                            improved = True
                            break
                
                if improved:
                    break
            
            if not improved:
                break
        
        if current_colors < best_colors:
            best = current[:]
            best_colors = current_colors
    
    return best
# EVOLVE-BLOCK-END
