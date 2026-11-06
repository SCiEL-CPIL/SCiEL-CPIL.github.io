import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # Targets to try in order of preference
    targets = [17, 16, 15, 14, 13, 12]
    
    best_n = 0
    best_colors = []

    for n in targets:
        if time.time() > deadline - 5:  # Leave 5s for final return
            break
            
        # 1. Try backtracking with bitset C6 detection
        remaining_time = deadline - time.time() - 2
        if remaining_time < 5:
            continue
            
        # Allocate time based on n
        if n == 17:
            bt_time = min(remaining_time, 50)
        elif n == 16:
            bt_time = min(remaining_time, 30)
        else:
            bt_time = min(remaining_time, 15)
            
        bt_colors = run_backtracking(n, start_time + bt_time)
        if bt_colors is not None:
            return {'n': n, 'colors': bt_colors}

    # Fallback: Return a small valid coloring
    n_fallback = 5
    colors_fallback = [0] * (n_fallback * (n_fallback - 1) // 2)
    return {'n': n_fallback, 'colors': colors_fallback}

def run_backtracking(n, deadline):
    """Backtracking search with bitset-based C6 detection."""
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute edge index
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
    
    # Precompute all 6-cycles as edge index tuples
    cycles = []
    seen = set()
    for verts in combinations(range(n), 6):
        v0 = verts[0]
        rest = list(verts[1:])
        for perm in __import__('itertools').permutations(rest):
            if perm[0] > perm[-1]:
                continue
            cycle_verts = [v0] + list(perm)
            edge_indices = []
            valid = True
            for k in range(6):
                u = cycle_verts[k]
                v = cycle_verts[(k + 1) % 6]
                if u > v:
                    u, v = v, u
                if (u, v) not in edge_idx:
                    valid = False
                    break
                edge_indices.append(edge_idx[(u, v)])
            if not valid or len(set(edge_indices)) != 6:
                continue
            key = tuple(sorted(edge_indices))
            if key not in seen:
                seen.add(key)
                cycles.append(key)
    
    if not cycles:
        return [0] * num_edges
    
    # Convert to numpy for fast operations
    cycles_np = np.array(cycles, dtype=np.int32)
    
    colors = [-1] * num_edges
    # For each color, maintain a set of monochromatic cycles
    mono_cycles = [set() for _ in range(4)]
    
    # Map edge to list of cycles it belongs to
    edge_to_cycles = [[] for _ in range(num_edges)]
    for ci, cycle in enumerate(cycles):
        for e in cycle:
            edge_to_cycles[e].append(ci)
    
    def backtrack(edge_pos):
        if time.time() > deadline:
            return False
        if edge_pos == num_edges:
            return True
        
        # Try colors 0-3
        for color in range(4):
            # Check if assigning this color creates a monochromatic cycle
            # Only need to check cycles where all other 5 edges are already this color
            conflict = False
            for ci in edge_to_cycles[edge_pos]:
                cycle = cycles[ci]
                # Check if all other edges in this cycle are already color
                other_edges = [e for e in cycle if e != edge_pos]
                all_other_same = all(colors[e] == color for e in other_edges)
                if all_other_same:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Assign color
            colors[edge_pos] = color
            # Add cycles where this edge is the last one to be colored
            for ci in edge_to_cycles[edge_pos]:
                cycle = cycles[ci]
                other_edges = [e for e in cycle if e != edge_pos]
                if all(colors[e] == color for e in other_edges):
                    mono_cycles[color].add(ci)
            
            if backtrack(edge_pos + 1):
                return True
            
            # Undo
            colors[edge_pos] = -1
            for ci in edge_to_cycles[edge_pos]:
                cycle = cycles[ci]
                other_edges = [e for e in cycle if e != edge_pos]
                if all(colors[e] == color for e in other_edges):
                    mono_cycles[color].discard(ci)
        
        return False
    
    # Use randomized edge ordering to help break symmetries
    random.seed(42)
    
    # Try a few random edge orderings
    for attempt in range(20):
        if time.time() > deadline:
            break
        if attempt == 0:
            order = list(range(num_edges))
        else:
            order = list(range(num_edges))
            random.shuffle(order)
        
        # Reorder edges
        reordered_edges = [edges[i] for i in order]
        reordered_edge_idx = {}
        for new_idx, old_idx in enumerate(order):
            reordered_edge_idx[edges[old_idx]] = new_idx
        
        # Rebuild edge_to_cycles for reordered edges
        reordered_edge_to_cycles = [[] for _ in range(num_edges)]
        for ci, cycle in enumerate(cycles):
            for e in cycle:
                reordered_edge_to_cycles[reordered_edge_idx[edges[e]]].append(ci)
        
        # Reset colors
        colors = [-1] * num_edges
        mono_cycles = [set() for _ in range(4)]
        
        # Backtrack in reordered sequence
        def backtrack_reordered(pos):
            if time.time() > deadline:
                return False
            if pos == num_edges:
                return True
            
            orig_edge = order[pos]
            
            for color in range(4):
                conflict = False
                for ci in reordered_edge_to_cycles[pos]:
                    cycle = cycles[ci]
                    other_edges = [e for e in cycle if e != orig_edge]
                    all_other_same = all(colors[e] == color for e in other_edges)
                    if all_other_same:
                        conflict = True
                        break
                
                if conflict:
                    continue
                
                colors[orig_edge] = color
                for ci in reordered_edge_to_cycles[pos]:
                    cycle = cycles[ci]
                    other_edges = [e for e in cycle if e != orig_edge]
                    if all(colors[e] == color for e in other_edges):
                        mono_cycles[color].add(ci)
                
                if backtrack_reordered(pos + 1):
                    return True
                
                colors[orig_edge] = -1
                for ci in reordered_edge_to_cycles[pos]:
                    cycle = cycles[ci]
                    other_edges = [e for e in cycle if e != orig_edge]
                    if all(colors[e] == color for e in other_edges):
                        mono_cycles[color].discard(ci)
            
            return False
        
        if backtrack_reordered(0):
            # Convert back to original order
            result = [0] * num_edges
            for orig_edge in range(num_edges):
                result[orig_edge] = colors[orig_edge]
            return result
    
    return None
