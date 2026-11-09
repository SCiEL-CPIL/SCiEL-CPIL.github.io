import time
import random
import numpy as np

N = 20
E_TOTAL = N * (N - 1) // 2  # 190
C_TOTAL = 20 * 19 * 18 * 17 // 24  # 4845

# Precompute all edges
_all_edges = []
for u in range(N):
    for v in range(u + 1, N):
        _all_edges.append((u, v))

# Precompute all 4-cycles as tuples of 4 edge indices
# For each set of 4 vertices, there are 3 distinct 4-cycles
import itertools

def _build_cycles():
    edge_index = {}
    for i, (u, v) in enumerate(_all_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        cycles.append((edge_index[(a,b)], edge_index[(b,c)], edge_index[(c,d)], edge_index[(d,a)]))
        # Cycle 2: a-b-d-c-a
        cycles.append((edge_index[(a,b)], edge_index[(b,d)], edge_index[(d,c)], edge_index[(c,a)]))
        # Cycle 3: a-c-b-d-a
        cycles.append((edge_index[(a,c)], edge_index[(c,b)], edge_index[(b,d)], edge_index[(d,a)]))
    return np.array(cycles, dtype=np.int32)

_cycles = _build_cycles()
C = len(_cycles)

# For each edge, list of cycles it belongs to
_edge_to_cycles = [[] for _ in range(E_TOTAL)]
for cid, cycle in enumerate(_cycles):
    for e in cycle:
        _edge_to_cycles[e].append(cid)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 35s margin
    
    # Start with complete graph, random colors
    colors = np.random.randint(0, N, size=E_TOTAL, dtype=np.int32)
    
    def is_valid():
        """Check if all cycles are rainbow."""
        # For each cycle, check if 4 colors are distinct
        c0 = colors[_cycles[:, 0]]
        c1 = colors[_cycles[:, 1]]
        c2 = colors[_cycles[:, 2]]
        c3 = colors[_cycles[:, 3]]
        # A cycle is rainbow iff all 4 are distinct
        # Check: c0 != c1, c0 != c2, c0 != c3, c1 != c2, c1 != c3, c2 != c3
        bad = ((c0 == c1) | (c0 == c2) | (c0 == c3) | 
               (c1 == c2) | (c1 == c3) | (c2 == c3))
        return np.sum(bad)
    
    # Track bad count
    bad_count = is_valid()
    
    # Simulated Annealing with numpy acceleration
    T = 1.0
    cooling = 0.999995
    min_T = 0.005
    
    best_bad = bad_count
    best_colors = colors.copy()
    
    # Precompute for faster delta calculation
    # For each edge, the 4 positions it occupies in its cycles
    # Actually, let's just do the numpy check each time - it's fast enough
    
    while time.time() < deadline - 5:
        if bad_count == 0:
            break
            
        # Pick random edge
        e_idx = np.random.randint(0, E_TOTAL)
        old_c = colors[e_idx]
        
        # Pick new color
        new_c = np.random.randint(0, N)
        while new_c == old_c:
            new_c = np.random.randint(0, N)
        
        # Calculate delta by checking affected cycles
        affected = _edge_to_cycles[e_idx]
        
        # Get colors of affected cycles before change
        cyc = _cycles[affected]
        c0 = colors[cyc[:, 0]].copy()
        c1 = colors[cyc[:, 1]].copy()
        c2 = colors[cyc[:, 2]].copy()
        c3 = colors[cyc[:, 3]].copy()
        
        # Before: bad if any two equal
        was_bad = ((c0 == c1) | (c0 == c2) | (c0 == c3) | 
                   (c1 == c2) | (c1 == c3) | (c2 == c3))
        
        # Apply new color to the correct position
        # For each cycle, find which position e_idx occupies
        pos_in_cycle = np.zeros(len(affected), dtype=np.int32)
        for i, cid in enumerate(affected):
            cycle = _cycles[cid]
            for pos in range(4):
                if cycle[pos] == e_idx:
                    pos_in_cycle[i] = pos
                    break
        
        # Apply change
        mask0 = pos_in_cycle == 0
        mask1 = pos_in_cycle == 1
        mask2 = pos_in_cycle == 2
        mask3 = pos_in_cycle == 3
        c0[mask0] = new_c
        c1[mask1] = new_c
        c2[mask2] = new_c
        c3[mask3] = new_c
        
        # After: bad if any two equal
        is_bad_now = ((c0 == c1) | (c0 == c2) | (c0 == c3) | 
                      (c1 == c2) | (c1 == c3) | (c2 == c3))
        
        delta = np.sum(is_bad_now) - np.sum(was_bad)
        
        # Acceptance
        if delta < 0:
            accept = True
        elif delta == 0:
            accept = random.random() < 0.5
        else:
            if T < 1e-9:
                accept = False
            else:
                accept = random.random() < np.exp(-delta / T)
        
        if accept:
            colors[e_idx] = new_c
            bad_count += delta
            if bad_count < best_bad:
                best_bad = bad_count
                best_colors = colors.copy()
        
        T *= cooling
        if T < min_T:
            T = min_T
    
    # Use best colors found
    colors = best_colors
    bad_count = is_valid()
    
    # If still bad, remove edges to break bad cycles
    if bad_count > 0:
        active = np.ones(E_TOTAL, dtype=bool)
        
        # Get bad cycles
        c0 = colors[_cycles[:, 0]]
        c1 = colors[_cycles[:, 1]]
        c2 = colors[_cycles[:, 2]]
        c3 = colors[_cycles[:, 3]]
        bad_mask = ((c0 == c1) | (c0 == c2) | (c0 == c3) | 
                    (c1 == c2) | (c1 == c3) | (c2 == c3))
        bad_cycles = np.where(bad_mask)[0]
        
        # Iteratively remove edges that are in the most bad cycles
        while len(bad_cycles) > 0 and time.time() < deadline - 1:
            if not np.any(active):
                break
            
            # Count how many bad cycles each active edge is in
            edge_counts = np.zeros(E_TOTAL, dtype=np.int32)
            for cid in bad_cycles:
                for e in _cycles[cid]:
                    if active[e]:
                        edge_counts[e] += 1
            
            # Find edge with max count among active
            max_count = np.max(edge_counts[active])
            if max_count == 0:
                break
            
            # Pick one such edge
            candidates = np.where((edge_counts == max_count) & active)[0]
            best_edge = candidates[np.random.randint(0, len(candidates))]
            
            # Remove it
            active[best_edge] = False
            
            # Update bad cycles: remove those containing best_edge
            still_bad = []
            for cid in bad_cycles:
                if best_edge not in _cycles[cid]:
                    still_bad.append(cid)
            bad_cycles = np.array(still_bad, dtype=np.int32) if still_bad else np.array([], dtype=np.int32)
        
        # Build result from active edges
        result = []
        for i in range(E_TOTAL):
            if active[i]:
                u, v = _all_edges[i]
                result.append([u, v, int(colors[i])])
        return result
    else:
        # All cycles are rainbow, return all edges
        result = []
        for i in range(E_TOTAL):
            u, v = _all_edges[i]
            result.append([u, v, int(colors[i])])
        return result
