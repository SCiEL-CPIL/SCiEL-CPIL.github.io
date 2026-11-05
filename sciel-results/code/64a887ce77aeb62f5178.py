"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _cyclic_dist(a, b, n):
    """Cyclic distance between a and b on cycle of length n (min of clockwise/ccw)."""
    d = abs(a - b) % n
    return min(d, n - d)

def _hc_path_len(a, b, n, direction=1):
    """Length of path from a to b going in given direction on HC."""
    if direction == 1:
        return (b - a) % n
    else:
        return (a - b) % n

def _find_cycles_with_chords(chords, n, max_chords_in_cycle=3):
    """Find all cycle lengths achievable with HC + given chords.
    Returns dict: length -> cycle (list of vertices)."""
    cycles_by_len = {}
    
    # Cycle of length n: the HC itself
    cycles_by_len[n] = list(range(n))
    
    # Cycles using 1 chord
    for (u, v) in chords:
        # Two possible cycles: go from u to v clockwise, or counterclockwise
        # Clockwise: u, u+1, ..., v (length = (v-u)%n + 1)
        cw_len = (v - u) % n
        if cw_len > 0:
            cyc_cw = list(range(u, v)) if u < v else list(range(u, n)) + list(range(v))
            if len(cyc_cw) == cw_len:
                if cw_len not in cycles_by_len and 3 <= cw_len <= n-1:
                    cycles_by_len[cw_len] = cyc_cw
        ccw_len = n - cw_len
        if ccw_len > 0 and ccw_len != cw_len:
            # Counterclockwise from u to v
            if u > v:
                cyc_ccw = list(range(v, u+1))
            else:
                cyc_ccw = list(range(u, n)) + list(range(v+1))
            if len(cyc_ccw) == ccw_len:
                if ccw_len not in cycles_by_len and 3 <= ccw_len <= n-1:
                    cycles_by_len[ccw_len] = cyc_ccw
    
    # Cycles using 2 chords
    for i in range(len(chords)):
        for j in range(i+1, len(chords)):
            u1, v1 = chords[i]
            u2, v2 = chords[j]
            # Try all 4 combinations of directions
            for d1 in [1, -1]:
                for d2 in [1, -1]:
                    # Cycle: u1 -> (HC) -> u2 -> (chord) -> v2 -> (HC) -> v1 -> (chord) -> u1
                    # Path u1 to u2 in direction d1
                    if d1 == 1:
                        p1 = (u2 - u1) % n
                    else:
                        p1 = (u1 - u2) % n
                    if p1 == 0:
                        continue
                    # Path v2 to v1 in direction d2
                    if d2 == 1:
                        p2 = (v1 - v2) % n
                    else:
                        p2 = (v2 - v1) % n
                    if p2 == 0:
                        continue
                    total_len = p1 + p2 + 2  # +2 for the two chords
                    if 3 <= total_len <= n-1 and total_len not in cycles_by_len:
                        # Build the cycle
                        if d1 == 1:
                            if u1 <= u2:
                                seg1 = list(range(u1, u2+1))
                            else:
                                seg1 = list(range(u1, n)) + list(range(u2+1))
                        else:
                            if u1 >= u2:
                                seg1 = list(range(u2, u1+1))[::-1]
                            else:
                                seg1 = list(range(u1, -1, -1)) + list(range(n, u2, -1))
                        
                        if d2 == 1:
                            if v2 <= v1:
                                seg2 = list(range(v2, v1+1))
                            else:
                                seg2 = list(range(v2, n)) + list(range(v1+1))
                        else:
                            if v2 >= v1:
                                seg2 = list(range(v1, v2+1))[::-1]
                            else:
                                seg2 = list(range(v2, -1, -1)) + list(range(n, v1, -1))
                        
                        # Combine: seg1 ends at u2, chord to v2, seg2, chord back to u1
                        # Actually the cycle is: u1...u2, v2...v1, then chords
                        # Let me rebuild: start at u1, go to u2 via HC, chord to v2, go to v1 via HC, chord to u1
                        cycle = seg1 + [x for x in seg2 if x != seg1[-1]]
                        # Remove duplicates and check
                        if len(cycle) == total_len and len(set(cycle)) == total_len:
                            cycles_by_len[total_len] = cycle
    
    return cycles_by_len

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin
    
    n = N
    
    # Start with Hamiltonian cycle
    hc_edges = [(i, (i+1) % n) for i in range(n)]
    
    # Candidate chords: all pairs not on HC
    # For efficiency, only consider chords with distance 2..n//2
    candidate_chords = []
    for i in range(n):
        for d in range(2, n//2 + 1):
            j = (i + d) % n
            if i < j:
                candidate_chords.append((i, j))
    
    # Greedy: start with no chords, add chords that cover most uncovered cycle lengths
    selected_chords = []
    covered_lengths = {n}  # HC covers length n
    cycles_by_len = {n: list(range(n))}
    
    while time.time() - start_time < DEADLINE - 5:
        # Check if all lengths 3..n are covered
        if all(k in covered_lengths for k in range(3, n+1)):
            break
        
        # Find best chord to add
        best_chord = None
        best_gain = 0
        best_new_cycles = {}
        
        for chord in candidate_chords:
            if chord in selected_chords:
                continue
            # Try adding this chord
            trial_chords = selected_chords + [chord]
            trial_cycles = _find_cycles_with_chords(trial_chords, n)
            trial_covered = set(trial_cycles.keys())
            
            new_covered = trial_covered - covered_lengths
            if len(new_covered) > best_gain:
                best_gain = len(new_covered)
                best_chord = chord
                best_new_cycles = trial_cycles
        
        if best_chord is None or best_gain == 0:
            break
        
        selected_chords.append(best_chord)
        covered_lengths = set(best_new_cycles.keys())
        cycles_by_len = best_new_cycles
    
    # Build final edge set
    all_edges = set(hc_edges)
    for u, v in selected_chords:
        if u > v:
            u, v = v, u
        all_edges.add((u, v))
    edges = list(all_edges)
    
    # Verify all cycle lengths 3..40 are covered
    if not all(k in cycles_by_len for k in range(3, n+1)):
        # Fallback to original construction
        chord_edges = [(0, k-1) for k in range(3, n)]
        all_edges_fb = set(hc_edges)
        for u, v in chord_edges:
            if u > v:
                u, v = v, u
            all_edges_fb.add((u, v))
        edges = list(all_edges_fb)
        cycles = []
        for k in range(3, n+1):
            if k == n:
                cycles.append(list(range(n)))
            else:
                cycles.append(list(range(k)))
        return {'edges': edges, 'cycles': cycles}
    
    # Build cycles list in order 3..40
    cycles = []
    for k in range(3, n+1):
        if k in cycles_by_len:
            cycles.append(cycles_by_len[k])
        else:
            # Should not happen
            cycles.append(list(range(k)))
    
    # Final verification
    edge_lookup = set()
    for a, b in edges:
        edge_lookup.add((a, b))
        edge_lookup.add((b, a))
    
    valid = True
    for cyc in cycles:
        k = len(cyc)
        if k < 3 or k > n:
            valid = False
            break
        if len(set(cyc)) != k:
            valid = False
            break
        for i in range(k):
            u = cyc[i]
            v = cyc[(i+1) % k]
            if (u, v) not in edge_lookup:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback
        chord_edges = [(0, k-1) for k in range(3, n)]
        all_edges_fb = set(hc_edges)
        for u, v in chord_edges:
            if u > v:
                u, v = v, u
            all_edges_fb.add((u, v))
        edges = list(all_edges_fb)
        cycles = []
        for k in range(3, n+1):
            if k == n:
                cycles.append(list(range(n)))
            else:
                cycles.append(list(range(k)))
    
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
