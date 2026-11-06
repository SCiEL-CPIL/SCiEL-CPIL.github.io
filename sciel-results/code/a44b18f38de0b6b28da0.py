"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)

DEADLINE = 85.0  # safety margin before 90s kill

def _get_components():
    """Find connected components and their sizes."""
    visited = set()
    components = []
    for i in range(N):
        if i not in visited:
            comp = set()
            queue = deque([i])
            visited.add(i)
            while queue:
                node = queue.popleft()
                comp.add(node)
                for nb in ADJ[node]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            components.append(comp)
    return components

def _warnsdorff_extend(path, seen, direction='forward'):
    """Extend path greedily using Warnsdorff heuristic."""
    while True:
        cur = path[-1] if direction == 'forward' else path[0]
        cands = [v for v in ADJ[cur] if v not in seen]
        if not cands:
            break
        # Warnsdorff: prefer vertices with fewer unvisited neighbors
        cands.sort(key=lambda v: (len(ADJ[v] - seen), v))
        nxt = cands[0]
        if direction == 'forward':
            path.append(nxt)
        else:
            path.insert(0, nxt)
        seen.add(nxt)
    return path

def _greedy_bidirectional(start, seen):
    """Extend path from both ends using Warnsdorff."""
    path = [start]
    seen = set(seen)
    seen.add(start)
    # Alternate extending forward and backward
    forward = True
    while True:
        cur = path[-1] if forward else path[0]
        cands = [v for v in ADJ[cur] if v not in seen]
        if not cands:
            if forward:
                forward = False
                continue
            else:
                break
        cands.sort(key=lambda v: (len(ADJ[v] - seen), v))
        nxt = cands[0]
        if forward:
            path.append(nxt)
        else:
            path.insert(0, nxt)
        seen.add(nxt)
        forward = not forward
    return path

def _dfs_with_pruning(start, target_len, time_limit, best_path, best_len):
    """DFS with Warnsdorff ordering and depth-first search for longer paths."""
    path = [start]
    seen = {start}
    start_time = time.time()
    
    def _dfs():
        nonlocal best_path, best_len
        if time.time() - start_time > time_limit:
            return False
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        cur = path[-1]
        cands = [v for v in ADJ[cur] if v not in seen]
        if not cands:
            return True
        
        # Warnsdorff ordering: prefer vertices with fewer unvisited neighbors
        cands.sort(key=lambda v: (len(ADJ[v] - seen), v))
        
        for v in cands:
            if time.time() - start_time > time_limit:
                return True
            path.append(v)
            seen.add(v)
            if not _dfs():
                return True
            path.pop()
            seen.remove(v)
        return True
    
    _dfs()
    return best_path, best_len

def _beam_search(start, beam_width=5, max_depth=50, time_limit=5.0):
    """Beam search for long paths."""
    start_time = time.time()
    # Each beam entry: (path, seen_set)
    beams = [([start], {start})]
    best_path = [start]
    best_len = 1
    
    for depth in range(max_depth):
        if time.time() - start_time > time_limit:
            break
        if not beams:
            break
        new_beams = []
        for path, seen in beams:
            cur = path[-1]
            cands = [v for v in ADJ[cur] if v not in seen]
            if not cands:
                continue
            # Score candidates: prefer fewer unvisited neighbors (Warnsdorff)
            cands.sort(key=lambda v: (len(ADJ[v] - seen), v))
            # Take top beam_width candidates
            for v in cands[:beam_width]:
                new_path = path + [v]
                new_seen = seen | {v}
                new_beams.append((new_path, new_seen))
                if len(new_path) > best_len:
                    best_len = len(new_path)
                    best_path = new_path[:]
        # Keep only top beam_width beams by length, then by Warnsdorff score
        if new_beams:
            new_beams.sort(key=lambda x: (-len(x[0]), len(ADJ[x[0][-1]] - x[1])))
            beams = new_beams[:beam_width]
        else:
            break
    
    return best_path, best_len

def construct():
    start_global = time.time()
    
    # Find components
    components = _get_components()
    # Sort by size descending
    components.sort(key=len, reverse=True)
    
    best_path = []
    best_len = 0
    
    # Phase 1: Quick greedy from all vertices in largest component
    if components:
        largest = components[0]
        for v in sorted(largest, key=lambda x: len(ADJ[x])):
            if time.time() - start_global > DEADLINE - 10:
                break
            path = _greedy_bidirectional(v, set())
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
    
    # Phase 2: DFS with Warnsdorff from promising starts
    # Pick starts: low-degree vertices in largest components
    promising_starts = []
    for comp in components[:3]:  # Top 3 components
        for v in sorted(comp, key=lambda x: (len(ADJ[x]), x)):
            promising_starts.append(v)
    
    # Allocate time for DFS
    remaining = DEADLINE - time.time() - 5  # Leave 5s margin
    if remaining > 2:
        # Try DFS from top promising starts
        n_starts = min(10, len(promising_starts))
        time_per_start = remaining / n_starts
        for start in promising_starts[:n_starts]:
            if time.time() - start_global > DEADLINE - 5:
                break
            path, length = _dfs_with_pruning(start, best_len + 1, time_per_start, best_path[:], best_len)
            if length > best_len:
                best_len = length
                best_path = path[:]
    
    # Phase 3: Beam search from best endpoints and promising starts
    remaining = DEADLINE - time.time() - 2
    if remaining > 1:
        # Try beam search from various starts
        beam_starts = promising_starts[:15]
        time_per_beam = remaining / max(1, len(beam_starts))
        for start in beam_starts:
            if time.time() - start_global > DEADLINE - 2:
                break
            path, length = _beam_search(start, beam_width=8, max_depth=60, time_limit=time_per_beam)
            if length > best_len:
                best_len = length
                best_path = path[:]
    
    # Phase 4: Try to extend the best path found
    if best_path and len(best_path) < N:
        remaining = DEADLINE - time.time() - 1
        if remaining > 0.5:
            # Try extending from both ends
            seen = set(best_path)
            # Extend forward
            ext_path = best_path[:]
            ext_seen = set(seen)
            while True:
                cur = ext_path[-1]
                cands = [v for v in ADJ[cur] if v not in ext_seen]
                if not cands:
                    break
                cands.sort(key=lambda v: (len(ADJ[v] - ext_seen), v))
                nxt = cands[0]
                ext_path.append(nxt)
                ext_seen.add(nxt)
            if len(ext_path) > best_len:
                best_len = len(ext_path)
                best_path = ext_path[:]
            
            # Extend backward
            ext_path = best_path[:]
            ext_seen = set(seen)
            while True:
                cur = ext_path[0]
                cands = [v for v in ADJ[cur] if v not in ext_seen]
                if not cands:
                    break
                cands.sort(key=lambda v: (len(ADJ[v] - ext_seen), v))
                nxt = cands[0]
                ext_path.insert(0, nxt)
                ext_seen.add(nxt)
            if len(ext_path) > best_len:
                best_len = len(ext_path)
                best_path = ext_path[:]
    
    # Fallback
    if not best_path:
        best_path = [0]
    
    return best_path
# EVOLVE-BLOCK-END
