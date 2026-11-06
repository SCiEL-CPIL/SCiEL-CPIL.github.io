import time
import sys
import numpy as np
import random

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

_N = P['n']
_EDGE_ORDER = P['edge_order']
_NEDGES = len(_EDGE_ORDER)

_EDGE_POS = {}
for _i, (_a, _b) in enumerate(_EDGE_ORDER, 1):
    _EDGE_POS[(_a, _b)] = _i
    _EDGE_POS[(_b, _a)] = _i

_ADJ = [[] for _ in range(_N)]
for _i, (_a, _b) in enumerate(_EDGE_ORDER, 1):
    _ADJ[_a].append((_i, _b))
    _ADJ[_b].append((_i, _a))
for _v in range(_N):
    _ADJ[_v].sort()

sys.setrecursionlimit(10000)


def _score_cycle(cycle):
    """Compute the max edge position in the cycle."""
    n = len(cycle)
    max_pos = 0
    for i in range(n):
        a = cycle[i]
        b = cycle[(i + 1) % n]
        pos = _EDGE_POS[(a, b)]
        if pos > max_pos:
            max_pos = pos
    return max_pos


def _is_valid_cycle(cycle):
    """Check if cycle is a valid Hamilton cycle."""
    if len(cycle) != _N:
        return False
    if len(set(cycle)) != _N:
        return False
    return all(0 <= v < _N for v in cycle)


def _greedy_construct(start_vertex=0):
    """Construct a Hamilton cycle using greedy nearest-neighbor approach."""
    visited = set()
    cycle = [start_vertex]
    visited.add(start_vertex)
    
    current = start_vertex
    for _ in range(_N - 1):
        # Find unvisited neighbor with minimum edge position
        best_next = None
        best_pos = float('inf')
        for neighbor in range(_N):
            if neighbor not in visited:
                pos = _EDGE_POS[(current, neighbor)]
                if pos < best_pos:
                    best_pos = pos
                    best_next = neighbor
        if best_next is None:
            return None
        cycle.append(best_next)
        visited.add(best_next)
        current = best_next
    
    # Check if we can close the cycle
    if _EDGE_POS[(current, start_vertex)] < float('inf'):
        return cycle
    return None


def _two_opt_improve(cycle, deadline):
    """Apply 2-opt moves to minimize the max edge position."""
    n = len(cycle)
    best_cycle = list(cycle)
    best_score = _score_cycle(best_cycle)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1):
            if time.time() >= deadline:
                return best_cycle
            for j in range(i + 2, n):
                if time.time() >= deadline:
                    return best_cycle
                
                # Current edges: (cycle[i], cycle[i+1]) and (cycle[j], cycle[(j+1)%n])
                # New edges: (cycle[i], cycle[j]) and (cycle[i+1], cycle[(j+1)%n])
                old1 = _EDGE_POS[(cycle[i], cycle[i + 1])]
                old2 = _EDGE_POS[(cycle[j], cycle[(j + 1) % n])]
                new1 = _EDGE_POS[(cycle[i], cycle[j])]
                new2 = _EDGE_POS[(cycle[i + 1], cycle[(j + 1) % n])]
                
                # Compute new max score
                max_other = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    a = cycle[k]
                    b = cycle[(k + 1) % n]
                    pos = _EDGE_POS[(a, b)]
                    if pos > max_other:
                        max_other = pos
                
                new_max = max(max_other, new1, new2)
                if new_max < best_score:
                    segment = cycle[i + 1:j + 1]
                    segment.reverse()
                    cycle[i + 1:j + 1] = segment
                    best_score = new_max
                    best_cycle = list(cycle)
                    improved = True
                    break
            if improved:
                break
    
    return best_cycle


def _or_opt_improve(cycle, deadline):
    """Apply Or-opt moves (relocate 1-3 consecutive vertices)."""
    n = len(cycle)
    best_cycle = list(cycle)
    best_score = _score_cycle(best_cycle)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for seg_len in [1, 2, 3]:
            if time.time() >= deadline:
                return best_cycle
            for i in range(n):
                if time.time() >= deadline:
                    return best_cycle
                # Extract segment of length seg_len starting at i
                segment = [cycle[(i + k) % n] for k in range(seg_len)]
                
                # Try inserting at each position j
                for j in range(n):
                    if time.time() >= deadline:
                        return best_cycle
                    if j == i or j == (i + seg_len - 1) % n:
                        continue
                    
                    # Create new cycle by removing segment and inserting at j
                    new_cycle = []
                    for k in range(n):
                        if k < i or k >= i + seg_len:
                            new_cycle.append(cycle[k])
                    
                    # Insert segment at position j
                    insert_pos = j if j < i else j - seg_len
                    for k in range(seg_len):
                        new_cycle.insert(insert_pos + k, segment[k])
                    
                    if len(new_cycle) != n:
                        continue
                    if len(set(new_cycle)) != n:
                        continue
                    
                    new_score = _score_cycle(new_cycle)
                    if new_score < best_score:
                        best_score = new_score
                        best_cycle = new_cycle
                        cycle = list(new_cycle)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return best_cycle


def _simulated_annealing(cycle, deadline, start_temp=100.0, end_temp=1.0, cooling=0.95):
    """Apply simulated annealing to escape local optima."""
    n = len(cycle)
    current_cycle = list(cycle)
    current_score = _score_cycle(current_cycle)
    best_cycle = list(current_cycle)
    best_score = current_score
    
    temp = start_temp
    rng = random.Random(42)
    
    while time.time() < deadline and temp > end_temp:
        # Generate a random move
        move_type = rng.choice(['2opt', 'oropt', 'swap'])
        
        if move_type == '2opt':
            i, j = sorted(rng.sample(range(n), 2))
            if j - i < 2:
                continue
            new_cycle = current_cycle[:i+1] + current_cycle[i+1:j+1][::-1] + current_cycle[j+1:]
        elif move_type == 'oropt':
            seg_len = rng.choice([1, 2, 3])
            i = rng.randint(0, n-1)
            j = rng.randint(0, n-1)
            if j == i or j == (i + seg_len - 1) % n:
                continue
            segment = [current_cycle[(i + k) % n] for k in range(seg_len)]
            new_cycle = []
            for k in range(n):
                if k < i or k >= i + seg_len:
                    new_cycle.append(current_cycle[k])
            insert_pos = j if j < i else j - seg_len
            for k in range(seg_len):
                new_cycle.insert(insert_pos + k, segment[k])
            if len(new_cycle) != n or len(set(new_cycle)) != n:
                continue
        else:  # swap
            i, j = rng.sample(range(n), 2)
            new_cycle = list(current_cycle)
            new_cycle[i], new_cycle[j] = new_cycle[j], new_cycle[i]
        
        if len(new_cycle) != n or len(set(new_cycle)) != n:
            continue
        
        new_score = _score_cycle(new_cycle)
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            current_cycle = new_cycle
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_cycle = list(new_cycle)
        
        temp *= cooling
    
    return best_cycle


def _multi_start_local_search(deadline):
    """Try multiple starting points and local search from each."""
    best_cycle = None
    best_score = float('inf')
    
    # Try greedy construction from different starts
    for start in range(_N):
        if time.time() >= deadline:
            break
        cycle = _greedy_construct(start)
        if cycle is not None and _is_valid_cycle(cycle):
            score = _score_cycle(cycle)
            if score < best_score:
                best_score = score
                best_cycle = cycle
    
    # Also try a simple random cycle
    if time.time() < deadline:
        rng = random.Random(123)
        random_cycle = list(range(_N))
        rng.shuffle(random_cycle)
        score = _score_cycle(random_cycle)
        if score < best_score:
            best_score = score
            best_cycle = random_cycle
    
    if best_cycle is None:
        # Fallback: identity cycle
        best_cycle = list(range(_N))
        best_score = _score_cycle(best_cycle)
    
    # Polish with 2-opt
    if time.time() < deadline - 5:
        best_cycle = _two_opt_improve(best_cycle, deadline - 5)
        best_score = _score_cycle(best_cycle)
    
    # Polish with Or-opt
    if time.time() < deadline - 3:
        best_cycle = _or_opt_improve(best_cycle, deadline - 3)
        best_score = _score_cycle(best_cycle)
    
    # Simulated annealing
    if time.time() < deadline - 2:
        best_cycle = _simulated_annealing(best_cycle, deadline - 2)
        best_score = _score_cycle(best_cycle)
    
    # Final 2-opt polish
    if time.time() < deadline:
        best_cycle = _two_opt_improve(best_cycle, deadline)
        best_score = _score_cycle(best_cycle)
    
    return best_cycle, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Multi-start local search
    best_cycle, best_score = _multi_start_local_search(deadline)
    
    return best_cycle
