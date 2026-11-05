import time
import random

N = 12
TOTAL_STATES = 1 << N  # 4096 possible subsets
SINGLETON_MASKS = [1 << i for i in range(N)]

def _bfs_threshold(a, b):
    """Compute exact shortest reset word length via BFS on subset automaton.
    Returns -1 if not synchronizing."""
    # Precompute transition tables for all subsets
    # trans[subset][letter] = next_subset
    # Use arrays for speed
    trans_a = [0] * TOTAL_STATES
    trans_b = [0] * TOTAL_STATES
    
    for s in range(TOTAL_STATES):
        sa = 0
        sb = 0
        for i in range(N):
            if s & (1 << i):
                sa |= (1 << a[i])
                sb |= (1 << b[i])
        trans_a[s] = sa
        trans_b[s] = sb
    
    full = TOTAL_STATES - 1
    # BFS
    # visited: -1 = unvisited, else depth
    visited = [-1] * TOTAL_STATES
    visited[full] = 0
    queue = [full]
    head = 0
    while head < len(queue):
        s = queue[head]
        head += 1
        d = visited[s]
        # Try 'a'
        ns = trans_a[s]
        if visited[ns] == -1:
            if ns & (ns - 1) == 0:  # singleton
                return d + 1
            visited[ns] = d + 1
            queue.append(ns)
        # Try 'b'
        ns = trans_b[s]
        if visited[ns] == -1:
            if ns & (ns - 1) == 0:  # singleton
                return d + 1
            visited[ns] = d + 1
            queue.append(ns)
    return -1

def _is_synchronizing(a, b):
    return _bfs_threshold(a, b) >= 0

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Initial constructions to seed the search
    candidates = []
    
    # 1. Pin construction: a = cycle, b = shift with one fixed point
    a_pin = [(i + 1) % N for i in range(N)]
    b_pin = [0] * N
    for i in range(1, N):
        b_pin[i] = (i + 1) % N
    candidates.append((list(a_pin), list(b_pin)))
    
    # 2. Another Pin variant: b maps 0->0, i->i-1 for i>0 (reverse shift with sink)
    b_pin2 = [0] * N
    for i in range(1, N):
        b_pin2[i] = i - 1
    candidates.append((list(a_pin), list(b_pin2)))
    
    # 3. Černý-like: a = cycle, b = maps all to 0 except one
    for defect in range(N):
        b_c = [0] * N
        b_c[defect] = (defect + 1) % N
        candidates.append((list(a_pin), list(b_c)))
    
    # 4. Random good starts
    rng = random.Random(42)
    for _ in range(20):
        a_rand = [rng.randint(0, N-1) for _ in range(N)]
        b_rand = [rng.randint(0, N-1) for _ in range(N)]
        candidates.append((a_rand, b_rand))
    
    best_a = None
    best_b = None
    best_score = -1
    
    # Evaluate all initial candidates
    for a_c, b_c in candidates:
        if time.time() > DEADLINE:
            break
        score = _bfs_threshold(a_c, b_c)
        if score > best_score:
            best_score = score
            best_a = list(a_c)
            best_b = list(b_c)
    
    if best_a is None:
        # Fallback: simple synchronizing automaton
        return [[(i + 1) % N for i in range(N)], [0] * N]
    
    # Steepest ascent hill climbing
    # Neighborhood: change one transition (letter, state)
    # 2 letters * N states = 24 neighbors
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        best_neighbor_score = best_score
        best_neighbor_a = None
        best_neighbor_b = None
        
        for letter in range(2):
            for state in range(N):
                if time.time() > DEADLINE:
                    break
                # Try all possible destinations for this transition
                cur_dest = (best_a if letter == 0 else best_b)[state]
                for new_dest in range(N):
                    if new_dest == cur_dest:
                        continue
                    # Create modified automaton
                    if letter == 0:
                        a_mod = list(best_a)
                        b_mod = list(best_b)
                        a_mod[state] = new_dest
                    else:
                        a_mod = list(best_a)
                        b_mod = list(best_b)
                        b_mod[state] = new_dest
                    
                    score = _bfs_threshold(a_mod, b_mod)
                    if score > best_neighbor_score:
                        best_neighbor_score = score
                        best_neighbor_a = a_mod
                        best_neighbor_b = b_mod
                if time.time() > DEADLINE:
                    break
            if time.time() > DEADLINE:
                break
        
        if best_neighbor_a is not None and best_neighbor_score > best_score:
            best_a = best_neighbor_a
            best_b = best_neighbor_b
            best_score = best_neighbor_score
            improved = True
    
    return [best_a, best_b]
