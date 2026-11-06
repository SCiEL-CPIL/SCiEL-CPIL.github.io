import time
import random

N = 9
SIZE = 1 << N
FULL_MASK = SIZE - 1
TIME_LIMIT = 118  # safety margin before 120s kill (150s wall-clock)

# Precompute neighbors for all 512 vertices
NEIGHBORS = [[] for _ in range(SIZE)]
for v in range(SIZE):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))


def _extend_end(path, used):
    """Extend path from the tail while possible (deterministic first valid)."""
    while True:
        cur = path[-1]
        nxt = None
        for nb in NEIGHBORS[cur]:
            if nb in used:
                continue
            ok = True
            for p in path[:-1]:
                if (nb ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                nxt = nb
                break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)


def _extend_start(path, used):
    """Extend path from the head while possible (deterministic first valid)."""
    while True:
        cur = path[0]
        nxt = None
        for nb in NEIGHBORS[cur]:
            if nb in used:
                continue
            ok = True
            for p in path[1:]:
                if (nb ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                nxt = nb
                break
        if nxt is None:
            break
        path.insert(0, nxt)
        used.add(nxt)


def _dfs_backtrack(start, rng, max_depth=150, deadline=None):
    """DFS with backtracking to find a long path."""
    path = [start]
    used = {start}
    best_path = list(path)
    
    def dfs(depth):
        nonlocal best_path
        if deadline and time.time() > deadline:
            return
        if len(path) > len(best_path):
            best_path = list(path)
            
        if len(path) >= max_depth:
            return
            
        cur = path[-1]
        nbs = NEIGHBORS[cur][:]
        rng.shuffle(nbs)
        
        for nb in nbs:
            if nb in used:
                continue
            ok = True
            for p in path[:-1]:
                if (nb ^ p).bit_count() == 1:
                    ok = False
                    break
            if not ok:
                continue
            
            path.append(nb)
            used.add(nb)
            
            dfs(depth + 1)
            
            path.pop()
            used.remove(nb)
            
            if deadline and time.time() > deadline:
                return

    dfs(0)
    return best_path


def _sa_search(path, rng, deadline):
    """Simulated Annealing with 2-bit flips and Segment Reversals."""
    current = list(path)
    used = set(current)
    best = list(current)
    best_len = len(current)

    T = 1.0
    T_min = 0.01
    alpha = 0.99985
    stuck = 0

    while time.time() < deadline:
        # 1) Always try to extend the ends (free gains)
        extended = False
        before = len(current)
        _extend_end(current, used)
        _extend_start(current, used)
        if len(current) > before:
            extended = True
            stuck = 0
        else:
            stuck += 1

        if extended:
            if len(current) > best_len:
                best = list(current)
                best_len = len(current)
            T = max(T * 0.99, T_min)
            continue

        # 2) No extension possible -> try a move
        L = len(current)
        if L < 3:
            break

        move_type = rng.random()
        
        # Save state for potential revert
        old_current = list(current)
        old_used = set(used)

        if move_type < 0.5:
            # 2-bit flip
            i = rng.randint(1, L - 2)
            u = current[i - 1]
            v = current[i]
            w = current[i + 1]

            v_prime = None
            for nb in NEIGHBORS[u]:
                if nb == v:
                    continue
                if (nb ^ w).bit_count() == 1:
                    v_prime = nb
                    break
            if v_prime is None or v_prime in used:
                T *= alpha
                continue

            ok = True
            for idx, p in enumerate(current):
                if idx == i - 1 or idx == i + 1:
                    continue
                if (v_prime ^ p).bit_count() == 1:
                    ok = False
                    break
            if not ok:
                T *= alpha
                continue

            # Perform swap
            used.remove(v)
            used.add(v_prime)
            current[i] = v_prime
            
            _extend_end(current, used)
            _extend_start(current, used)

            if len(current) > best_len:
                best = list(current)
                best_len = len(current)
                stuck = 0
                T = min(T * 1.02, 1.0)
            else:
                if rng.random() < T:
                    pass
                else:
                    current = old_current
                    used = old_used
                    
        else:
            # Segment Reversal
            seg_len = rng.randint(2, min(5, L - 2))
            i = rng.randint(0, L - seg_len - 1)
            j = i + seg_len - 1
            
            left_ok = True
            right_ok = True
            
            if i > 0:
                if (current[i-1] ^ current[j]).bit_count() != 1:
                    left_ok = False
                else:
                    for k in range(i+1, j):
                        if (current[i-1] ^ current[k]).bit_count() == 1:
                            left_ok = False
                            break
            if j < L - 1:
                if (current[i] ^ current[j+1]).bit_count() != 1:
                    right_ok = False
                else:
                    for k in range(i+1, j):
                        if (current[j+1] ^ current[k]).bit_count() == 1:
                            right_ok = False
                            break
                            
            if left_ok and right_ok:
                new_segment = current[i:j+1][::-1]
                current[i:j+1] = new_segment
                
                _extend_end(current, used)
                _extend_start(current, used)
                
                if len(current) > best_len:
                    best = list(current)
                    best_len = len(current)
                    stuck = 0
                    T = min(T * 1.02, 1.0)
                else:
                    if rng.random() < T:
                        pass
                    else:
                        current = old_current
                        used = old_used
                        
        T *= alpha
        if T < T_min:
            T = T_min

    return best


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    rng = random.Random(20260715)

    best_path = None
    best_len = 0

    # --- Phase 1: Build seeds using DFS Backtracking ---
    for s in range(0, 10):
        if time.time() > deadline - 20:
            break
        seed = _dfs_backtrack(s, rng, max_depth=150, deadline=deadline - 10)
        used = set(seed)
        _extend_end(seed, used)
        _extend_start(seed, used)
        
        if best_path is None or len(seed) > best_len:
            best_path = list(seed)
            best_len = len(seed)

    # --- Phase 2: SA on the best seed ---
    if time.time() < deadline - 10 and best_path:
        sa_deadline = min(deadline - 2, start_time + 60)
        improved = _sa_search(best_path, rng, sa_deadline)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)

    # --- Phase 3: Randomized restarts with DFS + SA Polish ---
    while time.time() < deadline - 5:
        start_v = rng.randint(0, FULL_MASK)
        fresh = _dfs_backtrack(start_v, rng, max_depth=120, deadline=deadline - 5)
        fu = set(fresh)
        _extend_end(fresh, fu)
        _extend_start(fresh, fu)

        if len(fresh) > best_len:
            best_path = list(fresh)
            best_len = len(fresh)

        sub_deadline = min(deadline - 2, time.time() + 5)
        if time.time() < sub_deadline:
            pol = _sa_search(fresh, rng, sub_deadline)
            if len(pol) > best_len:
                best_path = list(pol)
                best_len = len(pol)

        if time.time() > deadline - 5:
            break

    return best_path
