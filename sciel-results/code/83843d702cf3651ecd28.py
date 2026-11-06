"""Snake-in-the-box in Q_9: hybrid local search + greedy extension."""
import time
import random

N = 9
MASK = (1 << N) - 1
DEADLINE = 145.0  # safety margin before 150s kill


def _is_valid_snake(path):
    """Check that path is a valid induced path in Q_N."""
    if not path:
        return True
    used = set(path)
    if len(used) != len(path):
        return False
    for i in range(len(path) - 1):
        if (path[i] ^ path[i + 1]).bit_count() != 1:
            return False
    # Check non-consecutive differ by >= 2
    # This is expensive; do a sampled check for speed in inner loops
    # Full check only at the end
    return True


def _is_valid_snake_full(path):
    """Full validation: consecutive differ by 1, non-consecutive differ by >= 2."""
    n = len(path)
    if n == 0:
        return True
    used = set(path)
    if len(used) != n:
        return False
    for i in range(n - 1):
        if (path[i] ^ path[i + 1]).bit_count() != 1:
            return False
    for i in range(n):
        for j in range(i + 2, n):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True


def _can_extend(path, candidate):
    """Check if candidate can be appended to path (induced path constraint)."""
    if candidate in path:
        return False
    cur = path[-1]
    if (candidate ^ cur).bit_count() != 1:
        return False
    # candidate must differ from all non-last vertices by >= 2 bits
    for p in path[:-1]:
        if (candidate ^ p).bit_count() == 1:
            return False
    return True


def _greedy_extend(path, used_set, rng, max_steps=None):
    """Extend path greedily (with some randomization) until stuck or max_steps."""
    if max_steps is None:
        max_steps = 10000
    for _ in range(max_steps):
        cur = path[-1]
        # Shuffle bit order for diversity
        bits = list(range(N))
        rng.shuffle(bits)
        extended = False
        for b in bits:
            v = cur ^ (1 << b)
            if v in used_set:
                continue
            # Check induced constraint
            ok = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                path.append(v)
                used_set.add(v)
                extended = True
                break
        if not extended:
            break
    return path


def _greedy_extend_deterministic(path, used_set):
    """Deterministic greedy: lowest bit flip first."""
    while True:
        cur = path[-1]
        extended = False
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used_set:
                continue
            ok = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                path.append(v)
                used_set.add(v)
                extended = True
                break
        if not extended:
            break
    return path


def _try_local_search(path, used_set, rng, time_limit):
    """
    Local search: try to find a better extension by backtracking a few steps
    and choosing different branches.
    """
    best_path = list(path)
    best_len = len(best_path)
    
    # Try backtracking different depths
    for depth in [1, 2, 3, 4, 5]:
        if time.time() > time_limit:
            break
        if len(path) <= depth:
            continue
        
        # Save state
        saved_path = list(path)
        saved_used = set(used_set)
        
        # Remove last 'depth' vertices
        for _ in range(depth):
            v = path.pop()
            used_set.discard(v)
        
        # Try alternative extensions from the new endpoint
        cur = path[-1]
        bits = list(range(N))
        rng.shuffle(bits)
        
        for b in bits:
            if time.time() > time_limit:
                break
            v = cur ^ (1 << b)
            if v in used_set:
                continue
            ok = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    ok = False
                    break
            if not ok:
                continue
            
            # Tentatively add v and try to extend
            path.append(v)
            used_set.add(v)
            
            # Extend greedily from here
            _greedy_extend(path, used_set, rng, max_steps=500)
            
            if len(path) > best_len:
                best_path = list(path)
                best_len = len(best_path)
            
            # Restore
            for _ in range(len(path) - len(saved_path) + depth):
                path.pop()
            used_set.clear()
            used_set.update(saved_used)
        
        # Restore original state
        path.clear()
        path.extend(saved_path)
        used_set.clear()
        used_set.update(saved_used)
        
        if len(best_path) > len(path):
            path.clear()
            path.extend(best_path)
            used_set.clear()
            used_set.update(set(best_path))
    
    return path


def _random_walk_extension(path, used_set, rng, time_limit):
    """Try random walks to extend the path."""
    cur = path[-1]
    for _ in range(100000):
        if time.time() > time_limit:
            break
        b = rng.randint(0, N - 1)
        v = cur ^ (1 << b)
        if v in used_set:
            continue
        ok = True
        for p in path[:-1]:
            if (v ^ p).bit_count() == 1:
                ok = False
                break
        if ok:
            path.append(v)
            used_set.add(v)
            cur = v
        else:
            break
    return path


def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_path = []
    best_len = 0
    
    # Phase 1: Generate multiple greedy snakes with different random seeds
    num_seeds = 50
    for seed in range(num_seeds):
        if time.time() > DEADLINE:
            break
        
        rng = random.Random(seed)
        # Start from a random vertex
        start_v = rng.randint(0, (1 << N) - 1)
        path = [start_v]
        used_set = {start_v}
        
        # Greedy extend with randomization
        path = _greedy_extend(path, used_set, rng, max_steps=100000)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
    
    # Phase 2: Local search on the best path
    if best_path:
        used_set = set(best_path)
        time_limit = DEADLINE - 5  # leave 5s for validation
        best_path = _try_local_search(best_path, used_set, rng, time_limit)
        best_len = len(best_path)
    
    # Phase 3: Try more diverse starts
    if time.time() < DEADLINE - 10:
        for seed in range(100, 300):
            if time.time() > DEADLINE - 10:
                break
            rng = random.Random(seed)
            start_v = rng.randint(0, (1 << N) - 1)
            path = [start_v]
            used_set = {start_v}
            path = _greedy_extend(path, used_set, rng, max_steps=100000)
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
                
                # Quick local search on new best
                used_set = set(best_path)
                time_limit = min(DEADLINE - 5, time.time() + 2)
                best_path = _try_local_search(best_path, used_set, rng, time_limit)
                best_len = len(best_path)
    
    # Phase 4: Final validation
    if best_path and not _is_valid_snake_full(best_path):
        # Fallback to deterministic greedy
        path = [0]
        used_set = {0}
        path = _greedy_extend_deterministic(path, used_set)
        best_path = path
        best_len = len(path)
    
    # Ensure we have at least the seed length
    if len(best_path) < 104:
        path = [0]
        used_set = {0}
        path = _greedy_extend_deterministic(path, used_set)
        best_path = path
        best_len = len(path)
    
    return best_path


if __name__ == "__main__":
    result = construct()
    print(f"Snake length: {len(result)}")
