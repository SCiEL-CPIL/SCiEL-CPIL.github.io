"""Snake-in-the-box in Q_9: greedy construction + local search to extend the path."""
import time
import random

N = 9
TOTAL_VERTS = 1 << N

def _is_valid_snake(path):
    """Check if path is a valid induced path (snake)."""
    if len(path) < 2:
        return True
    used = set(path)
    if len(used) != len(path):
        return False
    for i in range(len(path) - 1):
        diff = path[i] ^ path[i + 1]
        if diff.bit_count() != 1:
            return False
    # Check non-consecutive vertices differ in at least 2 bits
    for i in range(len(path)):
        for j in range(i + 2, len(path)):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True

def _greedy_seed():
    """Greedy snake: always extend with the lowest bit flip that keeps the path induced."""
    path, used = [0], {0}
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check v is not at distance 1 from any non-adjacent vertex
            ok = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                nxt = v
                break
        if nxt is None:
            return path
        path.append(nxt)
        used.add(nxt)

def _try_extend(path, used, max_time=None, deadline=None):
    """Try to extend the path further using greedy with randomization.
    Returns the extended path or the original if no extension found."""
    if deadline is not None and time.time() > deadline:
        return path
    
    best = list(path)
    best_used = set(used)
    
    # Try multiple random bit orderings
    for attempt in range(200):
        if deadline is not None and time.time() > deadline:
            break
        cur = best[-1]
        extended = False
        for b in range(N):
            v = cur ^ (1 << b)
            if v in best_used:
                continue
            ok = True
            for p in best[:-1]:
                if (v ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                best.append(v)
                best_used.add(v)
                extended = True
                break
        if not extended:
            break
    
    return best

def _local_search_2opt(path, deadline):
    """Try 2-opt style moves: remove two non-adjacent edges and reconnect.
    Actually for snakes, we try to find a longer path by removing a vertex
    from the middle and reinserting it elsewhere, or by trying to extend
    from both ends."""
    best = list(path)
    best_len = len(best)
    best_used = set(best)
    
    # Try extending from the beginning
    while True:
        if time.time() > deadline:
            break
        cur = best[0]
        extended = False
        for b in range(N):
            v = cur ^ (1 << b)
            if v in best_used:
                continue
            ok = True
            for p in best[1:]:
                if (v ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                best.insert(0, v)
                best_used.add(v)
                extended = True
                break
        if not extended:
            break
    
    # Try extending from the end
    while True:
        if time.time() > deadline:
            break
        cur = best[-1]
        extended = False
        for b in range(N):
            v = cur ^ (1 << b)
            if v in best_used:
                continue
            ok = True
            for p in best[:-1]:
                if (v ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                best.append(v)
                best_used.add(v)
                extended = True
                break
        if not extended:
            break
    
    return best

def _try_splice(path, deadline):
    """Try to find a longer snake by removing a segment and trying to reconnect.
    This is a heuristic: pick two positions i < j (non-adjacent), remove path[i:j],
    and try to connect path[i-1] to path[j] or find a new middle."""
    # This is complex; skip for now and rely on other methods
    return path

def _random_restart_greedy(deadline):
    """Start from a random vertex and do greedy construction."""
    best_path = None
    best_len = 0
    
    while time.time() < deadline:
        start = random.randint(0, TOTAL_VERTS - 1)
        path, used = [start], {start}
        while True:
            cur = path[-1]
            nxt = None
            # Randomize bit order
            bits = list(range(N))
            random.shuffle(bits)
            for b in bits:
                v = cur ^ (1 << b)
                if v in used:
                    continue
                ok = True
                for p in path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        ok = False
                        break
                if ok:
                    nxt = v
                    break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
            # Try to extend further
            best_path = _try_extend(best_path, used, deadline=deadline)
    
    return best_path if best_path else None

def _warnsdorff_greedy():
    """Warnsdorff-like greedy: prefer moves that leave fewer options for the next step."""
    path, used = [0], {0}
    while True:
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            ok = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                # Count how many options v would have next
                count = 0
                for b2 in range(N):
                    v2 = v ^ (1 << b2)
                    if v2 in used or v2 == cur:
                        continue
                    ok2 = True
                    for p2 in path[:-1]:
                        if (v2 ^ p2).bit_count() == 1:
                            ok2 = False
                            break
                    if ok2:
                        count += 1
                candidates.append((count, b))
        
        if not candidates:
            break
        
        # Pick the candidate with fewest next options (Warnsdorff's rule)
        candidates.sort()
        _, b = candidates[0]
        nxt = cur ^ (1 << b)
        path.append(nxt)
        used.add(nxt)
    
    return path

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Start with the greedy seed
    best = _greedy_seed()
    best_len = len(best)
    
    # Try Warnsdorff greedy
    ws = _warnsdorff_greedy()
    if len(ws) > best_len:
        best = ws
        best_len = len(ws)
    
    # Try to extend the best path
    best = _try_extend(best, set(best), deadline=deadline)
    best_len = len(best)
    
    # Try local search (extend from both ends)
    best = _local_search_2opt(best, deadline)
    best_len = len(best)
    
    # Try random restarts with greedy
    if time.time() < deadline - 5:
        random.seed(42)
        rr = _random_restart_greedy(deadline - 5)
        if rr and len(rr) > best_len:
            best = rr
            best_len = len(rr)
            best = _try_extend(best, set(best), deadline=deadline)
            best = _local_search_2opt(best, deadline)
            best_len = len(best)
    
    # Final extension attempts
    best = _try_extend(best, set(best), deadline=deadline)
    best = _local_search_2opt(best, deadline)
    
    return best
# EVOLVE-BLOCK-END
