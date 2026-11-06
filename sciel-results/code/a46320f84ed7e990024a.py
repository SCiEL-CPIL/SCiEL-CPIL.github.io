"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 9
TOTAL_VERTICES = 1 << N

# EVOLVE-BLOCK-START
def _is_valid_snake(path):
    """Check if path is a valid induced path in Q_N."""
    n = len(path)
    if n == 0:
        return True
    used = set(path)
    if len(used) != n:
        return False
    for i in range(n - 1):
        diff = path[i] ^ path[i + 1]
        if diff & (diff - 1) != 0:  # not a power of 2
            return False
    for i in range(n):
        for j in range(i + 2, n):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True

def _greedy_extend(path, used, bit_order, deadline):
    """Try to extend path greedily using given bit_order with 2-ply lookahead.
    Returns extended path or None."""
    cur = path[-1]
    candidates = []
    for b in bit_order:
        v = cur ^ (1 << b)
        if v in used:
            continue
        # Check against all previous vertices (except the one we're extending from)
        ok = True
        for p in path[:-1]:
            if (v ^ p).bit_count() == 1:
                ok = False
                break
        if ok:
            candidates.append(v)
    
    if not candidates:
        return None
    
    if len(candidates) == 1:
        return candidates[0]
    
    # 2-ply lookahead: score each candidate by how many further extensions it allows
    best_v = None
    best_score = -1
    for v in candidates:
        score = 0
        # Look ahead: count valid next vertices after choosing v
        for b2 in bit_order:
            v2 = v ^ (1 << b2)
            if v2 in used or v2 == cur:
                continue
            # Check v2 against all path vertices except cur (since v2 would be after v)
            ok2 = True
            for p in path[:-1]:
                if (v2 ^ p).bit_count() == 1:
                    ok2 = False
                    break
            if ok2:
                score += 1
        # Prefer candidates with more future options; tie-break by bit order
        if score > best_score:
            best_score = score
            best_v = v
    
    return best_v

def _local_search(path, used, deadline):
    """Try to improve the snake by inserting vertices or extending ends.
    Returns improved path if found, else original."""
    if time.time() > deadline:
        return path
    
    best = path[:]
    best_len = len(path)
    
    # Try extending both ends multiple times
    max_iters = 200
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try to extend the end
        cur = path[-1]
        extended = False
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
                path.append(v)
                used.add(v)
                extended = True
                break
        
        # Try to extend the beginning
        cur = path[0]
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            ok = True
            for p in path[1:]:
                if (v ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                path.insert(0, v)
                used.add(v)
                extended = True
                break
        
        if not extended:
            # Try a random insertion: remove a vertex and try to reinsert elsewhere
            if len(path) > 5 and random.random() < 0.5:
                idx = random.randint(1, len(path) - 2)
                removed = path.pop(idx)
                used.discard(removed)
                
                # Try to insert at various positions
                inserted = False
                positions = list(range(len(path) + 1))
                random.shuffle(positions)
                for pos in positions[:20]:
                    if time.time() > deadline:
                        break
                    # Check if we can insert removed at position pos
                    prev_v = path[pos - 1] if pos > 0 else None
                    next_v = path[pos] if pos < len(path) else None
                    
                    ok = True
                    # Check adjacency constraints
                    if prev_v is not None and (removed ^ prev_v).bit_count() != 1:
                        ok = False
                    if next_v is not None and (removed ^ next_v).bit_count() != 1:
                        ok = False
                    
                    # Check non-adjacency constraints with all other vertices
                    if ok:
                        for i, p in enumerate(path):
                            if i == pos - 1 or i == pos:
                                continue
                            if (removed ^ p).bit_count() == 1:
                                ok = False
                                break
                    
                    if ok:
                        path.insert(pos, removed)
                        used.add(removed)
                        inserted = True
                        break
                
                if not inserted:
                    # Put it back
                    path.insert(idx, removed)
                    used.add(removed)
        
        if len(path) > best_len:
            best = path[:]
            best_len = len(path)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    best_path = []
    best_len = 0
    
    # Multiple restarts with different random bit orders
    random.seed(42)
    
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        # Generate a random bit order for this attempt
        bit_order = list(range(N))
        random.shuffle(bit_order)
        
        # Start with a random vertex
        start_vertex = random.randint(0, TOTAL_VERTICES - 1)
        path = [start_vertex]
        used = {start_vertex}
        
        # Greedy extension with lookahead
        while time.time() < deadline:
            nxt = _greedy_extend(path, used, bit_order, deadline)
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
        
        # Local search improvement
        if time.time() < deadline - 5:
            path = _local_search(path, used, deadline)
        
        if len(path) > best_len:
            best_path = path[:]
            best_len = len(path)
        
        # If we've found a very good snake, we can be more aggressive with time
        if best_len > 150 and time.time() > deadline - 30:
            break
        # Stop early if we've used too much time on one attempt
        if attempt > 3 and time.time() > deadline - 20:
            break
    
    # Final validation
    if best_path and _is_valid_snake(best_path):
        return best_path
    
    # Fallback: return the greedy result
    path, used = [0], {0}
    while True:
        cur, nxt = path[-1], None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used or any((v ^ p).bit_count() == 1 for p in path[:-1]):
                continue
            nxt = v
            break
        if nxt is None:
            return path
        path.append(nxt)
        used.add(nxt)
# EVOLVE-BLOCK-END
