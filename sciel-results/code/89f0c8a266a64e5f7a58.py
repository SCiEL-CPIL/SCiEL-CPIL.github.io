"""Snake-in-the-box in Q_9: local search + randomized restarts to maximize induced path length."""
import time
import random

N = 9
MASK = (1 << N) - 1

# Precompute neighbors for all 512 vertices
NEIGHBORS = [[] for _ in range(1 << N)]
for v in range(1 << N):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

def _is_valid_snake(path):
    """Check if path is a valid induced path (snake)."""
    if len(path) < 2:
        return True
    used = set(path)
    for i in range(len(path) - 1):
        if (path[i] ^ path[i+1]).bit_count() != 1:
            return False
    # Check non-consecutive vertices differ by at least 2 bits
    for i in range(len(path)):
        for j in range(i+2, len(path)):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True

def _greedy_extend(path, rng):
    """Extend path greedily with randomized tie-breaking. Returns extended path."""
    used = set(path)
    while True:
        cur = path[-1]
        candidates = []
        for nb in NEIGHBORS[cur]:
            if nb in used:
                continue
            # Check if adding nb violates induced path condition
            valid = True
            for p in path[:-1]:
                if (nb ^ p).bit_count() == 1:
                    valid = False
                    break
            if valid:
                candidates.append(nb)
        if not candidates:
            break
        # Randomly pick among candidates (or lowest for determinism in some cases)
        nxt = rng.choice(candidates)
        path.append(nxt)
        used.add(nxt)
    return path

def _local_search(path, rng, max_iters=20000):
    """Improve snake via random insertions/deletions/swaps. Returns best found."""
    best = list(path)
    best_len = len(best)
    used_best = set(best)
    
    for _ in range(max_iters):
        p = list(best)
        used = set(p)
        improved = False
        
        op = rng.random()
        
        if op < 0.4:
            # Try to insert a vertex somewhere
            # Pick a random position
            pos = rng.randint(0, len(p))
            # Try to find a vertex that can be inserted at position pos
            if pos == 0:
                prev_v = None
                next_v = p[0]
            elif pos == len(p):
                prev_v = p[-1]
                next_v = None
            else:
                prev_v = p[pos-1]
                next_v = p[pos]
            
            # Find candidate vertices adjacent to prev_v (or next_v if at ends)
            candidates = []
            if prev_v is not None:
                candidates = list(NEIGHBORS[prev_v])
            if next_v is not None:
                # Also need adjacency to next_v
                candidates = [c for c in candidates if c in NEIGHBORS[next_v]]
            else:
                # Just need to not conflict with rest of path
                pass
            
            for cand in candidates:
                if cand in used:
                    continue
                # Check induced path condition with all other vertices
                valid = True
                for i, v in enumerate(p):
                    if i == pos - 1 or i == pos:  # Skip the neighbors we're connecting to
                        continue
                    if (cand ^ v).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    # Also need to ensure cand is adjacent to prev_v and next_v
                    if prev_v is not None and (cand ^ prev_v).bit_count() != 1:
                        continue
                    if next_v is not None and (cand ^ next_v).bit_count() != 1:
                        continue
                    # Insert
                    p.insert(pos, cand)
                    used.add(cand)
                    improved = True
                    break
        
        elif op < 0.7:
            # Try to delete a vertex (to allow re-insertion elsewhere)
            if len(p) > 2:
                pos = rng.randint(1, len(p) - 2)  # Don't delete endpoints
                removed = p.pop(pos)
                used.discard(removed)
                # Now try to extend from both ends or insert elsewhere
                # First try extending from end
                end = p[-1]
                for nb in NEIGHBORS[end]:
                    if nb in used:
                        continue
                    valid = True
                    for v in p[:-1]:
                        if (nb ^ v).bit_count() == 1:
                            valid = False
                            break
                    if valid:
                        p.append(nb)
                        used.add(nb)
                        improved = True
                        break
                if not improved:
                    # Try extending from start
                    start = p[0]
                    for nb in NEIGHBORS[start]:
                        if nb in used:
                            continue
                        valid = True
                        for v in p[1:]:
                            if (nb ^ v).bit_count() == 1:
                                valid = False
                            break
                        if valid:
                            p.insert(0, nb)
                            used.add(nb)
                            improved = True
                            break
        
        else:
            # Swap two non-adjacent vertices? Or try 2-opt like reversal
            if len(p) >= 4:
                i = rng.randint(0, len(p) - 3)
                j = rng.randint(i + 2, len(p) - 1)
                # Reverse segment p[i+1..j]
                new_p = p[:i+1] + p[i+1:j+1][::-1] + p[j+1:]
                # Check if still a valid path
                if all((new_p[k] ^ new_p[k+1]).bit_count() == 1 for k in range(len(new_p)-1)):
                    # Check induced condition
                    valid = True
                    for a in range(len(new_p)):
                        for b in range(a+2, len(new_p)):
                            if (new_p[a] ^ new_p[b]).bit_count() == 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        p = new_p
                        improved = True
        
        if improved and len(p) > best_len:
            best = p
            best_len = len(p)
            used_best = set(best)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    rng = random.Random(42)
    
    best_path = None
    best_len = 0
    
    # Initial greedy snake
    path = [0]
    path = _greedy_extend(path, rng)
    best_path = list(path)
    best_len = len(path)
    
    # Multiple rounds of local search with different random seeds
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Local search on current best
        improved = _local_search(best_path, rng, max_iters=5000)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)
        
        # Periodically try fresh greedy starts with different seeds
        if iteration % 5 == 0:
            fresh = [0]
            fresh = _greedy_extend(fresh, rng)
            if len(fresh) > best_len:
                best_path = list(fresh)
                best_len = len(fresh)
        
        # Occasionally try greedy from a random start
        if iteration % 3 == 0:
            start_v = rng.randint(0, (1 << N) - 1)
            fresh = [start_v]
            fresh = _greedy_extend(fresh, rng)
            if len(fresh) > best_len:
                best_path = list(fresh)
                best_len = len(fresh)
    
    # Final check
    if best_path is None or len(best_path) < 2:
        # Fallback: simple greedy
        path = [0]
        used = {0}
        while True:
            cur = path[-1]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                if any((v ^ p).bit_count() == 1 for p in path[:-1]):
                    continue
                nxt = v
                break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
        best_path = path
    
    return best_path
