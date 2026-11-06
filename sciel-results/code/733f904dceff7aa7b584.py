import time
import random
import sys

N = 9
SIZE = 1 << N
FULL_MASK = SIZE - 1
TIME_LIMIT = 115  # safety margin before 150s kill

# Precompute neighbors for all 512 vertices
NEIGHBORS = [[] for _ in range(SIZE)]
for v in range(SIZE):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))


def _is_valid_snake(path):
    """Check if path is a valid induced path in the hypercube."""
    L = len(path)
    if L == 0:
        return True
    used = set(path)
    for i in range(L - 1):
        if (path[i] ^ path[i + 1]).bit_count() != 1:
            return False
    for i in range(L):
        for j in range(i + 2, L):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True


def _greedy_extend(path, used, blocked):
    """Extend path from the tail using greedy heuristic. Returns number of extensions made."""
    count = 0
    while True:
        cur = path[-1]
        best_nb = None
        best_score = -1
        for nb in NEIGHBORS[cur]:
            if nb in used or nb in blocked:
                continue
            # Score: fewer future options is better (Warnsdorff)
            opts = 0
            for nb2 in NEIGHBORS[nb]:
                if nb2 in used or nb2 == cur:
                    continue
                if nb2 in blocked:
                    continue
                # Check if nb2 would be blocked by path[:-1]
                would_block = False
                for p in path[:-1]:
                    if (nb2 ^ p).bit_count() == 1:
                        would_block = True
                        break
                if not would_block:
                    opts += 1
            score = -opts  # fewer options = higher priority
            if score > best_score:
                best_score = score
                best_nb = nb
        if best_nb is None:
            break
        path.append(best_nb)
        used.add(best_nb)
        # Update blocked set
        for p in path[:-1]:
            for nb in NEIGHBORS[best_nb]:
                if nb != p and (nb ^ p).bit_count() == 1:
                    blocked.add(nb)
        count += 1
    return count


def _backtrack_solve(start, rng, deadline, target_len):
    """
    Backtracking solver with constraint propagation.
    Returns the best path found before deadline or reaching target_len.
    """
    best_path = [start]
    used = {start}
    blocked = set()  # vertices that are adjacent to non-consecutive path vertices
    
    # For a single vertex, no blocked vertices initially
    
    def dfs(path, used, blocked):
        nonlocal best_path
        
        if len(path) > len(best_path):
            best_path = list(path)
        
        if len(path) >= target_len:
            return True
        
        if time.time() > deadline:
            return False
        
        cur = path[-1]
        
        # Find all valid next moves
        candidates = []
        for nb in NEIGHBORS[cur]:
            if nb in used or nb in blocked:
                continue
            # Check if nb is adjacent to any path vertex except cur (path[-2])
            valid = True
            for i, p in enumerate(path[:-1]):
                if (nb ^ p).bit_count() == 1:
                    valid = False
                    break
            if not valid:
                continue
            candidates.append(nb)
        
        if not candidates:
            return False
        
        # Order candidates by Warnsdorff's rule: prefer those with fewer future options
        def warnsdorff_score(nb):
            opts = 0
            for nb2 in NEIGHBORS[nb]:
                if nb2 in used or nb2 == cur:
                    continue
                if nb2 in blocked:
                    continue
                # Check if nb2 would be blocked by current path
                would_block = False
                for p in path[:-1]:
                    if (nb2 ^ p).bit_count() == 1:
                        would_block = True
                        break
                if not would_block:
                    opts += 1
            return opts
        
        candidates.sort(key=warnsdorff_score)
        
        # Add some randomization to avoid deterministic traps
        if len(candidates) > 1:
            # With some probability, pick a random candidate from the top half
            if rng.random() < 0.3:
                top_half = candidates[:max(1, len(candidates) // 2)]
                chosen = rng.choice(top_half)
            else:
                chosen = candidates[0]
        else:
            chosen = candidates[0]
        
        # Try the chosen move
        path.append(chosen)
        used.add(chosen)
        
        # Update blocked set: chosen is now adjacent to all its neighbors except cur
        new_blocked = set()
        for nb in NEIGHBORS[chosen]:
            if nb != cur:
                # nb is blocked if it's adjacent to any path vertex except chosen
                for p in path[:-2]:  # all except last two
                    if (nb ^ p).bit_count() == 1:
                        new_blocked.add(nb)
                        break
        
        success = dfs(path, used, new_blocked)
        
        # Backtrack
        path.pop()
        used.discard(chosen)
        
        if success:
            return True
        
        return False
    
    dfs([start], {start}, set())
    return best_path


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    rng = random.Random(20260715)
    
    best_path = None
    best_len = 0
    
    # Phase 1: Try backtracking from multiple starts with increasing target lengths
    starts = [0, 1, 2, 3, FULL_MASK, FULL_MASK - 1]
    # Add some random starts
    for _ in range(20):
        starts.append(rng.randint(0, FULL_MASK))
    
    for start in starts:
        if time.time() > deadline - 10:
            break
        
        # Try to find a path of length at least best_len + 1
        target = max(best_len + 1, 110)
        path = _backtrack_solve(start, rng, min(deadline - 5, start_time + 15), target)
        
        if len(path) > best_len:
            best_path = path
            best_len = len(path)
            # Reset RNG for reproducibility within this phase
            rng = random.Random(20260715 + len(starts))
    
    # Phase 2: If we have time, try more aggressive backtracking with higher targets
    if best_path is None:
        best_path = [0]
        best_len = 1
    
    remaining_deadline = deadline - 5
    attempts = 0
    while time.time() < remaining_deadline and attempts < 50:
        start = rng.randint(0, FULL_MASK)
        target = best_len + 1
        path = _backtrack_solve(start, rng, min(remaining_deadline, time.time() + 8), target)
        
        if len(path) > best_len:
            best_path = path
            best_len = len(path)
        
        attempts += 1
    
    # Final validation
    if best_path and not _is_valid_snake(best_path):
        # Fallback to a simple valid path
        best_path = [0]
    
    return best_path
