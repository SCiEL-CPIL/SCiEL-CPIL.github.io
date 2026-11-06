"""Snake-in-the-box in Q_9: Backtracking + Simulated Annealing with 2-bit flip moves."""
import time
import random
import sys

N = 9
MASK = (1 << N) - 1
SIZE = 1 << N

# Precompute neighbors
NEIGHBORS = [[] for _ in range(SIZE)]
for v in range(SIZE):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

# Precompute neighbor bitmask for each vertex (for fast blocked/used checks)
NEIGHBOR_MASK = [0] * SIZE
for v in range(SIZE):
    m = 0
    for nb in NEIGHBORS[v]:
        m |= (1 << nb)
    NEIGHBOR_MASK[v] = m

def _solve_backtrack(start, deadline, rng):
    """
    Depth-first search with backtracking and Warnsdorff's rule using bitmasks.
    Returns the longest path found within the deadline.
    """
    best_path = [start]
    best_len = 1
    sys.setrecursionlimit(10000)
    
    def dfs(path, used_mask, blocked_mask):
        nonlocal best_path, best_len
        if time.time() > deadline:
            return
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
            
        cur = path[-1]
        cur_bit = 1 << cur
        
        # new_blocked for next step: neighbors of all vertices in current path
        # = blocked_mask (neighbors of path[:-1]) | neighbors of cur
        new_blocked_mask = blocked_mask | NEIGHBOR_MASK[cur]
        
        candidates = []
        for nb in NEIGHBORS[cur]:
            nb_bit = 1 << nb
            if used_mask & nb_bit:
                continue
            if blocked_mask & nb_bit:
                continue
            
            # Warnsdorff score: count valid next steps from nb
            # Next step y must be: in NEIGHBORS[nb], not in used|{nb}, not in new_blocked
            score = 0
            nb_used_mask = used_mask | nb_bit
            for y in NEIGHBORS[nb]:
                y_bit = 1 << y
                if nb_used_mask & y_bit:
                    continue
                if new_blocked_mask & y_bit:
                    continue
                score += 1
            
            candidates.append((score, nb))
        
        if not candidates:
            return
        
        candidates.sort(key=lambda x: x[0])
        
        for score, nb in candidates:
            path.append(nb)
            dfs(path, used_mask | (1 << nb), new_blocked_mask)
            path.pop()
            
            if time.time() > deadline:
                return

    dfs([start], 1 << start, 0)
    return best_path

def _is_valid_snake(path):
    """Verify that path is a valid snake."""
    if not path:
        return True
    used = set(path)
    if len(used) != len(path):
        return False
    for i in range(len(path)):
        for j in range(i + 2, len(path)):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    for i in range(len(path) - 1):
        if (path[i] ^ path[i + 1]).bit_count() != 1:
            return False
    return True


def _greedy_extend(path, rng):
    """Extend path greedily from the end using Warnsdorff's rule."""
    if not path:
        return path
    path = list(path)
    used_mask = 0
    for v in path:
        used_mask |= (1 << v)
    
    # blocked_mask: neighbors of path[:-1]
    blocked_mask = 0
    for v in path[:-1]:
        blocked_mask |= NEIGHBOR_MASK[v]
    
    while True:
        cur = path[-1]
        candidates = []
        for nb in NEIGHBORS[cur]:
            nb_bit = 1 << nb
            if used_mask & nb_bit:
                continue
            if blocked_mask & nb_bit:
                continue
            # Count onward moves
            score = 0
            for y in NEIGHBORS[nb]:
                y_bit = 1 << y
                if used_mask & y_bit:
                    continue
                if (1 << nb) & y_bit:
                    continue
                # y must not be adjacent to any vertex in path (all of them, since nb becomes non-last)
                # Actually y must not be in blocked_mask | NEIGHBOR_MASK[cur]
                if (blocked_mask | NEIGHBOR_MASK[cur]) & y_bit:
                    continue
                score += 1
            candidates.append((score, nb))
        
        if not candidates:
            break
        
        candidates.sort(key=lambda x: x[0])
        # Pick the best, with some randomness
        best_score = candidates[0][0]
        best_cands = [nb for s, nb in candidates if s == best_score]
        nb = rng.choice(best_cands)
        
        path.append(nb)
        used_mask |= (1 << nb)
        blocked_mask |= NEIGHBOR_MASK[cur]
    
    return path


def _sa_search(path, deadline, rng):
    """
    Simulated Annealing with 2-bit flip moves that preserve validity.
    A 2-bit flip: replace a vertex v_i in the path with v_i ^ (1<<b1) ^ (1<<b2)
    such that the new vertex is not in the path and the path remains a valid snake.
    """
    if not path:
        return path
    
    best_path = list(path)
    best_len = len(best_path)
    
    def _path_valid(p):
        if len(p) < 2:
            return True
        used = set(p)
        if len(used) != len(p):
            return False
        for i in range(len(p) - 1):
            if (p[i] ^ p[i+1]).bit_count() != 1:
                return False
        for i in range(len(p)):
            for j in range(i + 2, len(p)):
                if (p[i] ^ p[j]).bit_count() == 1:
                    return False
        return True
    
    # Precompute for each position, which 2-bit flips are valid
    # This is expensive, so we'll do it on the fly
    
    current_path = list(path)
    T = 5.0
    T_min = 0.1
    cooling = 0.9995
    
    while time.time() < deadline and T > T_min:
        if not current_path:
            break
        
        # Try a 2-bit flip at a random position
        pos = rng.randint(0, len(current_path) - 1)
        v = current_path[pos]
        
        # Generate all 2-bit flips
        flips = []
        for b1 in range(N):
            for b2 in range(b1 + 1, N):
                nv = v ^ (1 << b1) ^ (1 << b2)
                if nv == v:
                    continue
                # Check if nv is already in the path (except at position pos)
                in_path = False
                for i, u in enumerate(current_path):
                    if i != pos and u == nv:
                        in_path = True
                        break
                if in_path:
                    continue
                # Check adjacency constraints
                # nv must be adjacent to current_path[pos-1] if pos > 0
                # nv must be adjacent to current_path[pos+1] if pos < len-1
                # nv must not be adjacent to any other vertex
                valid = True
                if pos > 0:
                    if (nv ^ current_path[pos-1]).bit_count() != 1:
                        valid = False
                if pos < len(current_path) - 1:
                    if (nv ^ current_path[pos+1]).bit_count() != 1:
                        valid = False
                if valid:
                    for i, u in enumerate(current_path):
                        if i == pos or i == pos - 1 or i == pos + 1:
                            continue
                        if (nv ^ u).bit_count() == 1:
                            valid = False
                            break
                if valid:
                    flips.append(nv)
        
        if not flips:
            T *= cooling
            continue
        
        nv = rng.choice(flips)
        new_path = current_path[:]
        new_path[pos] = nv
        
        # Check if new path is valid
        if _path_valid(new_path):
            # Try to extend from both ends
            extended_end = _greedy_extend(new_path, rng)
            if len(extended_end) > len(current_path):
                current_path = extended_end
                if len(current_path) > best_len:
                    best_len = len(current_path)
                    best_path = list(current_path)
            else:
                current_path = new_path
        else:
            # Accept with probability exp(-delta/T) where delta is penalty
            # For invalid moves, we just reject
            pass
        
        T *= cooling
    
    # Final greedy extension
    final_path = _greedy_extend(best_path, rng)
    if len(final_path) > best_len:
        best_path = final_path
        best_len = len(final_path)
    
    return best_path


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    rng = random.Random(42)
    
    best_path = []
    best_len = 0
    
    # Phase 1: Backtracking search with multiple starts
    # Use bitmasks for speed
    starts = [0] + [rng.randint(0, SIZE - 1) for _ in range(20)]
    
    for s in starts:
        if time.time() > deadline - 30:  # Reserve 30s for SA
            break
        
        sub_deadline = min(deadline - 30, time.time() + 8)
        
        path = _solve_backtrack(s, sub_deadline, rng)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
    
    # Phase 2: Simulated Annealing refinement
    if best_path and time.time() < deadline - 5:
        sa_deadline = deadline - 5
        # Run SA on the best path
        sa_path = _sa_search(best_path, sa_deadline, rng)
        if len(sa_path) > best_len:
            best_path = sa_path
            best_len = len(sa_path)
        
        # Also try SA on a few other good paths if time permits
        if time.time() < deadline - 10:
            for s in starts[:5]:
                if time.time() > deadline - 10:
                    break
                sub_deadline = min(deadline - 10, time.time() + 3)
                path = _solve_backtrack(s, sub_deadline, rng)
                if len(path) > best_len - 10:  # Only refine if close to best
                    sa_path = _sa_search(path, min(deadline - 5, time.time() + 5), rng)
                    if len(sa_path) > best_len:
                        best_path = sa_path
                        best_len = len(sa_path)
    
    # Final greedy extension
    if best_path and time.time() < deadline - 2:
        final_path = _greedy_extend(best_path, rng)
        if len(final_path) > best_len:
            best_path = final_path
            best_len = len(final_path)
    
    return best_path
