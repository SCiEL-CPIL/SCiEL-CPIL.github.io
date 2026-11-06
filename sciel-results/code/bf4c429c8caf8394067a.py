"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 9
NUM_VERTS = 1 << N

# Precompute bit masks for each bit position
BIT_MASKS = [1 << b for b in range(N)]

# EVOLVE-BLOCK-START

def _is_valid_extension(v, path, used, path_set):
    """Check if v can be appended to the path (induced path constraint)."""
    if v in used:
        return False
    # v must differ from all non-consecutive vertices by at least 2 bits
    # Consecutive: path[-1] (differs by 1 bit by construction)
    # Non-consecutive: path[0:-1]
    for p in path[:-1]:
        if (v ^ p).bit_count() == 1:
            return False
    return True


def _warnsdorff_greedy(seed_path, rng, time_limit=None):
    """
    Extend a seed path using Warnsdorff's rule (prefer moves with fewest onward options).
    Returns the extended path.
    """
    path = list(seed_path)
    used = set(path)
    
    # Precompute neighbors for quick access
    # For each vertex, its 9 neighbors
    def get_neighbors(v):
        return [v ^ (1 << b) for b in range(N)]
    
    def count_onward_options(v, used_set):
        """Count how many valid onward moves exist from v (not used, not adjacent to path[:-1])."""
        count = 0
        for nb in get_neighbors(v):
            if nb in used_set:
                continue
            # Check induced constraint: nb must not be adjacent to any non-consecutive vertex
            # The "non-consecutive" vertices relative to nb being next after v are path[:-1]
            # Actually, if we add nb after v, then nb's non-consecutive neighbors are path[:-1]
            # and v is consecutive. So we need nb not adjacent to path[:-1].
            valid = True
            for p in path[:-1]:
                if (nb ^ p).bit_count() == 1:
                    valid = False
                    break
            if valid:
                count += 1
        return count
    
    while True:
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check induced path constraint
            valid = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    valid = False
                    break
            if not valid:
                continue
            candidates.append(v)
        
        if not candidates:
            break
        
        if len(candidates) == 1:
            nxt = candidates[0]
        else:
            # Warnsdorff: pick candidate with fewest onward options
            # For efficiency, if time is tight, just pick randomly
            if time_limit is not None and (time.time() - time_limit) > 0:
                nxt = rng.choice(candidates)
            else:
                scores = []
                for v in candidates:
                    # Count onward options from v
                    onward = 0
                    used_with_v = used | {v}
                    for nb in get_neighbors(v):
                        if nb in used_with_v:
                            continue
                        valid_nb = True
                        for p in path[:-1]:
                            if (nb ^ p).bit_count() == 1:
                                valid_nb = False
                                break
                        if valid_nb:
                            onward += 1
                    scores.append(onward)
                min_score = min(scores)
                best_candidates = [v for v, s in zip(candidates, scores) if s == min_score]
                nxt = rng.choice(best_candidates)
        
        path.append(nxt)
        used.add(nxt)
        
        if time_limit is not None and (time.time() - time_limit) > 0:
            break
    
    return path


def _backtracking_search(seed_path, rng, max_depth, time_budget):
    """
    Try to extend seed_path further using limited backtracking.
    Returns the best path found.
    """
    best_path = list(seed_path)
    best_len = len(seed_path)
    
    def dfs(path, used, depth):
        nonlocal best_path, best_len
        if len(path) > best_len:
            best_path = list(path)
            best_len = len(path)
        
        if depth >= max_depth:
            return
        if time_budget[0] < 0:
            return
        
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            valid = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    valid = False
                    break
            if valid:
                candidates.append(v)
        
        if not candidates:
            return
        
        # Sort by Warnsdorff
        if len(candidates) > 1:
            def onward_count(v):
                used_v = used | {v}
                cnt = 0
                for nb in [v ^ (1 << b) for b in range(N)]:
                    if nb in used_v:
                        continue
                    valid_nb = True
                    for p in path[:-1]:
                        if (nb ^ p).bit_count() == 1:
                            valid_nb = False
                            break
                    if valid_nb:
                        cnt += 1
                return cnt
            
            candidates.sort(key=onward_count)
        
        for v in candidates:
            if time_budget[0] < 0:
                return
            path.append(v)
            used.add(v)
            dfs(path, used, depth + 1)
            path.pop()
            used.discard(v)
    
    start_time = time.time()
    time_budget = [time_budget]
    dfs(list(seed_path), set(seed_path), 0)
    return best_path


def construct():
    """
    Construct a long induced path (snake) in Q_9 using Warnsdorff's rule
    with randomized restarts and limited backtracking.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    rng = random.Random(42)
    
    best_path = None
    best_len = 0
    
    # Generate diverse seeds using different random starting points and greedy extensions
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        
        # Random starting vertex
        start = rng.randint(0, NUM_VERTS - 1)
        
        # Build a greedy seed path from this start
        path = [start]
        used = {start}
        
        # Greedy with Warnsdorff for seed construction
        while True:
            cur = path[-1]
            candidates = []
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                valid = True
                for p in path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    candidates.append(v)
            
            if not candidates:
                break
            
            if len(candidates) == 1:
                nxt = candidates[0]
            else:
                # Warnsdorff: pick with fewest onward options
                if len(candidates) <= 3:
                    scores = []
                    for v in candidates:
                        used_v = used | {v}
                        cnt = 0
                        for nb in [v ^ (1 << b) for b in range(N)]:
                            if nb in used_v:
                                continue
                            valid_nb = True
                            for p in path[:-1]:
                                if (nb ^ p).bit_count() == 1:
                                    valid_nb = False
                                    break
                            if valid_nb:
                                cnt += 1
                        scores.append(cnt)
                    min_s = min(scores)
                    best_c = [v for v, s in zip(candidates, scores) if s == min_s]
                    nxt = rng.choice(best_c)
                else:
                    # For efficiency with many candidates, random pick
                    nxt = rng.choice(candidates)
            
            path.append(nxt)
            used.add(nxt)
            
            if time.time() > deadline:
                break
        
        # Try to extend with backtracking for a bit
        if time.time() < deadline:
            remaining_time = deadline - time.time()
            if remaining_time > 0.5:
                # Use a small time budget for backtracking
                bt_budget = min(remaining_time * 0.3, 2.0)
                extended = _backtracking_search(path, rng, max_depth=10, time_budget=bt_budget)
                if len(extended) > len(path):
                    path = extended
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        # If we've found a very long path, we can be more aggressive
        if best_len >= 150:
            # Try more restarts with different strategies
            pass
        
        # Break early if we're running out of time
        if time.time() > deadline - 5:
            break
    
    if best_path is None:
        # Fallback: simple greedy from 0
        path = [0]
        used = {0}
        while True:
            cur = path[-1]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                valid = True
                for p in path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    nxt = v
                    break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
        best_path = path
    
    return best_path

# EVOLVE-BLOCK-END
