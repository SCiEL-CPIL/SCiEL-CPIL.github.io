import time
import numpy as np

N = 9
TOTAL = 1 << N
MASK = TOTAL - 1

# Precompute neighbor lists for fast lookup
_NEIGHBORS = np.array([[v ^ (1 << b) for b in range(N)] for v in range(TOTAL)], dtype=np.uint32)

def _elapsed(deadline):
    return time.time() >= deadline

def _grow_bidirectional(path, used_set, deadline):
    """Grow the path bidirectionally using max_degree strategy."""
    used = np.zeros(TOTAL, dtype=bool)
    for v in path:
        used[v] = True
    
    path = list(path)
    
    while not _elapsed(deadline):
        best_end = None
        best_choice = None
        best_score = -1
        
        for end in (0, 1):  # 0 = head, 1 = tail
            cur = path[0] if end == 0 else path[-1]
            connect_idx = 0 if end == 0 else len(path) - 1
            
            best_for_end = None
            best_for_end_score = -1
            
            for nb in _NEIGHBORS[cur]:
                if used[nb]:
                    continue
                
                # Check validity: nb must differ from all path vertices except cur
                # We can precompute a forbidden set or check against path
                # For speed, we check against path
                valid = True
                for i, p in enumerate(path):
                    if i == connect_idx:
                        continue
                    if (nb ^ p).bit_count() == 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Max_degree: count how many valid extensions exist from nb
                score = 0
                for nn in _NEIGHBORS[nb]:
                    if used[nn]:
                        continue
                    # Check nn against path
                    ok = True
                    for p in path:
                        if (nn ^ p).bit_count() == 1:
                            ok = False
                            break
                    if ok:
                        score += 1
                
                if score > best_for_end_score:
                    best_for_end_score = score
                    best_for_end = nb
            
            if best_for_end is not None:
                # Prefer the end with the most options
                if best_for_end_score > best_score:
                    best_score = best_for_end_score
                    best_end = end
                    best_choice = best_for_end
        
        if best_end is None:
            break
        
        # Add the chosen vertex
        if best_end == 1:
            path.append(best_choice)
        else:
            path.insert(0, best_choice)
        used[best_choice] = True
        
        if _elapsed(deadline):
            break
    
    return path

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    best_path = [0]
    best_len = 1
    
    rng = np.random.default_rng(seed=42)
    
    # Run multiple restarts with different starting vertices
    while not _elapsed(deadline):
        start_v = int(rng.integers(TOTAL))
        
        # Start with a greedy path from start_v
        path = [start_v]
        used = {start_v}
        
        # Forward greedy to get an initial path
        while not _elapsed(deadline):
            cur = path[-1]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                # Check against all path vertices except the last
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
        
        if _elapsed(deadline):
            break
        
        # Now grow bidirectionally with max_degree
        path = _grow_bidirectional(path, used, deadline)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
        
        if _elapsed(deadline):
            break
    
    return best_path
