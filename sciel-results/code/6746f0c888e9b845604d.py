"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 9
TOTAL = 1 << N

# EVOLVE-BLOCK-START
def _is_valid_extension(v, path, used):
    """Check if v can be appended to path: not used, and differs from all non-last path vertices by >= 2 bits."""
    if v in used:
        return False
    # Must differ from all vertices except the last by at least 2 bits
    for p in path[:-1]:
        if (v ^ p).bit_count() == 1:
            return False
    return True

def _count_valid_extensions(cur, used, path):
    """Count how many valid next vertices exist from cur."""
    count = 0
    for b in range(N):
        v = cur ^ (1 << b)
        if _is_valid_extension(v, path, used):
            count += 1
    return count

def _greedy_with_lookahead(seed_bit_order=None, rng=None):
    """Build a snake using greedy with lookahead: prefer flips that maximize future options."""
    if rng is None:
        rng = random.Random()
    
    path = [0]
    used = {0}
    
    while True:
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if _is_valid_extension(v, path, used):
                # Lookahead: count valid extensions from v
                lookahead = _count_valid_extensions(v, used, path)
                candidates.append((lookahead, rng.random(), b, v))
        
        if not candidates:
            break
        
        # Sort by lookahead descending, then random tiebreaker
        candidates.sort(key=lambda x: (-x[0], x[1]))
        best_lookahead, _, best_b, best_v = candidates[0]
        
        path.append(best_v)
        used.add(best_v)
    
    return path

def _try_extend(path, used, rng, max_iters=500):
    """Try to extend the path using randomized local search."""
    path = list(path)
    used = set(used)
    
    for _ in range(max_iters):
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if _is_valid_extension(v, path, used):
                candidates.append(v)
        
        if not candidates:
            break
        
        # Pick randomly among candidates (or best)
        v = rng.choice(candidates)
        path.append(v)
        used.add(v)
    
    return path

def construct():
    """Construct a snake in Q_9 using randomized greedy with lookahead and multiple restarts."""
    best_path = []
    best_len = 0
    start_time = time.time()
    time_limit = 110  # Leave some margin for 120s limit
    
    # Try multiple random restarts with different random seeds
    for trial in range(200):
        if time.time() - start_time > time_limit:
            break
        
        rng = random.Random(trial)
        
        # Randomize the starting vertex
        start_v = rng.randint(0, TOTAL - 1)
        
        # Build initial path with randomized bit ordering
        path = [start_v]
        used = {start_v}
        
        while True:
            cur = path[-1]
            candidates = []
            for b in range(N):
                v = cur ^ (1 << b)
                if _is_valid_extension(v, path, used):
                    # Lookahead score
                    lookahead = _count_valid_extensions(v, used, path)
                    # Add some randomness to escape local optima
                    score = lookahead + rng.random() * 0.5
                    candidates.append((score, b, v))
            
            if not candidates:
                break
            
            # Pick the best candidate
            candidates.sort(key=lambda x: -x[0])
            _, _, best_v = candidates[0]
            path.append(best_v)
            used.add(best_v)
        
        # Try to extend further with randomized search
        path = _try_extend(path, used, rng, max_iters=200)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        # Early stop if we've been doing well
        if best_len >= 150 and trial > 50:
            # Continue but reduce effort
            pass
    
    # Final verification and return
    if len(best_path) > len(_greedy_with_lookahead(rng=random.Random(0))):
        return best_path
    else:
        # Fallback to a known-good greedy
        return _greedy_with_lookahead(rng=random.Random(0))
# EVOLVE-BLOCK-END
