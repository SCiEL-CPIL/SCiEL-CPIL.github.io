"""Direct CSP approach: backtracking with forward checking for minimal 4-AP coloring."""
import time
import random

_DEADLINE = 88.0
_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _generate_progressions(n=128):
    """Generate all nonconstant 4-term progressions (a, a+d, a+2d, a+3d) 1-indexed."""
    progs = []
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            progs.append((a, a + d, a + 2 * d, a + 3 * d))
    return progs

_PROGRESSIONS = _generate_progressions(128)
_NUM_PROGS = len(_PROGRESSIONS)
N = 128

# For each position (0-indexed), store which progressions contain it and at what index (0-3)
_POS_TO_PROGS = [[] for _ in range(N)]
for idx, prog in enumerate(_PROGRESSIONS):
    for i, p in enumerate(prog):
        _POS_TO_PROGS[p - 1].append((idx, i))

# For each progression, store the 4 positions as 0-indexed
_PROG_POS = [[p - 1 for p in prog] for prog in _PROGRESSIONS]

def _check_progression_valid(prog_idx, colors):
    """Check if progression prog_idx has at least 3 distinct colors.
    Returns True if valid or incomplete, False if definitely invalid."""
    pos = _PROG_POS[prog_idx]
    c0, c1, c2, c3 = colors[pos[0]], colors[pos[1]], colors[pos[2]], colors[pos[3]]
    
    # If any unassigned, we can't say it's invalid yet (unless all assigned and < 3 distinct)
    if -1 in (c0, c1, c2, c3):
        return True  # Incomplete, assume valid for now
    
    # All assigned: check distinct count
    distinct = len({c0, c1, c2, c3})
    return distinct >= 3

def _is_valid_full(colors):
    """Check if entire coloring is valid."""
    for i in range(_NUM_PROGS):
        if not _check_progression_valid(i, colors):
            return False
    return True

def _backtrack_solve(k, seed, max_depth=200000):
    """Solve using backtracking with forward checking.
    Returns coloring if found, None otherwise."""
    rng = random.Random(seed)
    colors = [-1] * N
    depth = 0
    
    # Order positions by how many progressions they're in (most constrained first)
    # But for simplicity, try natural order first, then random orders
    order = list(range(N))
    rng.shuffle(order)
    
    def backtrack(pos_idx):
        nonlocal depth
        if _time_left() < 1.0:
            return False
        depth += 1
        if depth > max_depth:
            return False
        
        if pos_idx == N:
            return _is_valid_full(colors)
        
        pos = order[pos_idx]
        # Try colors in random order
        color_order = list(range(k))
        rng.shuffle(color_order)
        
        for c in color_order:
            colors[pos] = c
            
            # Forward check: for each progression containing pos, check if it's now invalid
            valid = True
            for (prog_idx, slot) in _POS_TO_PROGS[pos]:
                if not _check_progression_valid(prog_idx, colors):
                    valid = False
                    break
            
            if valid:
                if backtrack(pos_idx + 1):
                    return True
            
            colors[pos] = -1
        
        return False
    
    if backtrack(0):
        return colors
    return None

def _greedy_with_backtracking(k, seed):
    """Hybrid: greedy assignment with backtracking on conflicts."""
    rng = random.Random(seed)
    colors = [-1] * N
    
    # Assign positions in order, using greedy with limited backtracking
    for i in range(N):
        pos = i
        available = []
        for c in range(k):
            colors[pos] = c
            valid = True
            for (prog_idx, slot) in _POS_TO_PROGS[pos]:
                if not _check_progression_valid(prog_idx, colors):
                    valid = False
                    break
            if valid:
                available.append(c)
            colors[pos] = -1
        
        if not available:
            return None
        
        colors[pos] = rng.choice(available)
    
    return colors if _is_valid_full(colors) else None

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_colors = None
    best_nc = N + 1
    
    # Try k=3 first (optimal if possible)
    for k in [3, 4, 5, 6, 7, 8, 9, 10]:
        if _time_left() < 2.0:
            break
        
        found = False
        # Try multiple strategies and seeds
        for seed in range(100):
            if _time_left() < 1.0:
                break
            
            # Strategy 1: Pure backtracking
            colors = _backtrack_solve(k, seed, max_depth=50000)
            if colors is not None:
                nc = len(set(colors))
                if nc < best_nc:
                    best_nc = nc
                    best_colors = list(colors)
                if nc == k:
                    found = True
                    break
                elif nc < k:
                    # We used fewer than k colors, which is great
                    found = True
                    break
            
            # Strategy 2: Greedy with verification
            if _time_left() < 0.5:
                break
            colors = _greedy_with_backtracking(k, seed)
            if colors is not None:
                nc = len(set(colors))
                if nc < best_nc:
                    best_nc = nc
                    best_colors = list(colors)
                if nc <= k:
                    found = True
                    break
        
        if found and best_nc <= k:
            # We found a valid coloring with k or fewer colors
            # Try to see if we can do better with fewer colors
            if k > 3 and best_nc > 3:
                continue  # Try smaller k
            else:
                break
    
    # If we found a valid coloring, return it
    if best_colors is not None:
        return best_colors
    
    # Fallback: use many colors to guarantee validity
    # Each position gets a unique color
    return list(range(N))
