"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _precompute_progressions(n=256):
    """Precompute all arithmetic progressions in [1, n] as index arrays."""
    progressions = []
    for step in range(1, n + 1):
        for start in range(1, n + 1, step):
            indices = list(range(start - 1, n, step))
            if len(indices) >= 1:
                progressions.append(np.array(indices, dtype=np.int32))
    return progressions

def _build_reverse_index(n, progressions):
    """Build mapping: position -> list of progression indices containing it."""
    rev = [[] for _ in range(n)]
    for pi, idx in enumerate(progressions):
        for i in idx:
            rev[i].append(pi)
    return rev

def _init_state(signs, progressions, rev):
    """Initialize prog_sums and abs_freq from signs."""
    m = len(progressions)
    prog_sums = np.zeros(m, dtype=np.int32)
    for pi, idx in enumerate(progressions):
        prog_sums[pi] = np.sum(signs[idx])
    abs_freq = {}
    for s in prog_sums:
        a = abs(s)
        abs_freq[a] = abs_freq.get(a, 0) + 1
    return prog_sums, abs_freq

def _get_max_abs(abs_freq):
    """Get current max absolute sum from frequency dict."""
    if not abs_freq:
        return 0
    return max(k for k, v in abs_freq.items() if v > 0)

def _flip_and_update(pos, signs, prog_sums, abs_freq, rev):
    """Flip position pos, update prog_sums and abs_freq. Returns old max before flip."""
    old_max = _get_max_abs(abs_freq)
    delta = -2 * signs[pos]
    for pi in rev[pos]:
        old_s = prog_sums[pi]
        old_a = abs(old_s)
        abs_freq[old_a] -= 1
        if abs_freq[old_a] == 0:
            del abs_freq[old_a]
        new_s = old_s + delta
        prog_sums[pi] = new_s
        new_a = abs(new_s)
        abs_freq[new_a] = abs_freq.get(new_a, 0) + 1
    signs[pos] = -signs[pos]
    new_max = _get_max_abs(abs_freq)
    return old_max, new_max

def _unflip_and_update(pos, signs, prog_sums, abs_freq, rev):
    """Revert flip at pos. Returns old max before revert (which is current max after flip)."""
    old_max = _get_max_abs(abs_freq)
    delta = -2 * signs[pos]  # signs[pos] is already flipped, so delta to revert
    for pi in rev[pos]:
        old_s = prog_sums[pi]
        old_a = abs(old_s)
        abs_freq[old_a] -= 1
        if abs_freq[old_a] == 0:
            del abs_freq[old_a]
        new_s = old_s + delta
        prog_sums[pi] = new_s
        new_a = abs(new_s)
        abs_freq[new_a] = abs_freq.get(new_a, 0) + 1
    signs[pos] = -signs[pos]
    new_max = _get_max_abs(abs_freq)
    return old_max, new_max

def _greedy_polish(signs, progressions, rev, time_limit):
    """Greedy polish: repeatedly try all single flips, accept if improves max."""
    m = len(progressions)
    prog_sums, abs_freq = _init_state(signs, progressions, rev)
    t0 = time.time()
    improved = True
    while improved and (time.time() - t0) < time_limit:
        improved = False
        cur_max = _get_max_abs(abs_freq)
        for pos in range(256):
            if (time.time() - t0) > time_limit:
                break
            old_max, new_max = _flip_and_update(pos, signs, prog_sums, abs_freq, rev)
            if new_max < cur_max:
                cur_max = new_max
                improved = True
            else:
                _unflip_and_update(pos, signs, prog_sums, abs_freq, rev)
    return signs.copy()

def _sa_search(signs, progressions, rev, rng, time_budget):
    """Simulated annealing with incremental updates."""
    n = 256
    m = len(progressions)
    prog_sums, abs_freq = _init_state(signs, progressions, rev)
    best_signs = signs.copy()
    best_disc = _get_max_abs(abs_freq)
    
    T = 2.0
    cooling = 0.999
    min_T = 0.05
    t0 = time.time()
    
    while time.time() < t0 + time_budget and T > min_T:
        pos = rng.randrange(n)
        old_max, new_max = _flip_and_update(pos, signs, prog_sums, abs_freq, rev)
        
        if new_max < best_disc:
            best_disc = new_max
            best_signs = signs.copy()
        elif new_max <= old_max or rng.random() < np.exp(-(new_max - old_max) / T):
            pass  # accept
        else:
            _unflip_and_update(pos, signs, prog_sums, abs_freq, rev)
        
        T *= cooling
    
    return best_signs, best_disc

def construct():
    n = 256
    t_start = time.time()
    deadline = t_start + 80.0  # safety margin before 90s kill
    
    # Precompute
    progressions = _precompute_progressions(n)
    rev = _build_reverse_index(n, progressions)
    m = len(progressions)
    
    # Multiple restarts
    best_signs = None
    best_disc = float('inf')
    
    # Try several different initializations
    inits = []
    
    # Init 1: alternating
    signs1 = np.array([1 if i % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    inits.append(signs1)
    
    # Init 2: Thue-Morse-like
    def thue_morse(n):
        tm = [1]
        while len(tm) < n:
            tm.extend([-x for x in tm])
        return np.array(tm[:n], dtype=np.int8)
    inits.append(thue_morse(n))
    
    # Init 3: random with fixed seed
    rng0 = random.Random(12345)
    signs3 = np.array([rng0.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
    inits.append(signs3)
    
    # Init 4: another random
    rng0 = random.Random(99999)
    signs4 = np.array([rng0.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
    inits.append(signs4)
    
    # Evaluate each init and pick the best as starting point
    for s in inits:
        ps, af = _init_state(s, progressions, rev)
        d = _get_max_abs(af)
        if d < best_disc:
            best_disc = d
            best_signs = s.copy()
    
    # Run SA from the best initial, with multiple restarts
    rng = random.Random(42)
    num_restarts = 5
    elapsed = time.time() - t_start
    remaining = deadline - t_start - 5.0  # leave 5s for polish
    time_per_restart = remaining / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        if r == 0:
            start_signs = best_signs.copy()
        else:
            # Perturb best_signs
            start_signs = best_signs.copy()
            for _ in range(rng.randrange(10, 50)):
                p = rng.randrange(n)
                start_signs[p] = -start_signs[p]
        
        t_budget = min(time_per_restart, deadline - time.time() - 5.0)
        if t_budget < 0.1:
            break
        
        sa_signs, sa_disc = _sa_search(start_signs, progressions, rev, rng, t_budget)
        
        if sa_disc < best_disc:
            best_disc = sa_disc
            best_signs = sa_signs.copy()
    
    # Final greedy polish
    if time.time() < deadline - 1.0:
        polish_time = min(deadline - time.time() - 0.5, 10.0)
        best_signs = _greedy_polish(best_signs, progressions, rev, polish_time)
        # Verify
        ps, af = _init_state(best_signs, progressions, rev)
        final_disc = _get_max_abs(af)
        if final_disc < best_disc:
            best_disc = final_disc
    
    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
