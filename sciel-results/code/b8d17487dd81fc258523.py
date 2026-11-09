import numpy as np
import time

def _multiplicity(A, n=128):
    """Return array of representation counts for sums in [0, 2n].
    Unordered pairs, equal summands allowed. A is a sorted list of ints in [0,n]."""
    # Use convolution: indicator vector f of length n+1, f[a]=1 if a in A
    f = np.zeros(2 * n + 1, dtype=np.int64)
    for a in A:
        f[a] = 1
    # f * f gives ordered pairs count for each sum
    conv = np.convolve(f, f, mode='full')  # length 4n+1, index s = count of ordered pairs (a,b) with a+b=s
    # For unordered with equal allowed: count_unordered(s) = (ordered(s) + (1 if s even and s/2 in A else 0)) / 2
    # Because ordered counts (a,b) and (b,a) separately; diagonal (a,a) counted once.
    rep = np.zeros(2 * n + 1, dtype=np.int64)
    for s in range(2 * n + 1):
        ordered = conv[s]
        diag = 1 if (s % 2 == 0 and (s // 2) < len(f) and f[s // 2] == 1) else 0
        rep[s] = (ordered + diag) // 2
    return rep

def _peak(A, n=128):
    rep = _multiplicity(A, n)
    # target interval [128, 256]
    return int(rep[128:257].max())

def _is_valid(A, n=128):
    rep = _multiplicity(A, n)
    return bool(np.all(rep[128:257] >= 1))

def _symmetric_from_half(half, n=128):
    """half is a set of ints in [0, n//2]. Build symmetric A in [0,n]."""
    A = set()
    for a in half:
        A.add(a)
        A.add(n - a)
    return sorted(A)

def construct():
    t0 = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    n = 128
    mid = n // 2  # 64

    # Initial half-set: choose a balanced starting point.
    # Start with half containing a spread of values to ensure coverage.
    # A good start: take a "dense enough" symmetric set.
    # Let's start with half = a random-ish balanced set of size ~30 in [0,64].
    # Actually, to guarantee initial validity, use a known-good structure.

    # Strategy: build A so that it's a "basis". A simple valid A: 
    # A = [0, 64, 128] gives sums {0,64,128,192,256} - not enough.
    # We need coverage of [128,256]. 

    # Let's use a greedy constructive start: 
    # Start with half = range(0, 65) would give A = all [0,128] (full), valid but high peak.
    # Then we trim via local search to reduce peak while keeping validity.

    # Better: start with a moderately dense symmetric set, then local-search.
    # Initial: half = every other value + some, to get |A| ~ 64-80.

    # Let's start with half = set of all even numbers in [0,64] plus a few odds.
    # Actually simplest valid start: A = full [0,128]. Peak will be high but valid.
    # Then simulated annealing removes elements to lower peak while maintaining validity.

    # But removing from full set: each removal reduces pair counts.
    # Let's start with a sparser valid set to have more room.

    # Constructive valid start: use two interleaved arithmetic progressions.
    # Let A contain all multiples of 2 in [0,128] (65 elements) -> sums are even in [0,256].
    # That only covers even targets. Need odd too.
    # Add all multiples of 2 shifted: actually add one odd element? 
    # If A has both even and odd, sums cover all parities.
    # Let A = {0,2,4,...,128} ∪ {1,3,5,...,127}? That's full.
    # 
    # Let's just start with full [0,128] and trim.

    A = set(range(0, n + 1))
    # Ensure valid
    assert _is_valid(sorted(A), n)

    best_A = sorted(A)
    best_peak = _peak(best_A, n)

    # Simulated annealing: randomly remove or add elements, accept if peak improves or valid maintained.
    # Objective: minimize peak, subject to validity.
    # We'll do: try to remove a random element; if still valid, compute new peak; 
    # accept if peak <= best_peak (prefer lower). Also try swaps.

    import random
    random.seed(42)

    current_A = set(A)
    current_peak = best_peak

    def try_move():
        nonlocal current_A, current_peak, best_A, best_peak
        # Choose a move: remove, add, or swap
        move = random.choice(['remove', 'add', 'swap'])
        if move == 'remove':
            if len(current_A) <= 3:
                return
            a = random.choice(sorted(current_A))
            new_A = set(current_A)
            new_A.discard(a)
            new_list = sorted(new_A)
            if not _is_valid(new_list, n):
                return
            new_peak = _peak(new_list, n)
            # Accept if peak improves or equal (prefer smaller set for tie)
            if new_peak < current_peak or (new_peak == current_peak and len(new_A) < len(current_A)):
                current_A = new_A
                current_peak = new_peak
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = sorted(current_A)
        elif move == 'add':
            # Add a random element not in A
            candidates = [x for x in range(n + 1) if x not in current_A]
            if not candidates:
                return
            a = random.choice(candidates)
            new_A = set(current_A)
            new_A.add(a)
            new_list = sorted(new_A)
            if not _is_valid(new_list, n):
                return
            new_peak = _peak(new_list, n)
            if new_peak < current_peak or (new_peak == current_peak and len(new_A) < len(current_A)):
                current_A = new_A
                current_peak = new_peak
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = sorted(current_A)
        elif move == 'swap':
            if len(current_A) < 2:
                return
            a = random.choice(sorted(current_A))
            candidates = [x for x in range(n + 1) if x not in current_A]
            if not candidates:
                return
            b = random.choice(candidates)
            new_A = set(current_A)
            new_A.discard(a)
            new_A.add(b)
            new_list = sorted(new_A)
            if not _is_valid(new_list, n):
                return
            new_peak = _peak(new_list, n)
            if new_peak < current_peak or (new_peak == current_peak and len(new_A) < len(current_A)):
                current_A = new_A
                current_peak = new_peak
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = sorted(current_A)

    # Run local search with time budget
    while time.time() - t0 < DEADLINE:
        # Batch of moves
        for _ in range(200):
            if time.time() - t0 > DEADLINE:
                break
            try_move()
        # Periodic restart: if stuck, try a different random seed perturbation
        # (simple: continue)

    # Ensure best_A is valid (should be)
    if not _is_valid(best_A, n):
        # Fallback: return a known valid set
        best_A = sorted(range(0, n + 1))

    return list(best_A)
