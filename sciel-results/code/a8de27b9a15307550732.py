"""Ambient integer sets with small largest Sidon subset."""
import time
import itertools
import random

# EVOLVE-BLOCK-START
def _max_sidon_size(arr):
    """Exhaustively compute the largest subset of arr with all unordered pair
    sums unique INCLUDING diagonals (i.e., sums a+b for i<=j in the subset
    must all be distinct)."""
    n = len(arr)
    if n == 0:
        return 0
    # Precompute for each pair (i,j) with i<=j the sum
    # A subset S is Sidon iff all sums a_i + a_j for i<=j, i,j in S are distinct.
    # We do a DFS/backtracking with pruning.
    best = 0
    # Sort to help pruning (optional)
    a = list(arr)
    # Map value to index for quick conflict check via sums
    # We'll use a set of used sums during construction.
    # Order elements by value.
    a_sorted = sorted(a)
    n = len(a_sorted)

    # Precompute pair sums as a matrix for quick access
    # sums[i][j] for i<=j
    sums = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            sums[i][j] = a_sorted[i] + a_sorted[j]

    # Backtracking: build subset incrementally; maintain set of used sums.
    # State: current list of chosen indices, used_sums set.
    # Prune: if len(chosen) + (n - idx) <= best, stop.
    # We iterate over candidate next indices in order.

    def dfs(start, chosen, used_sums):
        nonlocal best
        cur_len = len(chosen)
        if cur_len > best:
            best = cur_len
        # Try adding each element from 'start' onwards
        for k in range(start, n):
            # Check if a_sorted[k] can be added: all sums with existing chosen must be new
            ok = True
            new_sums = []
            for ci in chosen:
                s = sums[ci][k] if ci <= k else sums[k][ci]
                if s in used_sums:
                    ok = False
                    break
                new_sums.append(s)
            if not ok:
                continue
            # Diagonal sum k+k
            s_diag = sums[k][k]
            if s_diag in used_sums:
                continue
            new_sums.append(s_diag)
            # Add k
            chosen.append(k)
            for s in new_sums:
                used_sums.add(s)
            dfs(k+1, chosen, used_sums)
            # Undo
            for s in new_sums:
                used_sums.discard(s)
            chosen.pop()

    dfs(0, [], set())
    return best


def _collision_score(arr):
    """Higher = more additive collisions = likely smaller max Sidon subset.
    Count number of distinct sum values that are achieved by >=2 unordered pairs
    (i<=j). More collisions means fewer distinct sums, harder to be Sidon."""
    n = len(arr)
    sum_counts = {}
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            sum_counts[s] = sum_counts.get(s, 0) + 1
    # Count pairs that share a sum
    collisions = 0
    for s, c in sum_counts.items():
        if c >= 2:
            collisions += c - 1
    # Also penalize large range (sparsity)
    rng = max(arr) - min(arr)
    return collisions, rng


def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    rng = random.Random(42)

    # Initial candidate: dense small range
    # [0,128] but we want small range to force collisions
    # Start with a set in [0, 25]
    best_arr = list(range(12))  # [0..11], range 11
    best_sidon = _max_sidon_size(best_arr)
    best_score = _collision_score(best_arr)

    # Local search: try to reduce max_sidon_size
    # Moves: replace one element with another in [0,128], or shift all by offset,
    # or compress into a smaller window.
    # We'll do iterative improvement with random restarts.

    # Precompute: for a given set, max_sidon_size is expensive.
    # Use collision_score as a fast surrogate for ranking, but verify with exact.

    def try_candidate(arr):
        nonlocal best_arr, best_sidon, best_score
        if len(set(arr)) != 12:
            return False
        if any(x < 0 or x > 128 for x in arr):
            return False
        # Fast surrogate
        col, rng_val = _collision_score(arr)
        # Exact check
        ms = _max_sidon_size(arr)
        if ms < best_sidon:
            best_sidon = ms
            best_arr = list(arr)
            best_score = (col, rng_val)
            return True
        elif ms == best_sidon:
            # Prefer more collisions / smaller range as tiebreaker
            if (col, -rng_val) > (best_score[0], -best_score[1]):
                best_arr = list(arr)
                best_score = (col, rng_val)
            return True
        return False

    # Strategy 1: try dense windows of size W in [0,128]
    # For each window [s, s+W), pick 12 elements that maximize collisions.
    # A dense window of size ~15-20 should give max_sidon around 4-5.
    for W in range(12, 40):
        if time.time() > deadline:
            break
        for start in range(0, 129 - W + 1, 3):
            if time.time() > deadline:
                break
            window = list(range(start, start + W))
            if len(window) < 12:
                continue
            # Try to pick 12 from window that maximize collisions
            # Greedy: start with all consecutive, then swap to increase collisions
            cand = window[:12]
            # Local improvement within window
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(12):
                    for v in window:
                        if v in cand:
                            continue
                        new_cand = list(cand)
                        new_cand[i] = v
                        if len(set(new_cand)) != 12:
                            continue
                        # Fast surrogate
                        col_new, _ = _collision_score(new_cand)
                        col_cur, _ = _collision_score(cand)
                        if col_new > col_cur:
                            cand = new_cand
                            improved = True
                            break
                    if improved:
                        break
            try_candidate(cand)

    # Strategy 2: random restarts with local search
    for restart in range(200):
        if time.time() > deadline:
            break
        # Random dense set
        W = rng.randint(14, 30)
        start = rng.randint(0, 128 - W)
        window = list(range(start, start + W))
        cand = rng.sample(window, 12)
        try_candidate(cand)

        # Local search: swap elements
        for _ in range(50):
            if time.time() > deadline:
                break
            i = rng.randint(0, 11)
            # Try replacing cand[i] with a nearby value
            for delta in [-3, -2, -1, 1, 2, 3]:
                new_v = cand[i] + delta
                if new_v < 0 or new_v > 128:
                    continue
                if new_v in cand:
                    continue
                new_cand = list(cand)
                new_cand[i] = new_v
                if try_candidate(new_cand):
                    cand = new_cand
                    break

    # Strategy 3: targeted search for known-good structures
    # Sets with many small differences create collisions
    # e.g., {0,1,2,3,4,5,6,7,8,9,10,11} has max_sidon = ?
    # Let's try arithmetic progressions and near-APs
    for start in range(0, 20):
        if time.time() > deadline:
            break
        for d in [1, 2]:
            if time.time() > deadline:
                break
            cand = []
            for k in range(12):
                v = start + k * d
                if v <= 128:
                    cand.append(v)
            if len(cand) == 12:
                try_candidate(cand)

    # Final: ensure we return a valid 12-element set
    if len(set(best_arr)) != 12 or any(x < 0 or x > 128 for x in best_arr):
        best_arr = list(range(12))

    return list(best_arr)
# EVOLVE-BLOCK-END
