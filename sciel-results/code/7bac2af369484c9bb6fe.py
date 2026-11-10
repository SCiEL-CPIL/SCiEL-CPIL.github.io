import random
import time
import numpy as np

_START = time.time()
_DEADLINE = 150.0 - 3.0   # safety margin


def _edge_index(i, j):
    # 0-based index for pair (i,j), 0<=i<j<16, in lexicographic order
    return i * 15 - (i * (i + 1)) // 2 + (j - i - 1)


def _build_k4s():
    # For each K4, store its 6 edge indices
    k4s = []
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                for d in range(c + 1, 16):
                    k4s.append((
                        _edge_index(a, b),
                        _edge_index(a, c),
                        _edge_index(a, d),
                        _edge_index(b, c),
                        _edge_index(b, d),
                        _edge_index(c, d),
                    ))
    return k4s


_K4S = _build_k4s()
_K4S_ARR = np.array(_K4S, dtype=np.int32)  # shape (1820, 6)
_NUM_K4 = len(_K4S)


def _colour_counts(colours):
    cnt = [0] * 120
    for c in colours:
        cnt[c] += 1
    return cnt


def _validate_np(colours_arr):
    """Vectorized validation: return True if every K4 has >= 5 distinct colours."""
    # colours_arr: numpy array of shape (120,)
    # For each K4, get the 6 edge colours, check if distinct count >= 5
    edge_cols = colours_arr[_K4S_ARR]  # shape (1820, 6)
    # For each row, count distinct values
    # Trick: for each row, check if any pair is equal in a way that reduces distinct count below 5
    # Distinct count < 5 means at least 2 pairs are equal (pigeonhole: 6 items into <5 bins)
    # Actually: distinct < 5 means at least one colour appears >= 2 times AND total distinct <= 4
    # Simpler: for each row, compute min, max, and check... 
    # Better: for each row of 6 values, the number of distinct values is 6 - (number of "collisions" in a specific sense)
    # Most reliable: for each row, sort and count unique
    # But we can do: for each row, check if all 6 are distinct (6), or 5 distinct, etc.
    # Fastest vectorized approach: for each row, count how many pairs are equal
    # If a row has 6 values, the number of distinct values d satisfies:
    # d = 6 - (number of "extra" occurrences beyond the first for each colour)
    # This is hard to vectorize directly.
    # Alternative: for each row, check if any value appears >= 2 times AND the total distinct is < 5
    # Let's just use: for each row, the number of distinct values = 6 - sum over colours of (count - 1) for count >= 1
    # That's still complex.
    # Practical approach: for each row, check if there exist two different positions with same value, 
    # and if so, check if the remaining 4 also have collisions.
    # Simplest vectorized: for each row, check all C(6,2)=15 pairs. If a pair is equal, 
    # we need to verify the total distinct count.
    # Let me just do: for each row, compute the number of distinct values by checking 
    # if the row has < 5 distinct. A row has < 5 distinct iff there exist at least 
    # 2 pairs (i,j) and (k,l) (possibly sharing an index) such that v[i]=v[j] and v[k]=v[l] 
    # and the set of values has size <= 4.
    # This is getting complicated. Let me use a different approach:
    # For each row, the number of distinct values is at least 5 iff 
    # for every value x, the count of x in the row is such that sum of (count-1) over all x <= 1
    # i.e., at most one "duplicate" pair total.
    # Equivalently: the row has >= 5 distinct iff the number of equal pairs (i<j with v[i]=v[j]) <= 1
    # Wait, that's not right either. If we have values [1,1,1,2,3,4], equal pairs: (0,1),(0,2),(1,2) = 3 pairs, distinct = 4 < 5.
    # If [1,1,2,3,4,5], equal pairs: (0,1) = 1 pair, distinct = 5.
    # If [1,2,3,4,5,6], equal pairs: 0, distinct = 6.
    # So: distinct >= 5 iff number of equal pairs <= 1? 
    # [1,1,1,2,3,4]: equal pairs = 3, distinct = 4. Correct.
    # [1,1,2,2,3,4]: equal pairs = (0,1),(2,3) = 2, distinct = 4. Correct.
    # [1,1,2,3,4,5]: equal pairs = 1, distinct = 5. Correct.
    # [1,1,1,2,3,4]: equal pairs = 3, distinct = 4.
    # So it seems: distinct >= 5 iff num_equal_pairs <= 1.
    # Let me verify: can we have 2 equal pairs and still 5 distinct? 
    # 6 values, 5 distinct means exactly one value appears twice, others once. Equal pairs = 1.
    # 6 values, 4 distinct means either (2,2,1,1) or (3,1,1,1) or (2,2,2) or (4,1,1) etc.
    # (2,2,1,1): equal pairs = 1+1 = 2. Distinct = 4.
    # (3,1,1,1): equal pairs = C(3,2)=3. Distinct = 4.
    # (2,2,2): equal pairs = 1+1+1 = 3. Distinct = 3.
    # So yes: distinct >= 5 iff num_equal_pairs <= 1.
    
    # Compute number of equal pairs for each row
    # For 6 values, there are 15 pairs
    pairs_idx = [(0,1),(0,2),(0,3),(0,4),(0,5),
                 (1,2),(1,3),(1,4),(1,5),
                 (2,3),(2,4),(2,5),
                 (3,4),(3,5),
                 (4,5)]
    num_equal = np.zeros(_NUM_K4, dtype=np.int32)
    for p, q in pairs_idx:
        num_equal += (edge_cols[:, p] == edge_cols[:, q]).astype(np.int32)
    
    # Valid iff all rows have num_equal <= 1
    return np.all(num_equal <= 1)


def _try_merge(colours, cnt, a, b):
    # Return new colours list if merging colour b into a keeps validity, else None.
    if cnt[b] > cnt[a]:
        return None
    new = list(colours)
    for idx, c in enumerate(new):
        if c == b:
            new[idx] = a
    # Validate all K4s using numpy
    new_arr = np.array(new, dtype=np.int32)
    if not _validate_np(new_arr):
        return None
    newcnt = list(cnt)
    newcnt[a] += newcnt[b]
    newcnt[b] = 0
    return new, newcnt


def _num_colours(cnt):
    return sum(1 for x in cnt if x > 0)


def construct():
    # Start from the known valid 17-colour solution
    colours = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    cnt = _colour_counts(colours)
    best = list(colours)
    best_cnt = _num_colours(cnt)

    rng = random.Random(12345)
    used = [c for c in range(120) if cnt[c] > 0]

    # Deterministic greedy merge: repeatedly try to merge the least-used colour
    # into another colour, preferring to keep the palette small.
    def greedy(colours, cnt):
        colours = list(colours)
        cnt = list(cnt)
        nc = _num_colours(cnt)
        progress = True
        while progress:
            progress = False
            used = [c for c in range(120) if cnt[c] > 0]
            if len(used) <= 1:
                break
            # Sort by count ascending: try to eliminate least-used first
            used.sort(key=lambda c: (cnt[c], c))
            b = used[0]  # least used
            if cnt[b] == 0:
                continue
            # Try merging b into each other used colour, most-used first
            targets = [a for a in used if a != b]
            targets.sort(key=lambda c: (-cnt[c], c))
            for a in targets:
                res = _try_merge(colours, cnt, a, b)
                if res is not None:
                    colours, cnt = res
                    nc -= 1
                    progress = True
                    break
        return colours, cnt, nc

    colours, cnt, nc = greedy(colours, cnt)
    if nc < best_cnt:
        best = list(colours)
        best_cnt = nc

    # Local search / hill climbing with random restarts to escape local minima
    def local_search(colours, cnt, iters=20000):
        colours = list(colours)
        cnt = list(cnt)
        nc = _num_colours(cnt)
        for _ in range(iters):
            if time.time() > _DEADLINE:
                break
            used = [c for c in range(120) if cnt[c] > 0]
            if len(used) <= 1:
                break
            # pick a random colour to try to eliminate (prefer least used with some randomness)
            r = rng.random()
            if r < 0.7:
                # 70%: pick from least-used half
                used_sorted = sorted(used, key=lambda c: cnt[c])
                b = used_sorted[rng.randrange(min(len(used_sorted) // 2 + 1, len(used_sorted)))]
            else:
                b = used[rng.randrange(len(used))]
            if cnt[b] == 0:
                continue
            # pick a random target, prefer most-used
            targets = [a for a in used if a != b]
            r2 = rng.random()
            if r2 < 0.6:
                targets_sorted = sorted(targets, key=lambda c: -cnt[c])
                a = targets_sorted[rng.randrange(min(len(targets_sorted) // 2 + 1, len(targets_sorted)))]
            else:
                a = targets[rng.randrange(len(targets))]
            res = _try_merge(colours, cnt, a, b)
            if res is not None:
                colours, cnt = res
                nc -= 1
        return colours, cnt, nc

    # Try several random perturbations + local search
    for restart in range(500):
        if time.time() > _DEADLINE:
            break
        # Start from best, apply a few random single-edge colour changes to perturb
        cur = list(best)
        # small perturbation: change a few edges to random colours from current palette
        num_changes = rng.randint(1, 3)
        curcnt = _colour_counts(cur)
        used_colors = [c for c in range(120) if curcnt[c] > 0]
        for _ in range(num_changes):
            idx = rng.randrange(120)
            cur[idx] = used_colors[rng.randrange(len(used_colors))]
        # validate; if invalid, skip
        cur_arr = np.array(cur, dtype=np.int32)
        if not _validate_np(cur_arr):
            continue
        curcnt = _colour_counts(cur)
        # local search
        cur, curcnt, nc = local_search(cur, curcnt, iters=8000)
        if nc < best_cnt:
            best = list(cur)
            best_cnt = nc
            # try greedy from here too
            gcol, gcnt, gnc = greedy(best, _colour_counts(best))
            if gnc < best_cnt:
                best = list(gcol)
                best_cnt = gnc

    return best
