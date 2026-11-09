"""Combined snake construction for Q_9: bidirectional Warnsdorff growth + fast vectorized validation + limited backtracking."""
import time
import random
import numpy as np

N = 9
MAX_VERTS = 2 ** N

# Precompute popcount lookup table for 9-bit values
_POPCOUNT_TABLE = np.array([i.bit_count() for i in range(512)], dtype=np.int32)

def _popcount(arr):
    """Vectorized popcount for array of small integers (< 512)."""
    return _POPCOUNT_TABLE[arr.astype(np.int64)]

def _validate(path):
    """Validate that path is a valid induced path in Q_N."""
    n = len(path)
    if n == 0:
        return False
    seen = set(path)
    if len(seen) != n:
        return False
    for i in range(n):
        if path[i] < 0 or path[i] >= MAX_VERTS:
            return False
    # Check consecutive differ by exactly 1 bit
    for i in range(n - 1):
        d = (path[i] ^ path[i + 1]).bit_count()
        if d != 1:
            return False
    # Check non-consecutive differ by at least 2 bits
    for i in range(n):
        for j in range(i + 2, n):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True


def construct():
    """
    Construct the longest induced path (snake) in Q_9 within the time budget.
    Strategy:
      1. Run original greedy to establish baseline (104).
      2. Run bidirectional Warnsdorff growth with diverse seeds.
      3. Apply limited backtracking to extend best candidates.
      4. Return best valid result found.
    """
    start_time = time.time()
    deadline = start_time + 130  # 150s limit, 20s safety margin

    def _deadline_hit():
        return time.time() >= deadline

    best_path = None
    best_len = 0

    # --- Step 1: Original greedy baseline ---
    path_greedy, used_greedy = [0], {0}
    while not _deadline_hit():
        cur = path_greedy[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used_greedy:
                continue
            ok = True
            for p in path_greedy[:-1]:
                if (v ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                nxt = v
                break
        if nxt is None:
            break
        path_greedy.append(nxt)
        used_greedy.add(nxt)

    if len(path_greedy) > best_len:
        best_len = len(path_greedy)
        best_path = path_greedy[:]

    # --- Helper: fast validity check for extension at an end ---
    def _is_valid_append(v, path, used_set):
        """Check if v can be appended to end of path."""
        if v in used_set:
            return False
        for p in path[:-1]:
            if (v ^ p).bit_count() == 1:
                return False
        return True

    def _is_valid_prepend(v, path, used_set):
        """Check if v can be prepended to start of path."""
        if v in used_set:
            return False
        for p in path[1:]:
            if (v ^ p).bit_count() == 1:
                return False
        return True

    # --- Helper: count safe options (Warnsdorff heuristic) ---
    def _count_safe_options(v, path, used_set, end_index):
        """Count how many neighbors of v would be valid to extend from.
        Fewer options = more urgent to place v (Warnsdorff)."""
        count = 0
        for b in range(N):
            u = v ^ (1 << b)
            if u in used_set:
                continue
            # Check if u would be a valid extension from v
            if end_index == 1:
                # u would be appended after v; check against path[:-1]
                valid = True
                for p in path[:-1]:
                    if (u ^ p).bit_count() == 1:
                        valid = False
                        break
            else:
                # u would be prepended before v; check against path[1:]
                valid = True
                for p in path[1:]:
                    if (u ^ p).bit_count() == 1:
                        valid = False
                        break
            if valid:
                count += 1
        return count

    # --- Bidirectional Warnsdorff growth ---
    def _bidirectional_grow(start_vertex, bit_order, rng, max_steps=None):
        """Grow a snake bidirectionally from start_vertex using Warnsdorff's rule."""
        path = [start_vertex]
        used = {start_vertex}
        grow_left = False

        steps = 0
        max_steps = max_steps or 500

        while not _deadline_hit() and steps < max_steps:
            if grow_left:
                cur = path[0]
                end_index = 0
            else:
                cur = path[-1]
                end_index = 1

            # Find all valid neighbors
            candidates = []
            for b in bit_order:
                v = cur ^ (1 << b)
                if v in used:
                    continue
                if end_index == 1:
                    if not _is_valid_append(v, path, used):
                        continue
                else:
                    if not _is_valid_prepend(v, path, used):
                        continue
                candidates.append(v)

            if not candidates:
                break

            # Warnsdorff: pick candidate with fewest safe options
            scored = []
            for v in candidates:
                opts = _count_safe_options(v, path, used, end_index)
                scored.append((opts, rng.random(), v))
            scored.sort(key=lambda x: (x[0], x[1]))

            v = scored[0][2]

            if end_index == 1:
                path.append(v)
            else:
                path.insert(0, v)
            used.add(v)
            steps += 1
            grow_left = not grow_left

        return path

    # --- Limited backtracking extension ---
    def _backtrack_extend(path, used_set, rng, max_depth=4):
        """Try to extend path using limited backtracking at the end."""
        if _deadline_hit():
            return path

        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if _is_valid_append(v, path, used_set):
                candidates.append(v)

        if not candidates:
            return path

        rng.shuffle(candidates)

        best_local = path[:]
        best_local_len = len(path)

        def _bt(current_path, current_used, depth):
            nonlocal best_local, best_local_len

            if _deadline_hit():
                return

            if len(current_path) > best_local_len:
                best_local = current_path[:]
                best_local_len = len(current_path)

            if depth >= max_depth:
                return

            cur = current_path[-1]
            cands = []
            for b in range(N):
                v = cur ^ (1 << b)
                if _is_valid_append(v, current_path, current_used):
                    cands.append(v)

            if not cands:
                return

            rng.shuffle(cands)
            for v in cands:
                if _deadline_hit():
                    break
                new_path = current_path + [v]
                new_used = current_used | {v}
                _bt(new_path, new_used, depth + 1)

        _bt(path, used_set, 0)
        return best_local

    # --- Main search loop ---
    rng = random.Random(42)

    # Try various starting vertices
    start_vertices = [0, MAX_VERTS - 1]
    for _ in range(6):
        start_vertices.append(rng.randint(0, MAX_VERTS - 1))

    attempts = 0
    max_attempts = 300

    for sv in start_vertices:
        if _deadline_hit() or attempts >= max_attempts:
            break

        # Try multiple bit orderings
        for trial in range(8):
            if _deadline_hit() or attempts >= max_attempts:
                break

            bit_order = list(range(N))
            rng.shuffle(bit_order)

            path = _bidirectional_grow(sv, bit_order, rng)
            attempts += 1

            if len(path) > best_len:
                if _validate(path):
                    best_len = len(path)
                    best_path = path[:]
                    # Try to extend with backtracking
                    used_set = set(best_path)
                    extended = _backtrack_extend(best_path, used_set, rng, max_depth=3)
                    if len(extended) > best_len and _validate(extended):
                        best_len = len(extended)
                        best_path = extended[:]

    # Continue with random restarts if time permits
    while not _deadline_hit() and attempts < max_attempts:
        sv = rng.randint(0, MAX_VERTS - 1)
        bit_order = list(range(N))
        rng.shuffle(bit_order)

        path = _bidirectional_grow(sv, bit_order, rng)
        attempts += 1

        if len(path) > best_len:
            if _validate(path):
                best_len = len(path)
                best_path = path[:]
                used_set = set(best_path)
                extended = _backtrack_extend(best_path, used_set, rng, max_depth=3)
                if len(extended) > best_len and _validate(extended):
                    best_len = len(extended)
                    best_path = extended[:]

    # Final fallback
    if best_path is None:
        return path_greedy

    # Final validation
    if not _validate(best_path):
        return path_greedy

    return best_path
