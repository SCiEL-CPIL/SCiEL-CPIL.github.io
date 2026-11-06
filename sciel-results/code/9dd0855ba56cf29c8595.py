"""Snake-in-the-box Q_9: Warnsdorff heuristic with backtracking and time budget."""
import time
import random
import sys

N = 9
ALL_MASK = (1 << N) - 1
DEADLINE_SECONDS = 110  # safety margin before 120s kill

def _popcount(x):
    return x.bit_count()

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    best_path = [0]
    best_len = 1  # 1 vertex = 0 edges, but we track vertices; edges = len-1

    # We'll use a backtracking search with Warnsdorff heuristic.
    # State: path (list of vertices), used (set of vertices in path)
    # For efficiency, we track:
    #   - path: list
    #   - used: set of used vertices
    #   - For each candidate next vertex v (adjacent to path[-1]), it must not be in used
    #     and must not be at distance 1 from any path[i] for i < len(path)-1 (non-consecutive).
    #     Actually the condition is: for all i with |i - (len(path)-1)| > 1, dist(v, path[i]) >= 2.
    #     Equivalently, v must not be at Hamming distance 1 from any path[i] where i <= len(path)-3.

    def time_up():
        return time.time() >= deadline

    # Precompute bit positions
    bit_positions = list(range(N))

    # Warnsdorff: for a candidate next vertex v, count how many valid moves it would have
    # from v (i.e., neighbors w of v such that w not in used and dist(w, path[i]) >= 2 for i <= len(path)-2 after adding v).
    # This is expensive; we approximate or compute on the fly.

    # Let's implement a recursive/iterative backtracking with heuristic ordering.

    def count_valid_exits(v, path, used, path_len):
        """Count how many neighbors of v are valid next steps if we were at v (with current path).
        A neighbor w of v is valid if:
          - w not in used
          - dist(w, path[i]) >= 2 for all i where i <= path_len - 2 (since v would be at index path_len)
        """
        count = 0
        for b in bit_positions:
            w = v ^ (1 << b)
            if w in used:
                continue
            # Check non-consecutive constraint: w must not be at dist 1 from path[i] for i <= path_len - 2
            ok = True
            for i in range(path_len - 1):  # indices 0 .. path_len-2
                if (w ^ path[i]).bit_count() == 1:
                    ok = False
                    break
            if ok:
                count += 1
        return count

    def try_extend(path, used, path_len):
        """Try to extend the path using Warnsdorff heuristic with backtracking.
        Returns True if extension was made (path modified), False if stuck."""
        if time_up():
            return False

        cur = path[-1]
        # Generate candidate next vertices
        candidates = []
        for b in bit_positions:
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check non-consecutive: v must not be at dist 1 from path[i] for i <= path_len - 2
            ok = True
            for i in range(path_len - 1):
                if (v ^ path[i]).bit_count() == 1:
                    ok = False
                    break
            if not ok:
                continue
            candidates.append(v)

        if not candidates:
            return False

        # Warnsdorff: sort by number of valid exits (fewest first)
        # To save time, only compute exits if more than a few candidates
        if len(candidates) > 1:
            scored = []
            for v in candidates:
                exits = count_valid_exits(v, path, used, path_len)
                scored.append((exits, v))
            scored.sort(key=lambda x: (x[0], x[1]))
            candidates = [v for _, v in scored]

        # Try candidates in order; if a candidate leads to a dead end later, backtracking handles it.
        # Here we just pick the first candidate (greedy with heuristic ordering).
        # For full backtracking, we'd need to recurse. Let's do limited backtracking.
        v = candidates[0]
        path.append(v)
        used.add(v)
        return True

    # Main search: greedy with heuristic, plus periodic restarts with randomization
    # Strategy:
    # 1. Do a greedy Warnsdorff path from 0.
    # 2. If stuck, try backtracking a few levels.
    # 3. Periodically restart with random starting vertex or random tie-breaking.

    # Let's implement a more robust backtracking search with a depth limit for backtracking.
    # We'll keep a stack of (path, used, path_len) states for backtracking.

    # Actually, let's do iterative deepening backtracking with Warnsdorff.
    # Given time constraints, let's do:
    # - Greedy Warnsdorff from multiple random starts
    # - Keep the best path found

    # For backtracking, we need to be able to undo. Let's use a simple approach:
    # Maintain path and used. When stuck, backtrack k steps and try alternative at each level.

    def backtrack_search(path, used, path_len, max_backtrack=50):
        """Attempt to extend the path with backtracking up to max_backtrack levels."""
        nonlocal best_path, best_len
        if time_up():
            return

        # Try to extend greedily with Warnsdorff
        while True:
            if time_up():
                return
            if not try_extend(path, used, path_len):
                break
            path_len += 1

        # Update best
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]

        # If we can't extend, try backtracking
        if time_up():
            return

        # Backtrack up to max_backtrack levels
        for bt in range(1, min(max_backtrack, path_len) + 1):
            if time_up():
                return
            if path_len <= bt:
                break
            # Undo bt steps
            for _ in range(bt):
                v = path.pop()
                used.remove(v)
                path_len -= 1

            # Now try alternative extensions from this state
            # We need to remember which candidates we already tried.
            # For simplicity, shuffle candidates and try each.
            cur = path[-1]
            candidates = []
            for b in bit_positions:
                v = cur ^ (1 << b)
                if v in used:
                    continue
                ok = True
                for i in range(path_len - 1):
                    if (v ^ path[i]).bit_count() == 1:
                        ok = False
                        break
                if ok:
                    candidates.append(v)

            if not candidates:
                continue

            # Shuffle to avoid deterministic bias
            random.shuffle(candidates)

            # Try each candidate
            for v in candidates:
                if time_up():
                    break
                path.append(v)
                used.add(v)
                new_len = path_len + 1
                # Recurse with reduced backtrack budget
                backtrack_search(path, used, new_len, max_backtrack=max(10, max_backtrack - bt))
                # Undo
                path.pop()
                used.remove(v)

            # Re-extend greedily after backtracking attempt
            # Actually the recursive call already did extension.
            break  # Only backtrack once per call to avoid exponential blowup

    # Run multiple greedy passes with different random seeds for tie-breaking
    for trial in range(200):
        if time_up():
            break

        # Random start
        if trial == 0:
            start = 0
        else:
            start = random.randint(0, ALL_MASK)

        path = [start]
        used = {start}
        path_len = 1

        # Greedy extension with Warnsdorff
        while not time_up():
            if not try_extend(path, used, path_len):
                break
            path_len += 1

        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]

        # If the path is long enough, try some backtracking
        if len(path) > best_len - 5 and not time_up():
            # Try a limited backtracking from the current end
            # Save state
            saved_path = path[:]
            saved_used = used.copy()
            saved_len = path_len

            # Attempt backtracking
            backtrack_search(path, used, path_len, max_backtrack=30)

            if len(path) <= best_len:
                # Restore
                path = saved_path
                used = saved_used
                path_len = saved_len

    # Return the best path found
    if best_len <= 1:
        return [0]
    return best_path

# Pre-seed random for reproducibility within a run
random.seed(42)
