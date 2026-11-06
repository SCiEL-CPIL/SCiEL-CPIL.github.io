"""Snake-in-the-box Q_9: backtracking search with time budget."""
import time
import random

N = 9
TOTAL_VERTS = 1 << N
DEADLINE = 120.0
SAFETY_MARGIN = 3.0  # seconds before hard limit


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN

    best_path = []
    best_len = 0

    # Precompute bit masks
    bit_masks = [1 << b for b in range(N)]

    def time_left():
        return deadline - time.time()

    def is_valid_extension(path, used, forbidden, v):
        """Check if v can be added to the path (not used, not adjacent to any non-last vertex)."""
        if v in used:
            return False
        if v in forbidden:
            return False
        return True

    def build_forbidden(path):
        """Build set of vertices at Hamming distance 1 from any path vertex (except last)."""
        forbidden = set()
        for v in path:
            for b in range(N):
                nb = v ^ bit_masks[b]
                if nb != path[-1]:  # last vertex is the current endpoint, its neighbors are candidates
                    forbidden.add(nb)
        # Also add all path vertices to forbidden (can't reuse)
        forbidden.update(path)
        return forbidden

    def greedy_extend(path, used, forbidden, start_time, deadline):
        """Greedy extension from current path. Returns extended path."""
        while time.time() < deadline:
            cur = path[-1]
            found = False
            for b in range(N):
                v = cur ^ bit_masks[b]
                if v in used or v in forbidden:
                    continue
                # Valid extension
                path.append(v)
                used.add(v)
                # Update forbidden: add neighbors of v (except cur which is path[-2])
                for b2 in range(N):
                    nb = v ^ bit_masks[b2]
                    if nb != cur:
                        forbidden.add(nb)
                found = True
                break
            if not found:
                break
        return path

    def backtrack(path, used, forbidden, depth, max_depth, start_time, deadline):
        """Backtracking search. Returns best path found."""
        nonlocal best_path, best_len

        if time.time() >= deadline:
            return

        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]

        if len(path) >= max_depth:
            return

        cur = path[-1]

        # Generate candidates: neighbors of cur not in used/forbidden
        candidates = []
        for b in range(N):
            v = cur ^ bit_masks[b]
            if v not in used and v not in forbidden:
                candidates.append((b, v))

        if not candidates:
            return

        # Heuristic: prefer candidates that have more available neighbors (more flexibility)
        # Compute for each candidate how many of its neighbors are still free
        def flexibility(v):
            free = 0
            for b in range(N):
                nb = v ^ bit_masks[b]
                if nb not in used and nb not in forbidden and nb != cur:
                    free += 1
            return free

        candidates.sort(key=lambda x: flexibility(x[1]), reverse=True)

        # Try each candidate
        for b, v in candidates:
            if time.time() >= deadline:
                return

            # Add v
            path.append(v)
            used.add(v)

            # Compute new forbidden additions
            new_forbidden = []
            for b2 in range(N):
                nb = v ^ bit_masks[b2]
                if nb != cur and nb not in forbidden:
                    new_forbidden.append(nb)
                    forbidden.add(nb)

            backtrack(path, used, forbidden, depth + 1, max_depth, start_time, deadline)

            # Undo
            path.pop()
            used.discard(v)
            for nb in new_forbidden:
                forbidden.discard(nb)

    # Strategy: run multiple searches with different starting points and randomizations
    # First: greedy from 0 to get a baseline
    path = [0]
    used = {0}
    forbidden = set()
    # Build initial forbidden: neighbors of 0 (all of them since 0 is the only vertex)
    for b in range(N):
        forbidden.add(bit_masks[b])
    # Also 0 itself
    forbidden.add(0)

    path = greedy_extend(path, used, forbidden, start_time, deadline)
    if len(path) > best_len:
        best_len = len(path)
        best_path = path[:]

    # Now try backtracking from various starting points
    # Use a set of tried starts to avoid duplicates
    tried_starts = set()
    random.seed(42)

    # Try starting from 0 with backtracking (limited depth for speed)
    for attempt in range(50):
        if time.time() >= deadline:
            break

        # Choose a starting vertex: mix of 0 and random
        if attempt < 5:
            start = 0
        else:
            start = random.randint(0, TOTAL_VERTS - 1)

        if start in tried_starts:
            continue
        tried_starts.add(start)

        # Initialize path
        path = [start]
        used = {start}
        forbidden = set()
        for b in range(N):
            nb = start ^ bit_masks[b]
            forbidden.add(nb)
        forbidden.add(start)

        # Greedy extend first to get a decent starting path
        max_greedy_steps = 60  # limit greedy to save time for backtracking
        step_count = 0
        while time.time() < deadline and step_count < max_greedy_steps:
            cur = path[-1]
            found = False
            for b in range(N):
                v = cur ^ bit_masks[b]
                if v in used or v in forbidden:
                    continue
                path.append(v)
                used.add(v)
                for b2 in range(N):
                    nb = v ^ bit_masks[b2]
                    if nb != cur:
                        forbidden.add(nb)
                found = True
                step_count += 1
                break
            if not found:
                break

        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]

        # Now do limited backtracking from the end of this greedy path
        # Try to extend further with backtracking
        remaining_time = deadline - time.time()
        if remaining_time < 1.0:
            continue

        # Backtrack with limited depth
        max_bt_depth = min(30, int(remaining_time * 5))  # adaptive
        backtrack(path, used, forbidden, 0, max_bt_depth, start_time, deadline)

    # Final: try a few more random restarts with pure backtracking from scratch
    for attempt in range(20):
        if time.time() >= deadline:
            break

        start = random.randint(0, TOTAL_VERTS - 1)
        if start in tried_starts:
            continue
        tried_starts.add(start)

        path = [start]
        used = {start}
        forbidden = set()
        for b in range(N):
            forbidden.add(start ^ bit_masks[b])
        forbidden.add(start)

        # Pure backtracking with limited depth
        max_bt_depth = min(20, int((deadline - time.time()) * 3))
        if max_bt_depth < 2:
            break
        backtrack(path, used, forbidden, 0, max_bt_depth, start_time, deadline)

    # Return the best path found
    if len(best_path) >= 2:
        return best_path
    else:
        # Fallback: return the greedy path from 0
        path = [0]
        used = {0}
        forbidden = set()
        for b in range(N):
            forbidden.add(bit_masks[b])
        forbidden.add(0)
        path = greedy_extend(path, used, forbidden, start_time, deadline)
        return path


# Quick test
if __name__ == "__main__":
    result = construct()
    print(f"Length: {len(result)}")
    print(f"Edges: {len(result) - 1}")
    # Validate
    valid = True
    for i in range(len(result) - 1):
        diff = (result[i] ^ result[i+1]).bit_count()
        if diff != 1:
            valid = False
            print(f"Invalid edge at {i}: {result[i]} -> {result[i+1]}, diff={diff}")
            break
    for i in range(len(result)):
        for j in range(i+2, len(result)):
            diff = (result[i] ^ result[j]).bit_count()
            if diff == 1:
                valid = False
                print(f"Invalid non-consecutive pair at {i},{j}: {result[i]},{result[j]}")
                break
        if not valid:
            break
    print(f"Valid: {valid}")
