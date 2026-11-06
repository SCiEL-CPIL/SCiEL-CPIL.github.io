"""Snake-in-the-box in the 9-dimensional hypercube - Local search + bitmask backtracking."""
N = 9
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute neighbor masks and bit masks
    neighbor_mask = [0] * (1 << N)
    for v in range(1 << N):
        mask = 0
        for b in range(N):
            nv = v ^ (1 << b)
            mask |= (1 << nv)
        neighbor_mask[v] = mask

    bit_masks = [1 << b for b in range(N)]

    def _valid_snake(path):
        """Check if path is a valid snake (induced path)."""
        if not path or len(path) < 2:
            return len(path) >= 1
        used = set(path)
        if len(used) != len(path):
            return False
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            if bin(a ^ b).count('1') != 1:
                return False
        # Check non-consecutive distances
        for i in range(len(path)):
            for j in range(i + 2, len(path)):
                if bin(path[i] ^ path[j]).count('1') == 1:
                    return False
        return True

    def _rebuild_forbidden(path):
        """Rebuild forbidden mask from a path."""
        forbidden = 0
        for v in path:
            forbidden |= (1 << v)
        for v in path[:-1]:
            forbidden |= neighbor_mask[v]
        return forbidden

    def _greedy_extend(path, deadline, rng, use_random_tiebreak=False):
        """Greedily extend a path using Warnsdorff heuristic."""
        if not path:
            return path

        forbidden = _rebuild_forbidden(path)
        cur = path[-1]

        while time.time() < deadline:
            candidates = []
            for b in range(N):
                v = cur ^ bit_masks[b]
                if forbidden & (1 << v):
                    continue
                new_forbidden = forbidden | (1 << v) | neighbor_mask[cur]
                onward = 0
                for b2 in range(N):
                    nv = v ^ bit_masks[b2]
                    if not (new_forbidden & (1 << nv)):
                        onward += 1
                candidates.append((onward, b, v))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[0])

            if use_random_tiebreak and len(candidates) > 1:
                # Among the best (lowest onward), pick randomly
                best_onward = candidates[0][0]
                top = [c for c in candidates if c[0] == best_onward]
                _, b, v = rng.choice(top)
            else:
                _, b, v = candidates[0]

            path.append(v)
            forbidden |= (1 << v) | neighbor_mask[cur]
            cur = v

        return path

    def _backtrack_search(start_vertex, deadline, max_depth, rng):
        """Backtracking search with Warnsdorff heuristic using bitmask state."""
        best_path = [start_vertex]
        best_len = 1

        def _search(path, forbidden_mask, depth):
            nonlocal best_path, best_len

            if time.time() > deadline:
                return

            if depth > best_len:
                best_len = depth
                best_path = path[:]

            if max_depth is not None and depth >= max_depth:
                return

            cur = path[-1]

            # Find candidate next vertices
            candidates = []
            for b in range(N):
                v = cur ^ bit_masks[b]
                if forbidden_mask & (1 << v):
                    continue
                new_forbidden = forbidden_mask | (1 << v) | neighbor_mask[cur]
                onward = 0
                for b2 in range(N):
                    nv = v ^ bit_masks[b2]
                    if not (new_forbidden & (1 << nv)):
                        onward += 1
                candidates.append((onward, b, v))

            if not candidates:
                return

            # Sort by onward moves (Warnsdorff: fewest onward first)
            candidates.sort(key=lambda x: x[0])

            # Try each candidate
            for onward, b, v in candidates:
                if time.time() > deadline:
                    return
                new_forbidden = forbidden_mask | (1 << v) | neighbor_mask[cur]
                path.append(v)
                _search(path, new_forbidden, depth + 1)
                path.pop()

        _search([start_vertex], (1 << start_vertex), 1)
        return best_path

    def _local_search(path, deadline, rng):
        """Local search: try to insert vertices into gaps or extend ends."""
        if not path or len(path) < 2:
            return path

        best = path[:]
        best_len = len(best)
        used = set(best)
        forbidden = _rebuild_forbidden(best)

        # Try extending both ends
        for _ in range(100):
            if time.time() > deadline:
                break

            # Try extending the end
            cur = best[-1]
            candidates = []
            for b in range(N):
                v = cur ^ bit_masks[b]
                if v in used:
                    continue
                if forbidden & (1 << v):
                    continue
                new_forbidden = forbidden | (1 << v) | neighbor_mask[cur]
                onward = 0
                for b2 in range(N):
                    nv = v ^ bit_masks[b2]
                    if not (new_forbidden & (1 << nv)):
                        onward += 1
                candidates.append((onward, v))

            if candidates:
                candidates.sort(key=lambda x: x[0])
                _, v = candidates[0]
                new_path = best + [v]
                if _valid_snake(new_path):
                    best = new_path
                    best_len += 1
                    used.add(v)
                    forbidden = _rebuild_forbidden(best)
                    continue

            # Try extending the beginning
            cur = best[0]
            candidates = []
            for b in range(N):
                v = cur ^ bit_masks[b]
                if v in used:
                    continue
                if forbidden & (1 << v):
                    continue
                new_forbidden = forbidden | (1 << v) | neighbor_mask[cur]
                onward = 0
                for b2 in range(N):
                    nv = v ^ bit_masks[b2]
                    if not (new_forbidden & (1 << nv)):
                        onward += 1
                candidates.append((onward, v))

            if candidates:
                candidates.sort(key=lambda x: x[0])
                _, v = candidates[0]
                new_path = [v] + best
                if _valid_snake(new_path):
                    best = new_path
                    best_len += 1
                    used.add(v)
                    forbidden = _rebuild_forbidden(best)
                    continue

            # Try inserting a vertex between two consecutive vertices
            # This is more complex - skip for now

        return best

    def _swap_improve(path, deadline, rng):
        """Try to improve the snake by swapping segments or trying different branches."""
        if not path or len(path) < 3:
            return path

        best = path[:]
        best_len = len(best)

        for _ in range(50):
            if time.time() > deadline:
                break

            # Try replacing a vertex with an alternative
            pos = rng.randint(1, len(best) - 1)
            prev_v = best[pos - 1]
            next_v = best[pos + 1] if pos + 1 < len(best) else None
            old_v = best[pos]

            # Find alternative vertices that could replace old_v
            candidates = []
            for b in range(N):
                v = prev_v ^ bit_masks[b]
                if v == old_v:
                    continue
                if v in set(best):
                    continue
                # Check if v can connect to prev_v and next_v
                if next_v is not None:
                    if bin(v ^ next_v).count('1') != 1:
                        continue
                # Check that v doesn't create distance-1 conflicts with non-adjacent vertices
                valid = True
                new_path = best[:pos] + [v] + best[pos+1:]
                used_set = set(new_path)
                if len(used_set) != len(new_path):
                    valid = False
                else:
                    for i in range(len(new_path)):
                        for j in range(i+2, len(new_path)):
                            if bin(new_path[i] ^ new_path[j]).count('1') == 1:
                                valid = False
                                break
                        if not valid:
                            break
                if valid:
                    candidates.append(v)

            if candidates:
                v = rng.choice(candidates)
                new_path = best[:pos] + [v] + best[pos+1:]
                if _valid_snake(new_path):
                    best = new_path

        return best

    # Phase 1: Multiple greedy extensions from diverse starts
    rng = random.Random(42)
    best_path = None
    best_len = 0

    starts = [0, 1, (1 << N) - 1, 0x155, 0x1AA, 0x0FF, 0x1F0, 0x123, 0x1AB, 0x246, 0x3AB]
    for _ in range(30):
        starts.append(rng.randint(0, (1 << N) - 1))

    for sv in starts:
        if time.time() > deadline - 40:
            break
        path = _greedy_extend([sv], min(deadline - 35, time.time() + 2.0), rng, use_random_tiebreak=True)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path

    # Phase 2: Backtracking search from promising starts
    if best_path:
        # Use the middle of the best path as a start for backtracking
        mid = len(best_path) // 2
        sv = best_path[mid]
        search_deadline = min(deadline - 20, time.time() + 5.0)
        path = _backtrack_search(sv, search_deadline, max_depth=120, rng=rng)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path

    # Additional backtracking from random starts
    for _ in range(10):
        if time.time() > deadline - 15:
            break
        sv = rng.randint(0, (1 << N) - 1)
        search_deadline = min(deadline - 10, time.time() + 3.0)
        path = _backtrack_search(sv, search_deadline, max_depth=110, rng=rng)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path

    # Phase 3: Local search refinement
    if best_path:
        refined = _local_search(best_path[:], deadline - 10, rng)
        if len(refined) > best_len:
            best_len = len(refined)
            best_path = refined

    # Phase 4: More greedy extensions with different tie-breaking
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        sv = rng.randint(0, (1 << N) - 1)
        path = _greedy_extend([sv], min(deadline - 3, time.time() + 1.5), rng, use_random_tiebreak=True)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path

    # Phase 5: Final backtracking attempts
    if time.time() < deadline - 3:
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            sv = rng.randint(0, (1 << N) - 1)
            search_deadline = min(deadline - 1, time.time() + 1.5)
            path = _backtrack_search(sv, search_deadline, max_depth=125, rng=rng)
            if len(path) > best_len:
                best_len = len(path)
                best_path = path

    # Final: ensure we have a valid path
    if best_path is None or len(best_path) < 2:
        best_path = [0, 1]
        best_len = 2

    # Final validation
    if not _valid_snake(best_path):
        # Fallback to a simple valid snake
        best_path = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511]
        # Verify this is valid
        if not _valid_snake(best_path):
            best_path = [0, 1]

    return best_path
