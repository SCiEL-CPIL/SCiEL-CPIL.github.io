import time
import random

N = 9
MAX_TIME = 115.0  # Safety margin before 120s/150s kill


def _neighbors(v):
    return [v ^ (1 << b) for b in range(N)]


def _build_forbidden(path):
    """
    forbidden[v] = True if v is at distance 1 from any vertex in path except the last one.
    For a candidate extension u from current=path[-1], we need:
      - u not in used
      - u not in forbidden (i.e., dist(u, p) >= 2 for all p in path[:-1])
    """
    forbidden = set()
    # All vertices at distance 1 from any path vertex, but we will allow the current vertex's neighbors.
    # Simpler: compute all distance-1 neighbors of all path vertices, then remove those that are valid wrt current.
    # But we can maintain incrementally in greedy. For initial build, do O(L*N).
    for p in path:
        for nb in _neighbors(p):
            forbidden.add(nb)
    # The last vertex's neighbors are allowed (they are at distance 1 from current, which is OK).
    cur = path[-1]
    for nb in _neighbors(cur):
        forbidden.discard(nb)
    return forbidden


def _greedy_construct(seed_offset, max_steps=1000000):
    random.seed(seed_offset)
    path = [0]
    used = {0}
    current = 0
    forbidden = _build_forbidden(path)

    steps = 0
    while steps < max_steps:
        steps += 1
        valid_neighbors = []
        for b in range(N):
            v = current ^ (1 << b)
            if v in used or v in forbidden:
                continue
            valid_neighbors.append(v)

        if not valid_neighbors:
            break

        # Score by how many options it leaves (fast incremental update)
        best_v = None
        best_score = -1
        best_random = random.random()
        for v in valid_neighbors:
            # Compute number of valid neighbors from v after moving to v
            # New forbidden set:
            #   add neighbors of v (since v becomes non-current after next move)
            #   remove neighbors of current (since current becomes non-current? Actually current becomes part of path[:-1])
            # Let's just count by checking against current forbidden + used + {v}
            cnt = 0
            for b2 in range(N):
                u = v ^ (1 << b2)
                if u in used or u == current:
                    continue
                # u must not be in forbidden (which is dist1 from path[:-1])
                if u in forbidden:
                    continue
                cnt += 1
            score = cnt
            r = random.random()
            if score > best_score or (score == best_score and r > best_random):
                best_score = score
                best_v = v
                best_random = r

        if best_v is None:
            break

        # Update state
        path.append(best_v)
        used.add(best_v)
        # Update forbidden:
        # Remove neighbors of current (they were allowed because current was last; now current is non-last)
        for nb in _neighbors(current):
            forbidden.discard(nb)
        # Add neighbors of best_v (best_v is now last; its neighbors are allowed)
        # But we also need to add neighbors of best_v to forbidden? No:
        # forbidden = vertices at distance 1 from path[:-1].
        # After moving to best_v, path[:-1] includes current and all older.
        # So we need to add neighbors of current to forbidden (since current is now non-last).
        # And remove neighbors of best_v from forbidden (since best_v is now last).
        # Wait, I did the opposite above. Let's fix.
        # Actually:
        # Before move: forbidden = dist1(path[:-1]) where path[:-1] = path without current.
        # After move: forbidden_new = dist1(path) where path = old_path + [best_v], so path[:-1] = old_path.
        # So forbidden_new = dist1(old_path) = forbidden_old ∪ dist1(current) \ dist1(best_v)?
        # No: forbidden_old = dist1(old_path without current) = dist1(path[:-2]).
        # forbidden_new = dist1(path[:-1]) = dist1(path[:-2] ∪ {current}) = forbidden_old ∪ dist1(current).
        # And we should NOT include dist1(best_v) because best_v is the new current.
        # So:
        #   forbidden = forbidden | neighbors(current)
        #   (neighbors(best_v) are not in forbidden; they are allowed)
        for nb in _neighbors(current):
            forbidden.add(nb)
        # neighbors(best_v) should be allowed, so remove them from forbidden if present
        for nb in _neighbors(best_v):
            forbidden.discard(nb)

        current = best_v

    return path


def _try_extend(path, seed_offset, max_depth=60, max_nodes=200000):
    """
    Try to extend path by depth-limited randomized DFS.
    Returns a longer path if successful, else None.
    """
    random.seed(seed_offset)
    start_time = time.time()
    path = list(path)
    used = set(path)
    current = path[-1]
    forbidden = _build_forbidden(path)

    best_len = len(path)
    best_path_copy = list(path)

    def dfs(depth, nodes_left):
        nonlocal best_len, best_path_copy
        if nodes_left <= 0 or depth > max_depth:
            return
        if time.time() - start_time > MAX_TIME * 0.8:
            return

        # Update best if current path is longer
        if len(path) > best_len:
            best_len = len(path)
            best_path_copy = list(path)

        # Find valid neighbors
        valid_neighbors = []
        for b in range(N):
            v = current ^ (1 << b)
            if v in used or v in forbidden:
                continue
            valid_neighbors.append(v)

        if not valid_neighbors:
            return

        # Shuffle for randomness
        random.shuffle(valid_neighbors)

        for v in valid_neighbors:
            # Update state
            path.append(v)
            used.add(v)
            
            # Update forbidden set in-place
            # Current becomes 'past', so its neighbors are now forbidden
            for nb in _neighbors(current):
                forbidden.add(nb)
            # v becomes 'current', so its neighbors are now allowed (not forbidden)
            for nb in _neighbors(v):
                forbidden.discard(nb)

            dfs(depth + 1, nodes_left - 1)

            # Revert state
            # v is no longer current, so its neighbors are forbidden again
            for nb in _neighbors(v):
                forbidden.add(nb)
            # current is no longer 'past' in this branch's context, so its neighbors are allowed again
            for nb in _neighbors(current):
                forbidden.discard(nb)
            
            path.pop()
            used.discard(v)

            if nodes_left <= 0 or time.time() - start_time > MAX_TIME * 0.8:
                return

    dfs(0, max_nodes)
    return best_path_copy if best_len > len(path) else None


def _local_swap_improve(path, seed_offset, max_swaps=50, start_time=0):
    """
    Try 2-opt style swaps near the tail to unblock extension.
    Returns improved path or None.
    """
    random.seed(seed_offset)
    path = list(path)
    if len(path) < 4:
        return None

    # Try swapping two vertices near the tail
    for _ in range(max_swaps):
        if time.time() - start_time > MAX_TIME * 0.9:
            break
        i = random.randint(1, len(path) - 3)
        j = random.randint(i + 1, len(path) - 2)
        if j - i == 1:
            continue

        # Swap path[i] and path[j]
        path[i], path[j] = path[j], path[i]

        # Validate the path
        valid = True
        for k in range(len(path) - 1):
            if (path[k] ^ path[k + 1]).bit_count() != 1:
                valid = False
                break
        if valid:
            for a in range(len(path)):
                for b in range(a + 2, len(path)):
                    if (path[a] ^ path[b]).bit_count() == 1:
                        valid = False
                        break
                if not valid:
                    break

        if valid:
            # Try to extend
            extended = _try_extend(path, seed_offset + 1000, max_depth=30, max_nodes=50000)
            if extended and len(extended) > len(path):
                return extended
        else:
            # Revert
            path[i], path[j] = path[j], path[i]

    return None


def _is_valid_snake(path):
    if len(path) < 2:
        return True
    used = set(path)
    if len(used) != len(path):
        return False
    for i in range(len(path) - 1):
        if (path[i] ^ path[i+1]).bit_count() != 1:
            return False
    for i in range(len(path)):
        for j in range(i + 2, len(path)):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True

def construct():
    start_time = time.time()
    best_path = []

    # Build several initial snakes with different seeds
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for s in seeds:
        if time.time() - start_time > MAX_TIME * 0.5:
            break
        p = _greedy_construct(s)
        if len(p) > len(best_path):
            best_path = p

    # Try to extend the best path
    for s in range(20):
        if time.time() - start_time > MAX_TIME * 0.8:
            break
        extended = _try_extend(best_path, s + 100, max_depth=40, max_nodes=100000)
        if extended and len(extended) > len(best_path):
            best_path = extended

    # Try local swaps to unblock
    for s in range(10):
        if time.time() - start_time > MAX_TIME * 0.95:
            break
        improved = _local_swap_improve(best_path, s + 200, max_swaps=20)
        if improved and len(improved) > len(best_path):
            best_path = improved

    # Final validation
    if not _is_valid_snake(best_path):
        # Fallback to a simple valid snake if best_path is invalid
        # A trivial snake is just [0, 1] (diff 1 bit)
        # Or find the longest valid prefix
        valid_prefix = []
        for v in best_path:
            valid_prefix.append(v)
            if not _is_valid_snake(valid_prefix):
                valid_prefix.pop()
                break
        best_path = valid_prefix
        if not _is_valid_snake(best_path):
            best_path = [0, 1]

    return best_path
