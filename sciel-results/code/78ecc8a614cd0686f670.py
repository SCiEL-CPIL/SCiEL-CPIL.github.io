import time
import random
import sys
import numpy as np

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 118.0

def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    Strategy: Bidirectional growth with Warnsdorff heuristics and incremental local repair.
    Uses NumPy and bytearray for state optimization.
    """
    start_time = time.time()
    best_path = None
    best_len = 0

    # Precompute neighbors
    neighbors = [[] for _ in range(MAX_VERTS)]
    for v in range(MAX_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))

    def count_onward_moves(u, used, adj_count):
        """Count valid onward moves if we were to add u to the path."""
        count = 0
        for w in neighbors[u]:
            if not used[w] and adj_count[w] == 0:
                count += 1
        return count

    def bidirectional_grow(v_start, v_end_init, time_limit):
        """
        Grow a snake bidirectionally from v_start and v_end_init.
        Returns the longest valid snake found.
        """
        nonlocal best_path, best_len

        path = [v_start, v_end_init]
        used = bytearray(MAX_VERTS)
        used[v_start] = 1
        used[v_end_init] = 1
        adj_count = np.zeros(MAX_VERTS, dtype=np.uint8)

        for w in neighbors[v_start]:
            adj_count[w] += 1
        for w in neighbors[v_end_init]:
            adj_count[w] += 1

        # If the two starting vertices are adjacent, we have a valid edge
        if (v_start ^ v_end_init).bit_count() != 1:
            return None

        local_best = path[:]

        def update_best(p):
            nonlocal best_path, best_len, local_best
            if len(p) > len(local_best):
                local_best = p[:]
            if len(p) > best_len:
                best_len = len(p)
                best_path = p[:]

        update_best(path)

        # Grow bidirectionally
        while time.time() - start_time < time_limit:
            # Try to extend from the left end
            v_left = path[0]
            valid_left = []
            for u in neighbors[v_left]:
                if not used[u] and adj_count[u] == 1:
                    # u is adjacent only to v_left in current path
                    cnt = count_onward_moves(u, used, adj_count)
                    valid_left.append((cnt, u))

            # Try to extend from the right end
            v_right = path[-1]
            valid_right = []
            for u in neighbors[v_right]:
                if not used[u] and adj_count[u] == 1:
                    cnt = count_onward_moves(u, used, adj_count)
                    valid_right.append((cnt, u))

            if not valid_left and not valid_right:
                break

            # Choose which end to extend based on heuristic
            # Prefer the end with fewer options (Warnsdorff)
            if valid_left and (not valid_right or len(valid_left) <= len(valid_right)):
                # Extend left
                valid_left.sort()
                best_score = valid_left[0][0]
                candidates = [x for x in valid_left if x[0] <= best_score + 1]
                _, next_v = random.choice(candidates)
                path.insert(0, next_v)
                used[next_v] = 1
                for w in neighbors[next_v]:
                    adj_count[w] += 1
            else:
                # Extend right
                valid_right.sort()
                best_score = valid_right[0][0]
                candidates = [x for x in valid_right if x[0] <= best_score + 1]
                _, next_v = random.choice(candidates)
                path.append(next_v)
                used[next_v] = 1
                for w in neighbors[next_v]:
                    adj_count[w] += 1

            update_best(path)

        return local_best

    def local_repair(path, time_limit):
        """
        Try to improve a snake by removing a few vertices from the end and regrowing.
        Uses incremental state updates.
        """
        nonlocal best_path, best_len

        if not path or len(path) < 2:
            return path

        current = path[:]
        
        # Initialize state for current path
        used = bytearray(MAX_VERTS)
        adj_count = np.zeros(MAX_VERTS, dtype=np.uint8)
        for v in current:
            used[v] = 1
            for w in neighbors[v]:
                adj_count[w] += 1

        # Try removing last k vertices and regrowing
        for k in [1, 2, 3, 4, 5]:
            if time.time() - start_time > time_limit:
                break
            if len(current) <= k:
                break

            # Save state snapshot (copying bytearray and numpy array is fast)
            used_snapshot = used[:]
            adj_count_snapshot = adj_count.copy()
            
            orig_len = len(current)
            orig_path = current[:]

            # Remove last k vertices
            for _ in range(k):
                v_rem = current.pop()
                used[v_rem] = 0
                for w in neighbors[v_rem]:
                    adj_count[w] -= 1
            
            truncated_len = len(current)
            if truncated_len < 2:
                # Restore
                current = orig_path
                used = used_snapshot
                adj_count = adj_count_snapshot
                continue

            # Regrow from the truncated end
            new_path = current[:]
            
            while True:
                if time.time() - start_time > time_limit:
                    break
                v_end = new_path[-1]
                valid = []
                for u in neighbors[v_end]:
                    if not used[u] and adj_count[u] == 1:
                        cnt = count_onward_moves(u, used, adj_count)
                        valid.append((cnt, u))
                if not valid:
                    break
                valid.sort()
                best_score = valid[0][0]
                candidates = [x for x in valid if x[0] <= best_score + 1]
                _, next_v = random.choice(candidates)
                new_path.append(next_v)
                used[next_v] = 1
                for w in neighbors[next_v]:
                    adj_count[w] += 1

            if len(new_path) > len(current):
                current = new_path
                if best_path is None or len(current) > best_len:
                    best_len = len(current)
                    best_path = current[:]
            else:
                # Restore original state
                current = orig_path
                used = used_snapshot
                adj_count = adj_count_snapshot

        return current

    # Main: multi-start bidirectional growth
    # Phase 1: Quick greedy to get a baseline
    def quick_greedy(v_start):
        path = [v_start]
        used = bytearray(MAX_VERTS)
        used[v_start] = 1
        adj_count = np.zeros(MAX_VERTS, dtype=np.uint8)
        for w in neighbors[v_start]:
            adj_count[w] += 1

        while True:
            v_end = path[-1]
            valid = []
            for u in neighbors[v_end]:
                if not used[u] and adj_count[u] == 1:
                    cnt = count_onward_moves(u, used, adj_count)
                    valid.append((cnt, u))
            if not valid:
                break
            valid.sort()
            best_score = valid[0][0]
            candidates = [x for x in valid if x[0] <= best_score + 1]
            _, next_v = random.choice(candidates)
            path.append(next_v)
            used[next_v] = 1
            for w in neighbors[next_v]:
                adj_count[w] += 1
        return path

    # Phase 1: Get initial baselines
    for _ in range(10):
        v0 = random.randint(0, MAX_VERTS - 1)
        p = quick_greedy(v0)
        if best_path is None or len(p) > best_len:
            best_len = len(p)
            best_path = p[:]

    # Phase 2: Bidirectional growth with multiple starts
    attempts = 0
    while time.time() - start_time < DEADLINE - 5:
        attempts += 1
        # Choose two starting vertices that are adjacent
        v0 = random.randint(0, MAX_VERTS - 1)
        v1 = random.choice(neighbors[v0])

        # Give more time to early attempts
        remaining = DEADLINE - (time.time() - start_time)
        per_attempt = max(0.1, min(2.0, remaining / max(1, 50 - attempts * 0.5)))

        p = bidirectional_grow(v0, v1, time.time() - start_time + per_attempt)
        if p and len(p) > best_len:
            best_len = len(p)
            best_path = p[:]

        # Phase 3: Local repair on the best found so far
        if best_path and time.time() - start_time < DEADLINE - 10:
            repaired = local_repair(best_path[:], DEADLINE - 5)
            if repaired and len(repaired) > best_len:
                best_len = len(repaired)
                best_path = repaired[:]

    if best_path is None:
        return [0]
    return best_path
