import time
import random
import math
import sys

N = 9
MAX_V = 1 << N


def _precompute_neighbors():
    masks = [1 << b for b in range(N)]
    neighbors = [tuple(v ^ m for m in masks) for v in range(MAX_V)]
    return neighbors


def _is_valid_snake(path):
    if not path:
        return True
    seen = set()
    for v in path:
        if v in seen:
            return False
        seen.add(v)
    for i in range(len(path) - 1):
        diff = path[i] ^ path[i + 1]
        if diff & (diff - 1) != 0:
            return False
    L = len(path)
    for i in range(L):
        vi = path[i]
        for b in range(N):
            nb = vi ^ (1 << b)
            if nb in seen:
                if i > 0 and path[i - 1] == nb:
                    continue
                if i < L - 1 and path[i + 1] == nb:
                    continue
                return False
    return True


def _compute_state(path, neighbors):
    used = set(path)
    block = [0] * MAX_V
    for v in path:
        for nb in neighbors[v]:
            if nb not in used:
                block[nb] += 1
    return used, block


def _try_extend_end(path, end, neighbors, block, used):
    if end == 'end':
        cur = path[-1]
        for nb in neighbors[cur]:
            if nb not in used and block[nb] == 1:
                path.append(nb)
                used.add(nb)
                for nn in neighbors[nb]:
                    if nn not in used:
                        block[nn] += 1
                return True
    else:
        cur = path[0]
        for nb in neighbors[cur]:
            if nb not in used and block[nb] == 1:
                path.insert(0, nb)
                used.add(nb)
                for nn in neighbors[nb]:
                    if nn not in used:
                        block[nn] += 1
                return True
    return False


def _full_extend(path, neighbors):
    if not path:
        return path
    used, block = _compute_state(path, neighbors)
    changed = True
    while changed:
        changed = False
        if _try_extend_end(path, 'end', neighbors, block, used):
            changed = True
        if _try_extend_end(path, 'start', neighbors, block, used):
            changed = True
    return path


def _greedy_build(start, neighbors):
    path = [start]
    used = {start}
    while True:
        cur = path[-1]
        best_nb = -1
        best_deg = 999
        for nb in neighbors[cur]:
            if nb in used:
                continue
            deg = 0
            for nn in neighbors[nb]:
                if nn not in used and nn != cur:
                    deg += 1
            if deg < best_deg:
                best_deg = deg
                best_nb = nb
        if best_nb < 0:
            break
        path.append(best_nb)
        used.add(best_nb)
    return path


def _rotate_end_copy(path, neighbors):
    if len(path) < 3:
        return None
    L = len(path)
    old_last = path[-1]
    prev = path[-2]
    path_set = set(path)
    for nb in neighbors[prev]:
        if nb == old_last or nb in path_set:
            continue
        if L >= 3:
            p3 = path[-3]
            diff = nb ^ p3
            if diff & (diff - 1) == 0:
                continue
        valid = True
        for i in range(L - 2):
            diff = nb ^ path[i]
            if diff & (diff - 1) == 0:
                valid = False
                break
        if not valid:
            continue
        new_path = path[:-1] + [nb]
        new_path = _full_extend(new_path, neighbors)
        if _is_valid_snake(new_path) and len(new_path) >= L:
            return new_path
    return None


def _random_walk_end(path, neighbors):
    if len(path) < 3:
        return None
    old_last = path[-1]
    prev = path[-2]
    path_set = set(path)
    candidates = [nb for nb in neighbors[prev] if nb not in path_set and nb != old_last]
    if not candidates:
        return None
    nb = random.choice(candidates)
    valid = True
    L = len(path)
    for i in range(L - 2):
        diff = nb ^ path[i]
        if diff & (diff - 1) == 0:
            valid = False
            break
    if not valid:
        return None
    new_path = path[:-1] + [nb]
    new_path = _full_extend(new_path, neighbors)
    if _is_valid_snake(new_path):
        return new_path
    return None


def construct():
    deadline = time.time() + 135.0
    neighbors = _precompute_neighbors()
    random.seed(12345)

    best_path = []
    best_len = 0

    # Phase 1: Greedy builds with random restarts
    start_candidates = list(range(MAX_V))
    random.shuffle(start_candidates)

    max_greedy = 25
    for i in range(min(max_greedy, len(start_candidates))):
        if time.time() > deadline - 45:
            break
        start = start_candidates[i]
        path = _greedy_build(start, neighbors)
        path = _full_extend(path, neighbors)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]

    # Phase 2: Simulated Annealing
    if best_path:
        current = best_path[:]
        current_len = len(current)

        T_start = 250.0
        T_end = 0.3
        max_iter = 200000
        iter_count = 0
        time_limit = deadline - 8.0

        while time.time() < time_limit and iter_count < max_iter:
            progress = min(1.0, iter_count / max_iter)
            T = T_start * (T_end / T_start) ** progress

            r = random.random()

            if r < 0.45:
                new_path = _rotate_end_copy(current, neighbors)
                if new_path is not None:
                    new_len = len(new_path)
                    delta = new_len - current_len
                    if delta >= 0 or (delta > -40 and random.random() < math.exp(delta / max(T, 0.1))):
                        current = new_path
                        current_len = new_len
                        if current_len > best_len:
                            best_len = current_len
                            best_path = current[:]
            elif r < 0.85:
                new_path = _random_walk_end(current, neighbors)
                if new_path is not None:
                    new_len = len(new_path)
                    delta = new_len - current_len
                    if delta >= 0 or (delta > -40 and random.random() < math.exp(delta / max(T, 0.1))):
                        current = new_path
                        current_len = new_len
                        if current_len > best_len:
                            best_len = current_len
                            best_path = current[:]
            else:
                # Extend from start
                used, block = _compute_state(current, neighbors)
                if _try_extend_end(current, 'start', neighbors, block, used):
                    current_len = len(current)
                    if current_len > best_len:
                        best_len = current_len
                        best_path = current[:]

            iter_count += 1

    if best_path and _is_valid_snake(best_path):
        return best_path
    if best_path:
        return [0]
    return [0]
