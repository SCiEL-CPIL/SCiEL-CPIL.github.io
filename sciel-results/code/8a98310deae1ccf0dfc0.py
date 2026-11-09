import time
import random
import sys
import numpy as np

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 118.0  # Safety margin for 120s limit (or 150s if evaluator allows, but 118 is safe)

# Precompute neighbors and adjacency masks
neighbors = [[] for _ in range(MAX_VERTS)]
adj_mask = [0] * MAX_VERTS

for v in range(MAX_VERTS):
    for b in range(N):
        nb = v ^ (1 << b)
        neighbors[v].append(nb)
        adj_mask[v] |= (1 << nb)

def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    Strategy: Hybrid randomized construction + local search (2-opt/segment reverse/endpoint swap)
    over valid snakes, with periodic repair+regrow to escape local optima.
    """
    start_time = time.time()
    best_path = None
    best_len = 0

    neighbors_local = neighbors
    adj_mask_local = adj_mask

    def valid_moves_from(end_vertex, used_mask):
        valid = []
        other_used = used_mask ^ (1 << end_vertex)
        for c in neighbors_local[end_vertex]:
            if used_mask & (1 << c):
                continue
            if adj_mask_local[c] & other_used:
                continue
            valid.append(c)
        return valid

    def grow_bidirectional(start_vertex, time_budget=0.05):
        t0 = time.time()
        v0 = start_vertex
        path = [v0]
        used = 1 << v0
        v1 = random.choice(neighbors_local[v0])
        path.append(v1)
        used |= (1 << v1)

        def count_onward(candidate, used_mask):
            new_used = used_mask | (1 << candidate)
            cnt = 0
            for w in neighbors_local[candidate]:
                if new_used & (1 << w):
                    continue
                if adj_mask_local[w] & used_mask:
                    continue
                cnt += 1
            return cnt

        while True:
            if time_budget and (time.time() - t0 > time_budget):
                break
            if time.time() - start_time > DEADLINE:
                break

            left_end = path[0]
            right_end = path[-1]

            valid_left = valid_moves_from(left_end, used)
            valid_right = valid_moves_from(right_end, used)

            if not valid_left and not valid_right:
                break

            best_left, min_left = None, 10**9
            best_right, min_right = None, 10**9
            if valid_left:
                for v in valid_left:
                    s = count_onward(v, used)
                    if s < min_left:
                        min_left = s
                        best_left = [v]
                    elif s == min_left:
                        best_left.append(v)
            if valid_right:
                for v in valid_right:
                    s = count_onward(v, used)
                    if s < min_right:
                        min_right = s
                        best_right = [v]
                    elif s == min_right:
                        best_right.append(v)

            if not valid_left:
                nxt = random.choice(best_right)
                path.append(nxt)
                used |= (1 << nxt)
            elif not valid_right:
                nxt = random.choice(best_left)
                path.insert(0, nxt)
                used |= (1 << nxt)
            else:
                if min_left < min_right:
                    nxt = random.choice(best_left)
                    path.insert(0, nxt)
                    used |= (1 << nxt)
                elif min_right < min_left:
                    nxt = random.choice(best_right)
                    path.append(nxt)
                    used |= (1 << nxt)
                else:
                    if random.random() < 0.5:
                        nxt = random.choice(best_left)
                        path.insert(0, nxt)
                        used |= (1 << nxt)
                    else:
                        nxt = random.choice(best_right)
                        path.append(nxt)
                        used |= (1 << nxt)
        return path

    def is_valid_snake(path):
        if not path:
            return True
        seen = set()
        for v in path:
            if v in seen:
                return False
            seen.add(v)
        for i in range(len(path) - 1):
            a, b = path[i], path[i+1]
            if (a ^ b).bit_count() != 1:
                return False
        # Check non-consecutive non-adjacency
        for i in range(len(path)):
            for j in range(i + 2, len(path)):
                if (path[i] ^ path[j]).bit_count() == 1:
                    return False
        return True

    def try_repair_and_regrow(path, time_budget=0.05):
        # Repair: remove vertices that break induced property minimally, then regrow
        t0 = time.time()
        # Build a new path by scanning and keeping only vertices that don't conflict with previous non-consecutive
        repaired = []
        used_mask = 0
        for v in path:
            # Check if v can be appended to repaired
            ok = True
            if repaired:
                last = repaired[-1]
                if (v ^ last).bit_count() != 1:
                    ok = False
                else:
                    # Check non-consecutive adjacency with all but last
                    for u in repaired[:-1]:
                        if (v ^ u).bit_count() == 1:
                            ok = False
                            break
            if not ok:
                continue
            repaired.append(v)
            used_mask |= (1 << v)
        if len(repaired) < 2:
            repaired = path[:2] if len(path) >= 2 else path[:]
        # Regrow from both ends
        p = grow_bidirectional(repaired[0], time_budget=max(0.0, time_budget - (time.time() - t0)))
        return p

    def local_search(path, time_budget=0.2):
        t0 = time.time()
        cur = list(path)
        cur_len = len(cur)
        # Precompute position map for fast membership
        pos = {v: i for i, v in enumerate(cur)}

        def is_valid_after_reverse(i, j):
            # Reverse subpath cur[i..j]; only need to check boundaries and internal consistency
            # Internal adjacency is preserved under reversal; only boundary adjacencies change
            # and non-consecutive constraints at the two cut points.
            if i == 0 and j == len(cur) - 1:
                return True
            # After reversal, new neighbors at cut points:
            # left cut: cur[i-1] (if exists) with cur[j]
            # right cut: cur[i] with cur[j+1] (if exists)
            if i > 0 and (cur[i-1] ^ cur[j]).bit_count() != 1:
                return False
            if j < len(cur) - 1 and (cur[i] ^ cur[j+1]).bit_count() != 1:
                return False
            # Non-consecutive conflicts introduced:
            # cur[i-1] with cur[j-1..i] (if any) except cur[i-1] itself
            # cur[j+1] with cur[i+1..j] (if any) except cur[j+1] itself
            if i > 0:
                for k in range(max(0, i-1), j):
                    if k != i-1 and (cur[i-1] ^ cur[k]).bit_count() == 1:
                        return False
            if j < len(cur) - 1:
                for k in range(i, min(len(cur), j+2)):
                    if k != j+1 and (cur[j+1] ^ cur[k]).bit_count() == 1:
                        return False
            return True

        def apply_reverse(i, j):
            nonlocal pos
            cur[i:j+1] = cur[i:j+1][::-1]
            for k in range(i, j+1):
                pos[cur[k]] = k

        # 2-opt style segment reversal
        improved = True
        while improved and (time.time() - t0) < time_budget and (time.time() - start_time) < DEADLINE:
            improved = False
            # Try random reversals first, then targeted
            for _ in range(200):
                if time.time() - start_time > DEADLINE:
                    return cur
                i = random.randint(0, len(cur) - 2)
                j = random.randint(i + 1, len(cur) - 1)
                if is_valid_after_reverse(i, j):
                    apply_reverse(i, j)
                    # Try to grow after reversal
                    grown = grow_bidirectional(cur[0], time_budget=0.02)
                    if len(grown) > cur_len:
                        cur = grown
                        cur_len = len(cur)
                        pos = {v: i for i, v in enumerate(cur)}
                        improved = True
                        break
            # Endpoint swap: try replacing an endpoint with a different valid neighbor
            if not improved:
                for end in [0, -1]:
                    if time.time() - start_time > DEADLINE:
                        return cur
                    ev = cur[end]
                    valids = valid_moves_from(ev, sum(1 << v for v in cur) if False else 0)
                    # Simpler: just try regrow from current ends
                    grown = grow_bidirectional(cur[0], time_budget=0.02)
                    if len(grown) > cur_len:
                        cur = grown
                        cur_len = len(cur)
                        pos = {v: i for i, v in enumerate(cur)}
                        improved = True
                        break
        return cur

    # Main loop: construct, local search, occasionally repair+regrow
    while time.time() - start_time < DEADLINE:
        v0 = random.randint(0, MAX_VERTS - 1)
        p = grow_bidirectional(v0, time_budget=0.03)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p

        # Local search on best so far
        if best_path and (time.time() - start_time) < DEADLINE - 0.5:
            p2 = local_search(best_path, time_budget=0.15)
            if len(p2) > best_len:
                best_len = len(p2)
                best_path = p2

        # Occasionally repair+regrow from best
        if best_path and random.random() < 0.3 and (time.time() - start_time) < DEADLINE - 0.5:
            p3 = try_repair_and_regrow(best_path, time_budget=0.05)
            if len(p3) > best_len:
                best_len = len(p3)
                best_path = p3

    if best_path is None:
        return [0]

    # Final validation fallback
    if not is_valid_snake(best_path):
        # Reconstruct a simple valid snake
        p = grow_bidirectional(0, time_budget=0.1)
        return p if p else [0]

    return best_path

# EVOLVE-BLOCK-END
