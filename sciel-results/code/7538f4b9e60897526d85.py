import time
import random

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 118.0

# Precompute neighbors and neighbor bitmasks
_NEIGHBORS = [[] for _ in range(MAX_VERTS)]
_NEIGHBOR_MASK = [0] * MAX_VERTS
for _v in range(MAX_VERTS):
    for _b in range(N):
        _nb = _v ^ (1 << _b)
        _NEIGHBORS[_v].append(_nb)
        _NEIGHBOR_MASK[_v] |= (1 << _nb)

# Also precompute: for each vertex, the set of vertices at distance 2 (to speed up checks)
# Actually, the validity check is: candidate's neighbors (other than the end we're extending from)
# must not be in the used set. So we just need _NEIGHBORS and a used bitmask.

def construct():
    start_time = time.time()
    best_path = [0, 1]
    best_len = 2

    def check_time():
        return (time.time() - start_time) > DEADLINE

    def is_valid_extend(path, used_mask, candidate, end_vertex):
        """Check if candidate can be added adjacent to end_vertex without creating
        a non-consecutive adjacency."""
        if used_mask & (1 << candidate):
            return False
        # candidate's neighbors (other than end_vertex) must not be in used
        for u in _NEIGHBORS[candidate]:
            if u == end_vertex:
                continue
            if used_mask & (1 << u):
                return False
        return True

    def count_future_options(candidate, end_vertex, used_mask):
        """Count how many valid moves would be available after adding candidate."""
        new_used = used_mask | (1 << candidate)
        count = 0
        for nb in _NEIGHBORS[candidate]:
            if nb == end_vertex:
                continue
            if new_used & (1 << nb):
                continue
            # Check nb is valid to add (its other neighbors not in new_used)
            valid = True
            for u in _NEIGHBORS[nb]:
                if u == candidate:
                    continue
                if new_used & (1 << u):
                    valid = False
                    break
            if valid:
                count += 1
        return count

    def grow_bidirectional(path):
        """Grow path bidirectionally using Warnsdorff-style heuristic."""
        used_mask = 0
        for v in path:
            used_mask |= (1 << v)

        while not check_time():
            if len(path) < 2:
                break

            left_end = path[0]
            right_end = path[-1]

            # Get valid moves for left end
            left_moves = []
            for v in _NEIGHBORS[left_end]:
                if is_valid_extend(path, used_mask, v, left_end):
                    left_moves.append(v)

            # Get valid moves for right end
            right_moves = []
            for v in _NEIGHBORS[right_end]:
                if is_valid_extend(path, used_mask, v, right_end):
                    right_moves.append(v)

            if not left_moves and not right_moves:
                break

            # Choose which end to extend
            if not left_moves:
                side = 'right'
            elif not right_moves:
                side = 'left'
            else:
                # Warnsdorff: extend the end with fewer options
                if len(left_moves) < len(right_moves):
                    side = 'left'
                elif len(right_moves) < len(left_moves):
                    side = 'right'
                else:
                    side = random.choice(['left', 'right'])

            if side == 'left':
                # Pick best candidate from left_moves using future options
                if len(left_moves) <= 3:
                    next_v = random.choice(left_moves)
                else:
                    # Score each candidate by future options (fewer is better)
                    best_cand = None
                    best_score = float('inf')
                    for v in left_moves:
                        score = count_future_options(v, left_end, used_mask)
                        if score < best_score:
                            best_score = score
                            best_cand = v
                        if score == 0:
                            break
                    next_v = best_cand if best_cand is not None else random.choice(left_moves)
                path.insert(0, next_v)
                used_mask |= (1 << next_v)
            else:
                if len(right_moves) <= 3:
                    next_v = random.choice(right_moves)
                else:
                    best_cand = None
                    best_score = float('inf')
                    for v in right_moves:
                        score = count_future_options(v, right_end, used_mask)
                        if score < best_score:
                            best_score = score
                            best_cand = v
                        if score == 0:
                            break
                    next_v = best_cand if best_cand is not None else random.choice(right_moves)
                path.append(next_v)
                used_mask |= (1 << next_v)

        return path

    def perturb_and_regrow(path):
        """Remove a random internal vertex and regrow from both ends."""
        if len(path) < 4:
            return path

        # Remove a random internal vertex (not at ends)
        idx = random.randint(1, len(path) - 2)
        removed = path[idx]

        # Split into two segments
        left_seg = path[:idx]
        right_seg = path[idx + 1:]

        if len(left_seg) < 2:
            return path
        if len(right_seg) < 2:
            return path

        # Regrow left segment bidirectionally
        left_grown = grow_bidirectional(list(left_seg))
        # Regrow right segment bidirectionally
        right_grown = grow_bidirectional(list(right_seg))

        # Try to merge: find if an end of left_grown is adjacent to an end of right_grown
        # and the merge is valid
        best_merge = None
        best_merge_len = 0

        for i in range(len(left_grown)):
            for j in range(len(right_grown)):
                lv = left_grown[i]
                rv = right_grown[j]
                if lv == rv:
                    continue
                # Check if lv and rv are adjacent
                if (lv ^ rv).bit_count() != 1:
                    continue
                # Check merge validity: lv's other neighbors not in right_grown (except rv)
                # and rv's other neighbors not in left_grown (except lv)
                right_set = set(right_grown)
                left_set = set(left_grown)

                valid = True
                for u in _NEIGHBORS[lv]:
                    if u == rv:
                        continue
                    if u in right_set:
                        valid = False
                        break
                if valid:
                    for u in _NEIGHBORS[rv]:
                        if u == lv:
                            continue
                        if u in left_set:
                            valid = False
                            break

                if valid:
                    merged = left_grown + right_grown
                    if len(merged) > best_merge_len:
                        best_merge_len = len(merged)
                        best_merge = merged

        if best_merge and len(best_merge) > len(path):
            return best_merge
        return path

    # Main loop
    attempts = 0
    while not check_time():
        attempts += 1

        # Strategy 1: Fresh greedy bidirectional growth
        if attempts % 3 == 1:
            v0 = random.randint(0, MAX_VERTS - 1)
            v1 = random.choice(_NEIGHBORS[v0])
            path = [v0, v1]
            path = grow_bidirectional(path)

        # Strategy 2: Start from a longer random seed and grow
        elif attempts % 3 == 2:
            # Build a short path greedily first
            v0 = random.randint(0, MAX_VERTS - 1)
            path = [v0]
            used_mask = 1 << v0
            for _ in range(15):
                current = path[-1]
                valid_next = []
                for v in _NEIGHBORS[current]:
                    if used_mask & (1 << v):
                        continue
                    valid = True
                    for u in _NEIGHBORS[v]:
                        if u == current:
                            continue
                        if used_mask & (1 << u):
                            valid = False
                            break
                    if valid:
                        valid_next.append(v)
                if not valid_next:
                    break
                # Pick with fewest future options
                if len(valid_next) <= 2:
                    next_v = random.choice(valid_next)
                else:
                    best_cand = None
                    best_score = float('inf')
                    for v in valid_next:
                        score = count_future_options(v, current, used_mask)
                        if score < best_score:
                            best_score = score
                            best_cand = v
                        if score == 0:
                            break
                    next_v = best_cand if best_cand is not None else random.choice(valid_next)
                path.append(next_v)
                used_mask |= (1 << next_v)
            if len(path) >= 2:
                path = grow_bidirectional(path)
            else:
                path = [0, 1]
                path = grow_bidirectional(path)

        # Strategy 3: Perturb best path found so far
        else:
            if best_len >= 4:
                path = list(best_path)
                path = perturb_and_regrow(path)
            else:
                v0 = random.randint(0, MAX_VERTS - 1)
                v1 = random.choice(_NEIGHBORS[v0])
                path = [v0, v1]
                path = grow_bidirectional(path)

        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)

    return best_path
