import time
import random
import sys
import numpy as np

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 115.0  # Safety margin for 150s limit

# Precompute neighbors
neighbors = [[] for _ in range(MAX_VERTS)]
for v in range(MAX_VERTS):
    for b in range(N):
        nb = v ^ (1 << b)
        neighbors[v].append(nb)

# Precompute adjacency masks for conflict checking
adj_mask = [0] * MAX_VERTS
for v in range(MAX_VERTS):
    m = 0
    for nb in neighbors[v]:
        m |= (1 << nb)
    adj_mask[v] = m


def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    Strategy: Multi-phase approach:
    1. Greedy construction with Warnsdorff's rule and bidirectional growth
    2. Local search with gap insertion and endpoint pivoting
    3. Systematic polish: sweep all gaps for insertions until no more possible
    4. Final exhaustive polish phase
    """
    start_time = time.time()
    best_path = None
    best_len = 0

    neighbors_local = neighbors
    adj_mask_local = adj_mask
    max_verts = MAX_VERTS

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def path_to_mask(path):
        mask = 0
        for v in path:
            mask |= (1 << v)
        return mask

    def get_valid_extensions(end_vertex, used_mask):
        """Get list of valid candidates to extend from end_vertex."""
        valid = []
        other_used = used_mask ^ (1 << end_vertex)
        for c in neighbors_local[end_vertex]:
            if used_mask & (1 << c):
                continue
            if adj_mask_local[c] & other_used:
                continue
            valid.append(c)
        return valid

    def count_onward_moves(candidate, used_mask):
        """Count valid moves from candidate if it were added."""
        new_used = used_mask | (1 << candidate)
        count = 0
        other = used_mask ^ (1 << candidate)
        for w in neighbors_local[candidate]:
            if new_used & (1 << w):
                continue
            if adj_mask_local[w] & other:
                continue
            count += 1
        return count

    def greedy_grow(path, used_mask, max_iters=200, random_factor=0.15):
        """Grow path greedily using Warnsdorff's rule with randomization."""
        for _ in range(max_iters):
            if time_left() < 2.0:
                break

            left_end = path[0]
            right_end = path[-1]

            valid_left = get_valid_extensions(left_end, used_mask)
            valid_right = get_valid_extensions(right_end, used_mask)

            if not valid_left and not valid_right:
                break

            # Choose which end to extend
            if not valid_left:
                side = 'right'
            elif not valid_right:
                side = 'left'
            else:
                # Score both ends by minimum onward moves
                min_left = min(count_onward_moves(v, used_mask) for v in valid_left)
                min_right = min(count_onward_moves(v, used_mask) for v in valid_right)

                if min_left < min_right:
                    side = 'left'
                elif min_right < min_left:
                    side = 'right'
                else:
                    side = 'left' if random.random() < 0.5 else 'right'

            if side == 'left':
                valid = valid_left
            else:
                valid = valid_right

            if not valid:
                break

            # Score candidates by onward moves
            scored = [(count_onward_moves(v, used_mask), v) for v in valid]
            scored.sort(key=lambda x: x[0])

            # Pick with some randomization
            if random.random() < random_factor:
                top_half = max(1, len(scored) // 2)
                idx = random.randint(0, top_half - 1)
            else:
                idx = 0

            next_v = scored[idx][1]
            new_used = used_mask | (1 << next_v)

            if side == 'left':
                path.insert(0, next_v)
            else:
                path.append(next_v)

            used_mask = new_used

        return path, used_mask, len(path)

    def try_insert_at_gap(path, used_mask, gap_idx, candidates):
        """
        Try to insert one of the candidates at gap_idx (between path[gap_idx] and path[gap_idx+1]).
        Returns (new_path, new_used_mask, success).
        """
        # gap_idx is the index in path; we insert between path[gap_idx] and path[gap_idx+1]
        # Actually, let's define: we want to insert a vertex v such that
        # v is adjacent to path[gap_idx] and path[gap_idx+1]... no wait.
        # For a path, inserting at a "gap" means: the new vertex must be adjacent to exactly one
        # of the two path vertices it goes between, OR we're inserting at an endpoint.
        # 
        # Actually, for induced paths, we can only insert at endpoints or at gaps where
        # the new vertex is adjacent to both neighbors of the gap and not adjacent to anything else.
        # But that's very restrictive. More commonly, we extend at endpoints.
        #
        # Let me reconsider: "gap insertion" typically means:
        # - At position i (0 <= i <= len(path)), we can insert vertex v if:
        #   - If i == 0: v adjacent to path[0], v not used, v not adjacent to path[1:]
        #   - If i == len(path): v adjacent to path[-1], v not used, v not adjacent to path[:-1]
        #   - If 0 < i < len(path): v adjacent to path[i-1] AND path[i], v not used,
        #     v not adjacent to any other path vertex
        #
        # The middle case is very rare. Let's focus on endpoint extensions and also
        # try "pivot" moves (remove an endpoint, add a different one).

        return path, used_mask, False

    def local_search(path, used_mask, max_iterations=500):
        """
        Local search: try to extend at endpoints, pivot endpoints, and insert at gaps.
        """
        for it in range(max_iterations):
            if time_left() < 2.0:
                break

            improved = False

            # Try extending left end
            valid_left = get_valid_extensions(path[0], used_mask)
            if valid_left:
                # Pick best by onward moves
                scored = [(count_onward_moves(v, used_mask), v) for v in valid_left]
                scored.sort(key=lambda x: x[0])
                next_v = scored[0][1]
                new_used = used_mask | (1 << next_v)
                path.insert(0, next_v)
                used_mask = new_used
                improved = True
                continue

            # Try extending right end
            valid_right = get_valid_extensions(path[-1], used_mask)
            if valid_right:
                scored = [(count_onward_moves(v, used_mask), v) for v in valid_right]
                scored.sort(key=lambda x: x[0])
                next_v = scored[0][1]
                new_used = used_mask | (1 << next_v)
                path.append(next_v)
                used_mask = new_used
                improved = True
                continue

            # Try pivoting left end
            if len(path) >= 2:
                old_first = path[0]
                second = path[1]
                body = path[1:]
                body_mask = 0
                for v in body:
                    body_mask |= (1 << v)

                valid_pivot = []
                for c in neighbors_local[second]:
                    if c == old_first:
                        continue
                    if body_mask & (1 << c):
                        continue
                    if adj_mask_local[c] & (body_mask ^ (1 << second)):
                        continue
                    valid_pivot.append(c)

                if valid_pivot:
                    # Try each pivot option
                    random.shuffle(valid_pivot)
                    for pv in valid_pivot:
                        new_path = [pv] + body
                        new_mask = body_mask | (1 << pv)
                        # Check if this leads to more extensions
                        vl = get_valid_extensions(pv, new_mask)
                        vr = get_valid_extensions(new_path[-1], new_mask)
                        if vl or vr:
                            path = new_path
                            used_mask = new_mask
                            improved = True
                            break

            # Try pivoting right end
            if not improved and len(path) >= 2:
                old_last = path[-1]
                second_last = path[-2]
                body = path[:-1]
                body_mask = 0
                for v in body:
                    body_mask |= (1 << v)

                valid_pivot = []
                for c in neighbors_local[second_last]:
                    if c == old_last:
                        continue
                    if body_mask & (1 << c):
                        continue
                    if adj_mask_local[c] & (body_mask ^ (1 << second_last)):
                        continue
                    valid_pivot.append(c)

                if valid_pivot:
                    random.shuffle(valid_pivot)
                    for pv in valid_pivot:
                        new_path = body + [pv]
                        new_mask = body_mask | (1 << pv)
                        vl = get_valid_extensions(new_path[0], new_mask)
                        vr = get_valid_extensions(pv, new_mask)
                        if vl or vr:
                            path = new_path
                            used_mask = new_mask
                            improved = True
                            break

            # Try inserting at internal gaps (rare but possible)
            if not improved and len(path) >= 3:
                for gap in range(1, len(path) - 1):
                    left_v = path[gap - 1]
                    right_v = path[gap]
                    # Find common neighbors of left_v and right_v
                    common = set(neighbors_local[left_v]) & set(neighbors_local[right_v])
                    for c in common:
                        if used_mask & (1 << c):
                            continue
                        # Check c is not adjacent to any other path vertex
                        other_mask = used_mask ^ (1 << left_v) ^ (1 << right_v)
                        if adj_mask_local[c] & other_mask:
                            continue
                        # Valid insertion
                        new_path = path[:gap] + [c] + path[gap:]
                        new_mask = used_mask | (1 << c)
                        path = new_path
                        used_mask = new_mask
                        improved = True
                        break
                    if improved:
                        break

            if not improved:
                # No more simple improvements; try random perturbations
                if random.random() < 0.3 and len(path) >= 4:
                    # Remove a random internal segment and try to regrow
                    seg_len = random.randint(1, min(3, len(path) - 2))
                    start_idx = random.randint(1, len(path) - seg_len - 1)
                    removed = path[start_idx:start_idx + seg_len]
                    body = path[:start_idx] + path[start_idx + seg_len:]
                    body_mask = 0
                    for v in body:
                        body_mask |= (1 << v)

                    # Try to regrow from both ends
                    new_path, new_mask, new_len = greedy_grow(body, body_mask, max_iters=30, random_factor=0.3)
                    if len(new_path) > len(path):
                        path = new_path
                        used_mask = new_mask
                        improved = True

                if not improved and random.random() < 0.1:
                    break

        return path, used_mask, len(path)

    def polish(path, used_mask):
        """
        Exhaustive polish: repeatedly sweep all endpoints and gaps for insertions
        until no more improvements are possible.
        """
        changed = True
        while changed and time_left() > 2.0:
            changed = False

            # Try left extension
            valid_left = get_valid_extensions(path[0], used_mask)
            if valid_left:
                scored = [(count_onward_moves(v, used_mask), v) for v in valid_left]
                scored.sort(key=lambda x: x[0])
                next_v = scored[0][1]
                used_mask |= (1 << next_v)
                path.insert(0, next_v)
                changed = True
                continue

            # Try right extension
            valid_right = get_valid_extensions(path[-1], used_mask)
            if valid_right:
                scored = [(count_onward_moves(v, used_mask), v) for v in valid_right]
                scored.sort(key=lambda x: x[0])
                next_v = scored[0][1]
                used_mask |= (1 << next_v)
                path.append(next_v)
                changed = True
                continue

            # Try all internal gap insertions
            for gap in range(1, len(path) - 1):
                left_v = path[gap - 1]
                right_v = path[gap]
                common = set(neighbors_local[left_v]) & set(neighbors_local[right_v])
                for c in common:
                    if used_mask & (1 << c):
                        continue
                    other_mask = used_mask ^ (1 << left_v) ^ (1 << right_v)
                    if adj_mask_local[c] & other_mask:
                        continue
                    # Valid insertion
                    used_mask |= (1 << c)
                    path = path[:gap] + [c] + path[gap:]
                    changed = True
                    break
                if changed:
                    break

            # Try pivoting endpoints
            if not changed and len(path) >= 2:
                # Left pivot
                old_first = path[0]
                second = path[1]
                body = path[1:]
                body_mask = 0
                for v in body:
                    body_mask |= (1 << v)

                for c in neighbors_local[second]:
                    if c == old_first:
                        continue
                    if body_mask & (1 << c):
                        continue
                    if adj_mask_local[c] & (body_mask ^ (1 << second)):
                        continue
                    # Check if this pivot allows further extension
                    new_mask = body_mask | (1 << c)
                    vl = get_valid_extensions(c, new_mask)
                    if vl:
                        path = [c] + body
                        used_mask = new_mask
                        changed = True
                        break

                # Right pivot
                if not changed:
                    old_last = path[-1]
                    second_last = path[-2]
                    body = path[:-1]
                    body_mask = 0
                    for v in body:
                        body_mask |= (1 << v)

                    for c in neighbors_local[second_last]:
                        if c == old_last:
                            continue
                        if body_mask & (1 << c):
                            continue
                        if adj_mask_local[c] & (body_mask ^ (1 << second_last)):
                            continue
                        new_mask = body_mask | (1 << c)
                        vr = get_valid_extensions(c, new_mask)
                        if vr:
                            path = body + [c]
                            used_mask = new_mask
                            changed = True
                            break

        return path, used_mask, len(path)

    # Main multi-start loop
    num_starts = 0

    while time_left() > 3.0:
        num_starts += 1

        # Initialize with a random starting edge
        v0 = random.randint(0, max_verts - 1)
        path = [v0]
        used = 1 << v0

        v1 = random.choice(neighbors_local[v0])
        path.append(v1)
        used |= (1 << v1)

        # Phase 1: Greedy growth
        path, used, cur_len = greedy_grow(path, used, max_iters=100, random_factor=0.1)

        # Phase 2: Local search
        path, used, cur_len = local_search(path, used, max_iterations=300)

        # Phase 3: Polish
        path, used, cur_len = polish(path, used)

        if cur_len > best_len:
            best_len = cur_len
            best_path = path.copy()

        # Periodically try to extend the best path
        if num_starts % 3 == 0 and best_path is not None and time_left() > 5.0:
            bp_copy = best_path.copy()
            bm = path_to_mask(bp_copy)
            bp_copy, bm, bl = polish(bp_copy, bm)
            if bl > best_len:
                best_len = bl
                best_path = bp_copy.copy()

    if best_path is None:
        return [0, 1]

    return best_path
