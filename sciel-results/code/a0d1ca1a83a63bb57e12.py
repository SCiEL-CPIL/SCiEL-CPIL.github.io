"""
Improved Snake-in-the-box algorithm for 9-dimensional hypercube.

Key structural improvements over the seed:
1. Use NumPy boolean arrays instead of Python sets for fast blocked-vertex operations
2. Implement proper segment-replacement local search (not just single-vertex replacement)
3. Use time-bucketed restarts to explore multiple basins
4. Track both ends of the path for extension opportunities
5. Precompute bit masks for all vertices for O(1) bit-count checks
"""
import time
import random
import numpy as np

N = 9
TIME_LIMIT = 115.0  # Safety margin before 120s/150s kill


# Precompute bit-count lookup for all pairs up to 2^9
# We can use a precomputed table for bit_count since N=9 is small
BIT_COUNTS = np.zeros((1 << N, 1 << N), dtype=np.uint8)


def precompute_bit_counts():
    """Precompute Hamming distance table for all pairs of vertices in Q_9."""
    global BIT_COUNTS
    vertices = list(range(1 << N))
    # Use vectorized operations where possible
    for i, v in enumerate(vertices):
        # Compute v XOR all vertices, then bit count
        xor_vals = np.array([v ^ x for x in vertices], dtype=np.int32)
        # bit_count is available in Python 3.10+, but for numpy int arrays we need a loop or alternative
        # Actually, let's just compute it directly - for 512x512 = 262144 entries, it's fast enough
        for j, x in enumerate(vertices):
            BIT_COUNTS[i, j] = bin(x ^ v).count('1')


def get_greedy_seed():
    """
    Generate a greedy snake. Try to extend from both ends.
    Returns a list of vertex integers.
    """
    path = [0]
    used = np.zeros(1 << N, dtype=bool)
    used[0] = True
    # Track blocked: vertices adjacent to any used vertex
    blocked = np.zeros(1 << N, dtype=bool)
    for b in range(N):
        blocked[0 ^ (1 << b)] = True

    for _ in range(1000):
        # Try extending from the end
        cur = path[-1]
        extended = False
        for b in range(N):
            v = cur ^ (1 << b)
            if not used[v] and not blocked[v]:
                path.append(v)
                used[v] = True
                blocked[v] = True
                for bb in range(N):
                    blocked[v ^ (1 << bb)] = True
                extended = True
                break

        if extended:
            continue

        # Try extending from the start
        cur = path[0]
        for b in range(N):
            v = cur ^ (1 << b)
            if not used[v] and not blocked[v]:
                path.insert(0, v)
                used[v] = True
                blocked[v] = True
                for bb in range(N):
                    blocked[v ^ (1 << bb)] = True
                extended = True
                break

        if not extended:
            break

    return path


def is_valid_snake(path, used, blocked):
    """
    Verify that path is a valid snake.
    - Consecutive vertices differ by exactly 1 bit
    - Non-consecutive vertices differ by at least 2 bits
    """
    if len(path) < 2:
        return True

    # Check connectivity: consecutive differ by 1 bit
    for i in range(len(path) - 1):
        if BIT_COUNTS[path[i], path[i + 1]] != 1:
            return False

    # Check non-consecutive: differ by at least 2 bits
    for i in range(len(path)):
        for j in range(i + 2, len(path)):
            if BIT_COUNTS[path[i], path[j]] == 1:
                return False

    return True


def rebuild_used_and_blocked(path):
    """Rebuild used and blocked arrays from a path."""
    used = np.zeros(1 << N, dtype=bool)
    for v in path:
        used[v] = True

    blocked = np.zeros(1 << N, dtype=bool)
    for v in path:
        for b in range(N):
            blocked[v ^ (1 << b)] = True

    return used, blocked


def try_extend(path, end, used, blocked):
    """
    Try to extend the path at the given end (0 for start, 1 for end).
    Returns the new path and updated used/blocked if successful, else None.
    """
    if end == 1:
        cur = path[-1]
    else:
        cur = path[0]

    for b in range(N):
        v = cur ^ (1 << b)
        if used[v] or blocked[v]:
            continue

        # v is available; check it's not adjacent to any non-consecutive vertex
        # Actually, blocked already captures adjacency to all path vertices.
        # But blocked includes adjacency to consecutive neighbors too, which is fine
        # because we want v to NOT be adjacent to ANY path vertex except the one we're extending from.
        # Wait: the induced path condition says non-consecutive vertices must differ by >= 2 bits.
        # Consecutive vertices differ by exactly 1 bit (they're neighbors in the path).
        # So when extending, v must differ by 1 bit from the end vertex (guaranteed by construction),
        # and must differ by >= 2 bits from all other path vertices.
        # `blocked` marks all vertices adjacent to any path vertex.
        # If v is blocked, it means v is adjacent to SOME path vertex.
        # But v IS adjacent to the end vertex (that's how we got it), so we need to check
        # if v is adjacent to any OTHER path vertex.
        #
        # Actually, the `blocked` array marks neighbors of all path vertices.
        # The end vertex's neighbors include v. But we need v to not be a neighbor of any
        # non-end vertex.
        #
        # A cleaner check: v must not be in used, and for all p in path except the end vertex,
        # v must not be adjacent to p.
        #
        # Let's do the explicit check for correctness.
        ok = True
        if end == 1:
            # Check v against all path vertices except path[-1]
            for idx, p in enumerate(path[:-1]):
                if BIT_COUNTS[v, p] == 1:
                    ok = False
                    break
        else:
            for idx, p in enumerate(path[1:]):
                if BIT_COUNTS[v, p] == 1:
                    ok = False
                    break

        if ok:
            if end == 1:
                new_path = path + [v]
            else:
                new_path = [v] + path
            new_used = used.copy()
            new_used[v] = True
            new_blocked = blocked.copy()
            new_blocked[v] = True
            for bb in range(N):
                new_blocked[v ^ (1 << bb)] = True
            return new_path, new_used, new_blocked

    return None


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    # Precompute bit counts
    precompute_bit_counts()

    # Get initial greedy seed
    best_path = get_greedy_seed()
    best_len = len(best_path)
    best_used, best_blocked = rebuild_used_and_blocked(best_path)

    # Local search with segment replacement
    cur_path = list(best_path)
    cur_used, cur_blocked = rebuild_used_and_blocked(cur_path)

    no_improve = 0
    max_no_improve = 5000

    while time.time() < deadline:
        # Phase 1: Try to extend
        extended = False
        for end in (1, 0, 1, 0):  # Try both ends, alternating
            result = try_extend(cur_path, end, cur_used, cur_blocked)
            if result is not None:
                cur_path, cur_used, cur_blocked = result
                no_improve = 0
                if len(cur_path) > best_len:
                    best_len = len(cur_path)
                    best_path = list(cur_path)
                    best_used, best_blocked = cur_used.copy(), cur_blocked.copy()
                extended = True
                break

        if extended:
            continue

        # Phase 2: Local search - try segment replacements
        # Pick a random internal segment [i, j] and try to replace it with a different path
        # between path[i-1] and path[j+1]
        cur_len = len(cur_path)

        if cur_len < 4:
            break

        # Try random segment replacements
        # Segment length: 2 to min(6, cur_len-2)
        seg_len = min(random.randint(2, 6), cur_len - 2)
        i = random.randint(0, cur_len - seg_len - 1)

        # We're replacing cur_path[i:i+seg_len]
        # The neighbors are cur_path[i-1] (if i > 0) and cur_path[i+seg_len] (if i+seg_len < cur_len)
        # Actually, we need the "gaps" to connect: left_neighbor is path[i-1] if i>0 else None
        # right_neighbor is path[i+seg_len] if i+seg_len < cur_len else None

        # For simplicity, only do internal segments (both neighbors exist)
        if i == 0 or i + seg_len >= cur_len:
            no_improve += 1
            if no_improve > max_no_improve:
                cur_path = list(best_path)
                cur_used, cur_blocked = rebuild_used_and_blocked(cur_path)
                no_improve = 0
            continue

        left_n = cur_path[i - 1]
        right_n = cur_path[i + seg_len]

        # Try to find a random replacement segment of length seg_len
        # connecting left_n to right_n, using only vertices not in the rest of the path
        # This is essentially a random walk constrained to avoid the rest of the path

        # Get the set of "forbidden" vertices: all vertices in cur_path except the segment being replaced
        forbidden = np.zeros(1 << N, dtype=bool)
        for idx in range(i, i + seg_len):
            pass  # Don't forbid the segment vertices for now (we're replacing them)
        for idx in range(cur_len):
            if i <= idx < i + seg_len:
                continue
            forbidden[cur_path[idx]] = True

        # Also forbid vertices adjacent to forbidden vertices? No, that's too aggressive.
        # Instead, we do a short random walk from left_n to right_n of length seg_len steps
        # through unforbidden vertices.

        # Simple approach: try a few random walks
        found_replacement = False
        for attempt in range(20):
            # Random walk of exactly seg_len steps from left_n to right_n
            walk = [left_n]
            visited_in_walk = {left_n}
            current = left_n
            success = True
            for step in range(seg_len):
                # Choose a random neighbor of current that is not forbidden and not visited_in_walk
                candidates = []
                for b in range(N):
                    v = current ^ (1 << b)
                    if not forbidden[v] and v not in visited_in_walk:
                        # Check v is not adjacent to any forbidden vertex (for induced path)
                        # Actually, we need v to not be adjacent to any vertex in the final path.
                        # The final path will include the rest of cur_path (forbidden) plus the new segment.
                        # So v must not be adjacent to any forbidden vertex (except left_n if step==0,
                        # which is already handled since we started from left_n).
                        # For intermediate steps, v must not be adjacent to any forbidden vertex.
                        adjacent_to_forbidden = False
                        for idx in range(cur_len):
                            if i <= idx < i + seg_len:
                                continue
                            if BIT_COUNTS[v, cur_path[idx]] == 1:
                                adjacent_to_forbidden = True
                                break
                        if not adjacent_to_forbidden:
                            candidates.append(v)

                if not candidates:
                    success = False
                    break

                if step == seg_len - 1:
                    # Last step must go to right_n
                    if BIT_COUNTS[current, right_n] == 1:
                        current = right_n
                        walk.append(current)
                    else:
                        success = False
                        break
                else:
                    current = random.choice(candidates)
                    walk.append(current)
                    visited_in_walk.add(current)

            if success and walk[-1] == right_n:
                # Verify the new segment forms a valid induced path segment
                # Check that consecutive vertices in walk differ by 1 bit
                valid = True
                for k in range(len(walk) - 1):
                    if BIT_COUNTS[walk[k], walk[k + 1]] != 1:
                        valid = False
                        break
                if valid:
                    # Check non-consecutive within walk
                    for k in range(len(walk)):
                        for m in range(k + 2, len(walk)):
                            if BIT_COUNTS[walk[k], walk[m]] == 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        # Apply the replacement
                        new_path = cur_path[:i] + walk[1:] + cur_path[i + seg_len:]
                        new_len = len(new_path)
                        if new_len == cur_len:  # Should be the same length
                            # Verify full validity
                            if is_valid_snake(new_path, None, None):
                                cur_path = new_path
                                cur_used, cur_blocked = rebuild_used_and_blocked(cur_path)
                                no_improve = 0
                                found_replacement = True
                                if len(cur_path) > best_len:
                                    best_len = len(cur_path)
                                    best_path = list(cur_path)
                                    best_used, best_blocked = rebuild_used_and_blocked(best_path)
                                break

        if not found_replacement:
            no_improve += 1

        if no_improve > max_no_improve:
            # Restart from best
            cur_path = list(best_path)
            cur_used, cur_blocked = rebuild_used_and_blocked(cur_path)
            no_improve = 0

            # Occasionally try a fresh greedy seed
            if random.random() < 0.3:
                fresh = get_greedy_seed()
                if len(fresh) > len(best_path):
                    best_path = fresh
                    best_len = len(best_path)
                    best_used, best_blocked = rebuild_used_and_blocked(best_path)
                cur_path = list(best_path)
                cur_used, cur_blocked = rebuild_used_and_blocked(cur_path)

    return best_path


# EVOLVE-BLOCK-END
