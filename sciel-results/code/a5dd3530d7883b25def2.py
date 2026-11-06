"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 9
MASK = (1 << N) - 1

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 115.0  # safety margin before 120s kill
    rng = random.Random(42)

    # Precompute: for each vertex, the bitmask of its N neighbours.
    NEIGHBORS = [(1 << (v ^ (1 << b))) for v in range(1 << N) for b in range(N)]
    NEIGHBORS = np.array(NEIGHBORS, dtype=np.uint32).reshape(1 << N, N)

    def get_candidates(cur, used, blocked):
        """Vertices adjacent to cur, unused, and not at Hamming distance 1 from
        any path vertex (so they stay non-adjacent to all but the last)."""
        cands = []
        for b in range(N):
            v = cur ^ (1 << b)
            if (used >> v) & 1:
                continue
            if (blocked >> v) & 1:
                continue
            cands.append(v)
        return cands

    def future_options(v, used, blocked):
        """How many valid moves remain after appending v (fail-first heuristic)."""
        nb = NEIGHBORS[v]
        new_blocked = blocked | int(nb)
        c = 0
        for b in range(N):
            w = v ^ (1 << b)
            if (used >> w) & 1:
                continue
            if (new_blocked >> w) & 1:
                continue
            c += 1
        return c

    def greedy_extend(path, used, blocked, bit_order, randomize=0.0):
        """Extend greedily using the given bit order (with optional randomization)."""
        while True:
            cur = path[-1]
            cands = get_candidates(cur, used, blocked)
            if not cands:
                break
            if randomize > 0.0 and rng.random() < randomize:
                v = cands[rng.randrange(len(cands))]
            else:
                # fail-first: pick the candidate leaving the fewest future options
                best = min(cands, key=lambda v: future_options(v, used, blocked))
                v = best
            path.append(v)
            used |= 1 << v
            blocked |= int(NEIGHBORS[v])

    def backtrack(path, used, blocked, depth_limit):
        """Constraint backtracking with fail-first ordering. Returns True on success."""
        if time.time() > deadline:
            return False
        if len(path) >= depth_limit:
            return True

        cur = path[-1]
        cands = get_candidates(cur, used, blocked)
        if not cands:
            return False

        scored = sorted((future_options(v, used, blocked), v) for v in cands)
        for _, v in scored:
            path.append(v)
            used |= 1 << v
            blocked |= int(NEIGHBORS[v])
            if backtrack(path, used, blocked, depth_limit):
                return True
            path.pop()
            used &= ~(1 << v)
            blocked &= ~int(NEIGHBORS[v])
            if time.time() > deadline:
                return False
        return False

    best_path = [0]
    best_len = 1

    # Build a diverse set of bit orderings: identity + random permutations.
    orderings = [list(range(N))]
    for _ in range(40):
        o = list(range(N))
        rng.shuffle(o)
        orderings.append(o)

    for bit_ord in orderings:
        if time.time() > deadline:
            break

        # Greedy extension with this bit order.
        path, used, blocked = [0], 1, int(NEIGHBORS[0])
        greedy_extend(path, used, blocked, bit_ord)
        if len(path) > best_len:
            best_path, best_len = list(path), len(path)

        # Try to push one (or a few) steps further with backtracking.
        target = best_len + 1
        while target <= best_len + 8 and time.time() < deadline - 5:
            if backtrack(path, used, blocked, target):
                best_path, best_len = list(path), len(path)
                target = best_len + 1
            else:
                break

    # A couple of randomized greedy runs to shake out other local optima.
    for _ in range(6):
        if time.time() > deadline:
            break
        path, used, blocked = [0], 1, int(NEIGHBORS[0])
        greedy_extend(path, used, blocked, orderings[rng.randrange(len(orderings))], randomize=0.3)
        if len(path) > best_len:
            best_path, best_len = list(path), len(path)
        if time.time() < deadline - 8:
            if backtrack(path, used, blocked, best_len + 1):
                best_path, best_len = list(path), len(path)

    # Final validation (induced path: consecutive differ by 1 bit, others by >= 2).
    def validate(p):
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                d = (p[i] ^ p[j]).bit_count()
                if abs(i - j) == 1:
                    if d != 1:
                        return False
                else:
                    if d == 1:
                        return False
        return True

    if not validate(best_path):
        # Fallback: plain greedy with identity bit order.
        path, used, blocked = [0], 1, int(NEIGHBORS[0])
        greedy_extend(path, used, blocked, list(range(N)))
        best_path = path

    return best_path

# EVOLVE-BLOCK-END
