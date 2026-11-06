import time
import random
import numpy as np

N = 9
MAX_V = 1 << N
MASK_ALL = (1 << MAX_V) - 1

# Precompute neighbors
neighbors = [[] for _ in range(MAX_V)]
for v in range(MAX_V):
    for b in range(N):
        neighbors[v].append(v ^ (1 << b))

# Precompute popcount
_popcount = np.zeros(MAX_V, dtype=np.uint8)
for i in range(MAX_V):
    _popcount[i] = bin(i).count('1')

# For each vertex, precompute which other vertices are at distance >= 2
# We store as a set of forbidden (distance-1) vertices for quick checking
# dist1_mask[v] = bitmask of vertices at Hamming distance 1 from v
dist1_mask = [0] * MAX_V
for v in range(MAX_V):
    m = 0
    for nb in neighbors[v]:
        m |= (1 << nb)
    dist1_mask[v] = m


def construct():
    deadline = time.time() + 115.0
    best_path = []
    best_len = 0

    def is_valid_placement(path, used_mask, candidate):
        """Check if candidate can be added to path (adjacent to end, dist>=2 from all others)."""
        cur = path[-1]
        # Must be adjacent to cur
        if not ((candidate ^ cur) & ((candidate ^ cur) - 1) == 0):
            return False
        # Must be at distance >= 2 from all other used vertices except cur
        mask_others = used_mask ^ (1 << cur)
        # candidate must not be in dist1 of any vertex in mask_others
        # Equivalently: mask_others must not intersect dist1_mask[candidate]
        if (mask_others & dist1_mask[candidate]) != 0:
            return False
        return True

    def count_valid_extensions(cur, used_mask, exclude_cur=True):
        """Count how many valid candidates exist for extending from cur."""
        count = 0
        for nb in neighbors[cur]:
            if (used_mask >> nb) & 1:
                continue
            if exclude_cur:
                mask_others = used_mask ^ (1 << cur)
            else:
                mask_others = used_mask
            if (mask_others & dist1_mask[nb]) != 0:
                continue
            count += 1
        return count

    def greedy_extend(path, used_mask, randomize=True, max_steps=500):
        """Greedily extend using Warnsdorff heuristic."""
        nonlocal deadline
        steps = 0
        while steps < max_steps:
            if time.time() > deadline:
                return path, used_mask
            cur = path[-1]
            candidates = []
            for nb in neighbors[cur]:
                if (used_mask >> nb) & 1:
                    continue
                mask_others = used_mask ^ (1 << cur)
                if (mask_others & dist1_mask[nb]) != 0:
                    continue
                # Score: number of valid extensions from nb
                new_used = used_mask | (1 << nb)
                deg = 0
                for nn in neighbors[nb]:
                    if (new_used >> nn) & 1:
                        continue
                    mask_others_nb = new_used ^ (1 << nb)
                    if (mask_others_nb & dist1_mask[nn]) != 0:
                        continue
                    deg += 1
                candidates.append((deg, nb))
            if not candidates:
                break
            candidates.sort(key=lambda x: x[0])
            if randomize and len(candidates) > 1:
                k = min(3, len(candidates))
                chosen = random.choice(candidates[:k])[1]
            else:
                chosen = candidates[0][1]
            path.append(chosen)
            used_mask |= (1 << chosen)
            steps += 1
        return path, used_mask

    def extend_both_ends(path, used_mask, max_iters=20):
        """Try to extend the path at both ends repeatedly."""
        nonlocal deadline
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # Try start
            first = path[0]
            for nb in neighbors[first]:
                if (used_mask >> nb) & 1:
                    continue
                mask_others = used_mask ^ (1 << first)
                if (mask_others & dist1_mask[nb]) != 0:
                    continue
                path.insert(0, nb)
                used_mask |= (1 << nb)
                improved = True
                break
            if improved:
                continue
            # Try end
            last = path[-1]
            for nb in neighbors[last]:
                if (used_mask >> nb) & 1:
                    continue
                mask_others = used_mask ^ (1 << last)
                if (mask_others & dist1_mask[nb]) != 0:
                    continue
                path.append(nb)
                used_mask |= (1 << nb)
                improved = True
                break
            if not improved:
                break
        return path, used_mask

    def backtrack(path, used_mask, depth_limit=30):
        """Backtracking search with Warnsdorff ordering and time limit."""
        nonlocal best_path, best_len, deadline

        if time.time() > deadline:
            return path, used_mask

        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]

        if len(path) >= depth_limit:
            return path, used_mask

        cur = path[-1]
        candidates = []
        for nb in neighbors[cur]:
            if (used_mask >> nb) & 1:
                continue
            mask_others = used_mask ^ (1 << cur)
            if (mask_others & dist1_mask[nb]) != 0:
                continue
            new_used = used_mask | (1 << nb)
            deg = 0
            for nn in neighbors[nb]:
                if (new_used >> nn) & 1:
                    continue
                mask_others_nb = new_used ^ (1 << nb)
                if (mask_others_nb & dist1_mask[nn]) != 0:
                    continue
                deg += 1
            candidates.append((deg, nb))

        if not candidates:
            return path, used_mask

        candidates.sort(key=lambda x: x[0])

        for deg, nb in candidates:
            if time.time() > deadline:
                return path, used_mask
            path.append(nb)
            used_mask |= (1 << nb)
            path, used_mask = backtrack(path, used_mask, depth_limit)
            path.pop()
            used_mask ^= (1 << nb)

        return path, used_mask

    def simulated_annealing(path, used_mask, iterations=20000):
        """SA with 2-bit flip moves that preserve path validity."""
        nonlocal best_path, best_len, deadline

        current = path[:]
        current_mask = used_mask
        current_len = len(current)
        best_local = current[:]
        best_local_mask = current_mask
        best_local_len = current_len

        T = 5.0
        cooling = 0.9998

        for it in range(iterations):
            if time.time() > deadline:
                break

            # Pick a random position in the path to try a 2-bit flip
            if len(current) < 3:
                break
            pos = random.randint(1, len(current) - 2)
            v = current[pos]
            # Try flipping two bits
            b1 = random.randint(0, N - 1)
            b2 = random.randint(0, N - 1)
            while b2 == b1:
                b2 = random.randint(0, N - 1)
            new_v = v ^ (1 << b1) ^ (1 << b2)

            if new_v == v:
                continue
            if (current_mask >> new_v) & 1:
                continue

            # Check if new_v is at distance >= 2 from all vertices in the path
            # except its neighbors in the path (pos-1 and pos+1)
            ok = True
            for i, u in enumerate(current):
                if i == pos - 1 or i == pos + 1:
                    continue
                if _popcount[new_v ^ u] < 2:
                    ok = False
                    break
            if not ok:
                continue

            # Also check new_v is adjacent to both neighbors
            if _popcount[new_v ^ current[pos - 1]] != 1:
                continue
            if _popcount[new_v ^ current[pos + 1]] != 1:
                continue

            # Valid move
            current_mask ^= (1 << v)
            current_mask |= (1 << new_v)
            current[pos] = new_v

            # Check if this improves best
            if len(current) > best_local_len:
                best_local_len = len(current)
                best_local = current[:]
                best_local_mask = current_mask

            # Accept with probability
            # Energy: we want to keep the path valid (it is), so energy is 0
            # We accept all valid moves but with cooling to explore
            if random.random() > T:
                # Reject: undo
                current[pos] = v
                current_mask ^= (1 << new_v)
                current_mask |= (1 << v)

            T *= cooling

        if best_local_len > best_len:
            best_len = best_local_len
            best_path = best_local[:]
        return best_local, best_local_mask

    # Main search
    random.seed(42)

    # Phase 1: Multiple greedy restarts with randomization
    for attempt in range(200):
        if time.time() > deadline - 30:
            break
        start = random.randint(0, MAX_V - 1)
        path = [start]
        used_mask = 1 << start
        path, used_mask = greedy_extend(path, used_mask, randomize=True, max_steps=300)
        path, used_mask = extend_both_ends(path, used_mask, max_iters=15)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]

    # Phase 2: Backtracking from best path's start
    if best_path:
        start = best_path[0]
        path = [start]
        used_mask = 1 << start
        path, used_mask = backtrack(path, used_mask, depth_limit=25)
        path, used_mask = extend_both_ends(path, used_mask, max_iters=15)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]

    # Phase 3: SA refinement on best path
    if best_path and time.time() < deadline - 10:
        path, used_mask = simulated_annealing(best_path[:], 0, iterations=50000)
        # Rebuild used_mask
        used_mask = 0
        for v in path:
            used_mask |= (1 << v)
        path, used_mask = extend_both_ends(path, used_mask, max_iters=15)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]

    # Phase 4: More greedy restarts if time remains
    remaining = deadline - time.time()
    if remaining > 15:
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            start = random.randint(0, MAX_V - 1)
            path = [start]
            used_mask = 1 << start
            path, used_mask = greedy_extend(path, used_mask, randomize=True, max_steps=400)
            path, used_mask = extend_both_ends(path, used_mask, max_iters=20)
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]

    if not best_path:
        return [0]
    return best_path
