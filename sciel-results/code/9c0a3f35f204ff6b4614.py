import time
import random
import sys

N = 9
MAX_V = 1 << N
MAX_TIME = 115.0  # Safety margin before 150s kill

# Precompute neighbors for all vertices
_NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(MAX_V)]

# Precompute neighbor bitmask for each vertex (bit v is set if v is a neighbor)
_NEIGHBOR_MASK = [0] * MAX_V
for v in range(MAX_V):
    mask = 0
    for b in range(N):
        mask |= 1 << (v ^ (1 << b))
    _NEIGHBOR_MASK[v] = mask


def _warnsdorff_dfs(seed_offset=0, deadline=None):
    """
    Deep backtracking search using Warnsdorff's heuristic with randomized tie-breaking.
    Uses bitmask for fast forbidden-vertex checking.
    Returns the longest path found.
    """
    rng = random.Random(seed_offset)
    best_path = [0]
    best_len = 1
    aborted = False

    sys.setrecursionlimit(10000)

    def dfs(current, forbidden, path):
        nonlocal best_path, best_len, aborted

        if aborted:
            return False

        if deadline is not None and time.time() > deadline:
            aborted = True
            return False

        # Update best
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)

        # Get valid neighbors with Warnsdorff scores
        valid_neighbors = []
        for v in _NEIGHBORS[current]:
            if (forbidden >> v) & 1:
                continue
            # Compute new forbidden set if we move to v
            new_forbidden = forbidden | _NEIGHBOR_MASK[current] | (1 << v)
            # Count valid moves from v
            count = 0
            for u in _NEIGHBORS[v]:
                if not (new_forbidden >> u) & 1:
                    count += 1
            valid_neighbors.append((count, v))

        if not valid_neighbors:
            return False

        # Warnsdorff: sort by count ascending (fewest options first)
        # Add small random perturbation for tie-breaking diversity
        for i in range(len(valid_neighbors)):
            valid_neighbors[i] = (valid_neighbors[i][0] + rng.random() * 0.001, valid_neighbors[i][1])
        valid_neighbors.sort(key=lambda x: x[0])

        for count, v in valid_neighbors:
            if aborted:
                return False

            if deadline is not None and time.time() > deadline:
                aborted = True
                return False

            new_forbidden = forbidden | _NEIGHBOR_MASK[current] | (1 << v)
            path.append(v)

            if dfs(v, new_forbidden, path):
                path.pop()
                return True

            path.pop()

        return False

    path = [0]
    forbidden = 1  # vertex 0 is used
    dfs(0, forbidden, path)

    return best_path


def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME

    best_path = [0]
    best_len = 1

    # Run multiple Warnsdorff DFS attempts with different random seeds
    # The first run is the most important; subsequent runs add diversity
    seed = 0
    while time.time() < deadline - 2:
        p = _warnsdorff_dfs(seed, deadline=deadline)
        if len(p) > best_len:
            best_path = p
            best_len = len(p)
        seed += 1

    return best_path
