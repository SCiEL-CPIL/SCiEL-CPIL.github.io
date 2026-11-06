"""Constraint-based backtracking solver with MRV heuristic for snake-in-the-box in Q_9."""
import time

N = 9
TOTAL = 1 << N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 105.0  # safety margin before 120s limit

    # Precompute neighbors for each vertex
    neighbors = [[] for _ in range(TOTAL)]
    for v in range(TOTAL):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))

    # State for backtracking
    path = [0]
    used = bytearray(TOTAL)
    used[0] = 1
    best_path = list(path)
    best_len = 1

    # Track forbidden vertices (Hamming distance 1 from any non-last path vertex)
    forbidden = bytearray(TOTAL)
    for nb in neighbors[0]:
        forbidden[nb] = 1

    def is_deadline():
        return time.time() > deadline

    # MRV backtracking with time-aware pruning
    def backtrack():
        nonlocal best_path, best_len
        if is_deadline():
            return

        cur = path[-1]

        # Find candidate next vertices: neighbors of cur that are safe
        candidates = []
        for nb in neighbors[cur]:
            if used[nb] or forbidden[nb]:
                continue
            candidates.append(nb)

        if not candidates:
            # Dead end - update best if we have a longer path
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
            return

        # MRV: order candidates by how constrained their future is
        # Prefer candidates with fewer safe future neighbors (more constrained first)
        def score_candidate(v):
            count = 0
            for nb2 in neighbors[v]:
                if nb2 == cur:
                    continue
                if not used[nb2] and not forbidden[nb2]:
                    count += 1
            return count

        candidates.sort(key=score_candidate)

        for v in candidates:
            if is_deadline():
                return

            # Try placing v
            path.append(v)
            used[v] = 1

            # Update forbidden: all neighbors of v become forbidden
            saved_forbidden = []
            for nb in neighbors[v]:
                if not forbidden[nb]:
                    forbidden[nb] = 1
                    saved_forbidden.append(nb)

            backtrack()

            # Undo
            for nb in saved_forbidden:
                forbidden[nb] = 0
            used[v] = 0
            path.pop()

    backtrack()

    return best_path
# EVOLVE-BLOCK-END
