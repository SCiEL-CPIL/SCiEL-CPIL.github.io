"""Snake-in-the-box in Q_9: beam search with forbidden-set validity and randomized restarts."""
import time
import random

N = 9
TOTAL = 1 << N
DEADLINE = 140.0  # safety margin before 150s kill


def _neighbors_set(v):
    """Return set of all 9 neighbors of vertex v."""
    return {v ^ (1 << b) for b in range(N)}


def _greedy(seed):
    """Greedy snake with randomized tie-breaking."""
    rng = random.Random(seed)
    path = [0]
    used = {0}
    forbidden = set()  # neighbors of v_0..v_{k-1}; initially empty since k=0

    while True:
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v not in used and v not in forbidden:
                candidates.append(v)

        if not candidates:
            break

        rng.shuffle(candidates)
        nxt = candidates[0]
        path.append(nxt)
        used.add(nxt)
        forbidden |= _neighbors_set(cur)

    return path


def _beam_search(width, seed, time_limit):
    """Beam search for snake. Returns best path found."""
    rng = random.Random(seed)
    start = time.time()

    initial_path = [0]
    initial_used = {0}
    initial_forbidden = set()

    beam = [(initial_path, initial_used, initial_forbidden)]
    best_path = initial_path[:]
    best_len = len(initial_path)

    while time.time() - start < time_limit:
        new_beam = []
        for path, used, forbidden in beam:
            cur = path[-1]
            candidates = []
            for b in range(N):
                v = cur ^ (1 << b)
                if v not in used and v not in forbidden:
                    candidates.append(v)

            if not candidates:
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path[:]
                continue

            rng.shuffle(candidates)

            for v in candidates:
                new_path = path + [v]
                new_used = used | {v}
                new_forbidden = forbidden | _neighbors_set(cur)
                new_beam.append((new_path, new_used, new_forbidden))
                if len(new_path) > best_len:
                    best_len = len(new_path)
                    best_path = new_path[:]

        if not new_beam:
            break

        new_beam = new_beam[:width]
        beam = new_beam

    return best_path


def construct():
    start = time.time()
    best_path = None
    best_len = 0

    # Quick greedy with several seeds
    for seed in range(30):
        path = _greedy(seed)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]

    elapsed = time.time() - start
    remaining = DEADLINE - elapsed

    if remaining < 5:
        return best_path

    # Beam search with escalating widths
    configs = [
        (50, 100, 15),
        (100, 200, 20),
        (200, 300, 25),
        (400, 400, 30),
    ]

    for width, seed, time_budget in configs:
        if time.time() - start > DEADLINE - 5:
            break
        actual_budget = min(time_budget, DEADLINE - 5 - (time.time() - start))
        if actual_budget < 3:
            break
        result = _beam_search(width, seed, actual_budget)
        if len(result) > best_len:
            best_len = len(result)
            best_path = result[:]

    # Use remaining time for randomized beam searches
    while time.time() - start < DEADLINE - 5:
        width = random.randint(100, 500)
        seed = random.randint(0, 10**9)
        remaining_time = DEADLINE - 5 - (time.time() - start)
        if remaining_time < 5:
            break
        result = _beam_search(width, seed, remaining_time)
        if len(result) > best_len:
            best_len = len(result)
            best_path = result[:]

    return best_path
