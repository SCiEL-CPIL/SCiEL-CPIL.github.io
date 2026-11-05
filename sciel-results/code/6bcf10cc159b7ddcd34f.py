<<<<<<< SEARCH
def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')

    # Try multiple random strategies
    base_seed = 12345
    attempts = 0
    max_attempts = 150

    while time.time() < deadline - 1.0:
        seed = base_seed + attempts
        rng = random.Random(seed)
        pieces = _decompose_greedy(edges, n, deadline - 0.5, rng)
        cnt = len(pieces)

        if cnt < best_count:
            best_count = cnt
            best = pieces

        attempts += 1

        if attempts >= max_attempts:
            break

    # Also try the flow-based approach if time permits
    if time.time() < deadline - 5.0:
        flow_pieces = _decompose_flow(edges, n, deadline - 2.0)
        if len(flow_pieces) < best_count:
            best_count = len(flow_pieces)
            best = flow_pieces

    return best if best else [[u, v] for u, v in edges]
=======
def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')

    # Try multiple random strategies
    base_seed = 12345
    attempts = 0
    max_attempts = 150

    while time.time() < deadline - 1.0:
        seed = base_seed + attempts
        rng = random.Random(seed)
        pieces = _decompose_greedy(edges, n, deadline - 0.5, rng)
        cnt = len(pieces)

        if cnt < best_count:
            best_count = cnt
            best = pieces

        attempts += 1

        if attempts >= max_attempts:
            break

    # Also try the flow-based approach if time permits
    if time.time() < deadline - 5.0:
        flow_pieces = _decompose_flow(edges, n, deadline - 2.0)
        if len(flow_pieces) < best_count:
            best_count = len(flow_pieces)
            best = flow_pieces

    return best if best else [[u, v] for u, v in edges]
>>>>>>> REPLACE

