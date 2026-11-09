<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    """Return a correct compare-exchange network (list of (i, j), i<j) for N wires.

    Starts from the odd-even transposition network and, within a wall-clock budget,
    greedily removes redundant comparators (verified over all 2**N Boolean inputs).
    """
    net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    best = list(net)
    best_len = len(net)
    deadline = _seed_time + DEADLINE

    while time.time() < deadline:
        improved = False
        for idx in range(len(net) - 1, -1, -1):
            if time.time() >= deadline:
                break
            cand = net[:idx] + net[idx + 1:]
            if _is_sorted_network(cand):
                net = cand
                best = list(net)
                best_len = len(net)
                improved = True
                break
        if not improved:
            break

    return best


def _is_sorted_network(net):
    """True iff the network sorts every 2**N Boolean input (zero/one principle)."""
    M = len(net)
    for mask in range(1 << N):
        v = list(range(N))
        for i in range(N):
            v[i] = 1 if (mask >> i) & 1 else 0
        for i, j in net:
            a, b = v[i], v[j]
            if a > b:
                v[i], v[j] = b, a
        ok = True
        for i in range(N - 1):
            if v[i] > v[i + 1]:
                ok = False
                break
        if not ok:
            return False
    return True
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct():
    """Return a correct compare-exchange network (list of (i, j), i<j) for N wires.

    Starts from the odd-even transposition network and, within a wall-clock budget,
    greedily removes redundant comparators (verified over all 2**N Boolean inputs).
    Uses NumPy vectorization for fast verification.
    """
    net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    best = list(net)
    deadline = _seed_time + DEADLINE

    # Precompute all masks as a 2D array of shape (2^N, N)
    masks = np.arange(1 << N, dtype=np.int32)
    all_inputs = np.zeros((1 << N, N), dtype=np.int8)
    for i in range(N):
        all_inputs[:, i] = (masks >> i) & 1

    while time.time() < deadline:
        improved = False
        for idx in range(len(net) - 1, -1, -1):
            if time.time() >= deadline:
                break
            cand = net[:idx] + net[idx + 1:]
            if _is_sorted_network_fast(cand, all_inputs):
                net = cand
                best = list(net)
                improved = True
                break
        if not improved:
            break

    return best


def _is_sorted_network_fast(net, all_inputs):
    """True iff the network sorts every 2**N Boolean input (zero/one principle).
    
    Uses NumPy vectorization: simulates all 2^N inputs simultaneously.
    """
    if not net:
        # Empty network only sorts if all inputs are already sorted
        # Only mask 0 (all zeros) and mask (2^N - 1) (all ones) are sorted
        return False

    v = all_inputs.copy()  # shape (2^N, N)
    
    for i, j in net:
        a = v[:, i]
        b = v[:, j]
        # min goes to i, max goes to j
        v[:, i] = np.minimum(a, b)
        v[:, j] = np.maximum(a, b)
    
    # Check if all rows are sorted (non-decreasing)
    # v[:, k] <= v[:, k+1] for all k
    diffs = np.diff(v, axis=1)  # shape (2^N, N-1)
    return bool(np.all(diffs >= 0))
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
