"""Greedy sorting-network construction via vectorized Boolean simulation."""
import time
import numpy as np

N = 18
NUM_INPUTS = 1 << N  # 262144

# EVOLVE-BLOCK-START
def _build_initial_state():
    """Return initial (wires, viol, remaining) arrays for the identity network."""
    inputs = np.arange(NUM_INPUTS, dtype=np.uint8).reshape(NUM_INPUTS, N)
    wires = inputs.copy()
    viol = np.zeros(NUM_INPUTS, dtype=np.uint8)
    remaining = np.ones((N, N), dtype=bool)
    np.fill_diagonal(remaining, False)
    return wires, viol, remaining

def construct():
    deadline = time.time() + 80.0
    wires, viol, remaining = _build_initial_state()
    network = []
    N2 = N * N
    while True:
        if not remaining.any():
            break
        if time.time() > deadline - 1.0:
            break
        # Score each candidate comparator by reduction in total inversions.
        scores = np.zeros(N2, dtype=np.int32)
        for i in range(N):
            for j in range(i + 1, N):
                if not remaining[i, j]:
                    continue
                a = wires[:, i]
                b = wires[:, j]
                diff = a - b
                scores[i * N + j] = int(np.count_nonzero(diff < 0))
        flat_idx = int(np.argmax(scores))
        i, j = divmod(flat_idx, N)
        gain = scores[flat_idx]
        if gain <= 0:
            break
        a = wires[:, i]
        b = wires[:, j]
        swap = a > b
        wires[:, i] = np.where(swap, b, a)
        wires[:, j] = np.where(swap, a, b)
        viol[swap] ^= 1
        remaining[i, j] = False
        network.append((i, j))
        if not viol.any():
            break
    return network
# EVOLVE-BLOCK-END
