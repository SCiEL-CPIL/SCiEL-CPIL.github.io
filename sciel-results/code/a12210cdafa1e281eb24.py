"""Combined snake-in-the-box program: fast greedy + many randomized restarts."""
import time
import random
import numpy as np

N = 9
FULL = (1 << N) - 1  # mask of N bits
NEIGH_MASKS = [1 << b for b in range(N)]  # bit-flip masks

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before the 150 s kill
    rng = random.Random(12345)

    def greedy_extend(start, bit_order):
        """Fast greedy: extend from start following bit_order; O(1) validity via forbidden set."""
        path = [start]
        used = np.zeros(FULL + 1, dtype=np.int8)
        forbidden = np.zeros(FULL + 1, dtype=np.int8)
        used[start] = 1
        for b in NEIGH_MASKS:
            forbidden[start ^ b] = 1
        while True:
            cur = path[-1]
            nxt = -1
            for b in bit_order:
                v = cur ^ b
                if used[v] == 0 and forbidden[v] == 0:
                    nxt = v
                    break
            if nxt < 0:
                break
            path.append(nxt)
            used[nxt] = 1
            for b in NEIGH_MASKS:
                forbidden[nxt ^ b] = 1
        return path

    # Baseline: canonical greedy from 0 (deterministic, matches the seed's 104).
    best_path = greedy_extend(0, list(NEIGH_MASKS))
    best_len = len(best_path)

    # Randomized restarts: fresh start vertex + shuffled bit order.
    while time.time() < deadline:
        start = rng.randrange(FULL + 1)
        bit_order = list(NEIGH_MASKS)
        rng.shuffle(bit_order)
        path = greedy_extend(start, bit_order)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path

    return best_path
# EVOLVE-BLOCK-END
