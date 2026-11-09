import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill

    N = 17
    MIN_D = 6
    max_val = 1 << N  # 131072

    # Precompute popcount for all 2^N words (as numpy uint8)
    weights = np.zeros(max_val, dtype=np.uint8)
    for i in range(1, max_val):
        weights[i] = weights[i >> 1] + (i & 1)

    # Offsets with weight < MIN_D (these are the "conflict" offsets)
    offsets_lt_d = [i for i in range(max_val) if weights[i] < MIN_D]
    offsets_lt_d_set = set(offsets_lt_d)

    # ---- Phase 1: Build the 256-word linear code (greedy basis) ----
    def build_linear_code():
        basis = []
        current_code = [0]
        candidates = list(range(1, max_val))
        for g in candidates:
            if time.time() > deadline - 40:
                break
            if g == 0:
                continue
            valid = True
            for c in current_code:
                if weights[g ^ c] < MIN_D:
                    valid = False
                    break
            if valid:
                basis.append(g)
                old = list(current_code)
                current_code = []
                for c in old:
                    current_code.append(c)
                    current_code.append(c ^ g)
                if len(basis) >= 8:
                    break
        return current_code if len(current_code) >= 256 else None

    base_code = build_linear_code()
    if not base_code or len(base_code) < 256:
        return [0]

    # ---- Phase 2: Nonlinear extension via min-degree greedy with bit-packed forbidden mask ----
    # The forbidden mask is a bit array of length max_val.
    # forbidden[x] = 1 means x is forbidden (either in the code or within distance < MIN_D of the code).
    # We use a numpy uint64 array of length max_val // 64 = 2048.

    def initial_forbidden_mask(base_code):
        mask = np.zeros(max_val // 64, dtype=np.uint64)
        for c in base_code:
            for e in offsets_lt_d:
                x = c ^ e
                mask[x >> 6] |= (1 << (x & 63))
        return mask

    def popcount_uint64(x):
        # popcount for a single uint64
        return bin(x).count('1')

    def min_degree_greedy(base_code, mask, deadline):
        code = list(base_code)
        code_set = set(base_code)
        valid_set = set()

        # Build initial valid set: all x not in forbidden mask
        # We can iterate all x and check the bit.
        # 131072 iterations, each a bit check. Fast.
        for x in range(max_val):
            if not (mask[x >> 6] >> (x & 63)) & 1:
                valid_set.add(x)

        # Greedy loop
        while valid_set:
            if time.time() > deadline:
                break

            # Find min-degree vertex in valid_set
            best_v = None
            min_deg = float('inf')

            for v in valid_set:
                # Count neighbors in valid_set: u = v ^ e for e in offsets_lt_d, u in valid_set
                deg = 0
                for e in offsets_lt_d:
                    u = v ^ e
                    if u in valid_set:
                        deg += 1
                        if deg >= min_deg:
                            break
                if deg < min_deg:
                    min_deg = deg
                    best_v = v

            if best_v is None:
                break

            code.append(best_v)
            valid_set.discard(best_v)

            # Update forbidden mask: set bits for best_v ^ e for e in offsets_lt_d
            for e in offsets_lt_d:
                x = best_v ^ e
                mask[x >> 6] |= (1 << (x & 63))

            # Remove neighbors from valid_set
            neighbors = set()
            for e in offsets_lt_d:
                neighbors.add(best_v ^ e)
            valid_set.difference_update(neighbors)

        return code

    # Run multiple restarts
    best_code = None
    best_len = 0
    num_restarts = 10
    time_per_restart = (deadline - time.time()) / max(num_restarts, 1)

    for r in range(num_restarts):
        restart_deadline = start_time + 55 - 2  # leave 2s buffer
        if time.time() > restart_deadline - 1:
            break

        # Fresh mask for each restart
        mask = initial_forbidden_mask(base_code)
        code = min_degree_greedy(base_code, mask, restart_deadline)

        if len(code) > best_len:
            best_len = len(code)
            best_code = code

    if best_code is None:
        return base_code

    return best_code
