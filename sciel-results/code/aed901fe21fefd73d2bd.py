"""Size-efficient sorting network on 18 wires using Partial Order Derivation with bit-parallel verification."""
N = 18
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 56.0  # Safety margin before 60s kill

    def odd_even_transposition():
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators

    # Pre-compute bit masks for initialization
    NUM_WORDS = (1 << N) // 64  # 4096 words of 64 bits each

    # Pre-compute initial wires using efficient bit packing
    total_inputs = 1 << N
    indices = np.arange(total_inputs, dtype=np.uint32)
    weights = 1 << np.arange(64, dtype=np.uint64)

    initial_wires = [None] * N
    for i in range(N):
        bits_i = (indices >> i) & 1
        bits_i_64 = bits_i.astype(np.uint64)
        packed = (bits_i_64.reshape(NUM_WORDS, 64) * weights).sum(axis=1)
        initial_wires[i] = packed

    def verify_network(comparators):
        """
        Fast verification using bit-parallelism.
        Returns True if the network sorts all Boolean inputs.
        """
        wires = [w.copy() for w in initial_wires]

        for (i, j) in comparators:
            wi = wires[i]
            wj = wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj

        # Check sorted: for all k, wires[k] is subset of wires[k+1]
        for k in range(N - 1):
            if np.any(wires[k] & ~wires[k + 1]):
                return False
        return True

    def greedy_remove(comparators, max_rounds=50):
        """
        Iteratively remove redundant comparators.
        Uses systematic sampling to find removable comparators.
        """
        current = list(comparators)
        changed = True
        rounds = 0
        while changed and rounds < max_rounds:
            changed = False
            if time.time() > deadline - 1.0:
                break
            rounds += 1
            # Try removing from random positions, biased toward ends
            n = len(current)
            if n == 0:
                break
            # Try a few random positions, prioritizing ends
            positions = []
            num_samples = min(50, n)
            for _ in range(num_samples):
                pos = random.randint(0, n - 1)
                if pos not in positions:
                    positions.append(pos)
            # Also try first and last
            if n > 1:
                if 0 not in positions:
                    positions.append(0)
                if n - 1 not in positions:
                    positions.append(n - 1)

            for idx in positions:
                if idx >= len(current):
                    continue
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    changed = True
                    break

        return current

    # Start with odd-even transposition
    current = odd_even_transposition()
    current = greedy_remove(current)

    best_network = list(current)
    best_count = len(current)

    random.seed(42)

    # Local Search with diverse moves
    while time.time() < deadline - 2.0:
        if not current:
            break

        candidate = list(current)
        n_comp = len(candidate)

        if n_comp == 0:
            break

        r = random.random()

        if r < 0.35:
            # Move: Pick a comparator and move it to a new position
            if n_comp > 1:
                i = random.randint(0, n_comp - 1)
                j = random.randint(0, n_comp - 1)
                comp = candidate.pop(i)
                if i < j:
                    j -= 1
                j = max(0, min(j, len(candidate)))
                candidate.insert(j, comp)
        elif r < 0.65:
            # Swap: Swap two comparators
            if n_comp > 1:
                i, j = random.sample(range(n_comp), 2)
                candidate[i], candidate[j] = candidate[j], candidate[i]
        elif r < 0.85:
            # Replace: Replace a comparator with a random one
            i = random.randint(0, n_comp - 1)
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            candidate[i] = (u, v)
        else:
            # Targeted replace: Replace with a comparator near an existing one
            i = random.randint(0, n_comp - 1)
            old_u, old_v = candidate[i]
            # Try wires near the old comparator
            candidates = []
            for du in [-1, 0, 1]:
                for dv in [-1, 0, 1]:
                    nu = old_u + du
                    nv = old_v + dv
                    if 0 <= nu < N and 0 <= nv < N and nu != nv:
                        if nu > nv:
                            nu, nv = nv, nu
                        if (nu, nv) != (old_u, old_v):
                            candidates.append((nu, nv))
            if candidates:
                candidate[i] = random.choice(candidates)
            else:
                continue

        # Verify candidate
        if verify_network(candidate):
            # Accept and try to reduce
            candidate = greedy_remove(candidate)

            if len(candidate) < best_count:
                best_count = len(candidate)
                best_network = list(candidate)
                current = candidate
            else:
                # Accept if size is <= current size to allow exploration
                if len(candidate) <= len(current):
                    current = candidate

    # Final verification
    if not verify_network(best_network):
        best_network = odd_even_transposition()

    return best_network
