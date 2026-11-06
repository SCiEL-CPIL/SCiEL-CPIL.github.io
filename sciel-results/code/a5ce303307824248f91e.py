import time
import random
import numpy as np

N = 17
MAX_TIME = 55.0  # Safety margin for 60s limit

# Precompute valid sorted states for quick lookup
# A state is sorted if it is 0^a 1^b.
# Values: 0, 2^17-2^16, 2^17-2^15, ..., 2^17-1
_VALID_MASK = np.zeros(1 << N, dtype=bool)
for k in range(N + 1):
    if k == 0:
        val = 0
    else:
        val = (1 << N) - (1 << (N - k))
    _VALID_MASK[val] = True


def seed_network():
    """Odd-even transposition network (136 comparators)."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net


def verify_network(net):
    """
    Fast verification using Python integers (bit-vector simulation).
    Checks if the network sorts all 2^17 boolean inputs.
    """
    if not net:
        return N <= 1

    # Start with all possible inputs as a list of integers
    states = list(range(1 << N))

    # Apply each comparator
    for i, j in net:
        mask_i = 1 << i
        mask_j = 1 << j
        swap_mask = mask_i | mask_j
        # For each state, if bit i is 1 and bit j is 0, swap them
        # We can do this efficiently by checking the condition
        # Actually, let's use a list comprehension or in-place loop
        # Since we need speed, let's try a different approach:
        # For each state, compute new state
        # But 131072 * 136 operations = ~17.8M operations, might be slow in Python
        # Let's use numpy for vectorization
        pass

    # Actually, let's use the numpy approach which was in the original
    states = np.arange(1 << N, dtype=np.uint32)
    for i, j in net:
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        mask = (bi == 1) & (bj == 0)
        if np.any(mask):
            states[mask] ^= (1 << i) | (1 << j)

    return bool(np.all(_VALID_MASK[states]))


def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME

    rng = random.Random(42)

    # Start with seed
    best_net = seed_network()
    best_count = len(best_net)

    # Phase 1: Aggressive Greedy Minimization
    current = list(best_net)

    def minimize(net, time_limit):
        """Greedy removal of redundant comparators."""
        current = list(net)
        progress = True
        while progress and time.time() < time_limit:
            progress = False
            indices = list(range(len(current)))
            rng.shuffle(indices)
            for idx in indices:
                if time.time() > time_limit:
                    break
                new_net = current[:idx] + current[idx+1:]
                if verify_network(new_net):
                    current = new_net
                    progress = True
                    break
        return current

    # Initial minimization
    current = minimize(current, deadline)
    if len(current) < best_count:
        best_net = current
        best_count = len(current)
        print(f"Phase 1: Reduced to {best_count}", flush=True)

    # Phase 2: Random Restart Hill Climbing
    while time.time() < deadline:
        move_type = rng.random()

        if len(current) < 2:
            break

        if move_type < 0.4:
            # Swap two random comparators
            i, j = rng.sample(range(len(current)), 2)
            new_net = list(current)
            new_net[i], new_net[j] = new_net[j], new_net[i]

        elif move_type < 0.7:
            # Replace a random comparator with a new random one
            idx = rng.randint(0, len(current) - 1)
            new_i = rng.randint(0, N - 1)
            new_j = rng.randint(0, N - 1)
            if new_i == new_j:
                continue
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            new_net = list(current)
            new_net[idx] = (new_i, new_j)

        elif move_type < 0.9:
            # Add a random comparator
            new_i = rng.randint(0, N - 1)
            new_j = rng.randint(0, N - 1)
            if new_i == new_j:
                continue
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            pos = rng.randint(0, len(current))
            new_net = current[:pos] + [(new_i, new_j)] + current[pos:]

        else:
            # Restart from best known or minimize again
            new_net = list(current)
            minimized = minimize(new_net, deadline)
            if len(minimized) < len(current):
                current = minimized
                if len(current) < best_count:
                    best_net = current
                    best_count = len(current)
                    print(f"Phase 2 (minimize): Improved to {best_count}", flush=True)
            continue

        # Verify the new network
        if verify_network(new_net):
            if time.time() < deadline - 0.1:
                minimized = minimize(new_net, deadline)
                if len(minimized) < len(current):
                    current = minimized
                    if len(current) < best_count:
                        best_net = current
                        best_count = len(current)
                        print(f"Phase 2: Improved to {best_count}", flush=True)
                else:
                    current = minimized
            else:
                current = new_net
                if len(current) < best_count:
                    best_net = current
                    best_count = len(current)
        else:
            pass

    # Final safety check
    if not verify_network(best_net):
        best_net = seed_network()

    return best_net


if __name__ == "__main__":
    net = construct()
    print(f"Result size: {len(net)}")
    print(f"Valid: {verify_network(net)}")
