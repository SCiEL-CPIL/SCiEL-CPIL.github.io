import time
import random
import math
import numpy as np

N = 17
NUM_INPUTS = 1 << N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # Precompute initial bit-vectors for all 131072 boolean inputs
    # state[j] is a numpy uint64 array of shape (2048,) where each element
    # is a 64-bit word; together they represent a 131072-bit vector.
    # Actually, let's use a different approach: represent each wire's state
    # as a numpy array of uint64, with 2048 elements (2048*64 = 131072).
    NUM_WORDS = NUM_INPUTS // 64  # 2048

    def make_initial_states():
        """Create initial bit-vectors for each wire."""
        states = []
        for j in range(N):
            # For wire j, bit k is set if input k has bit j set
            # Input k: bit j = (k >> j) & 1
            # So for word w (covering inputs w*64 to w*64+63):
            # bit position p in word w corresponds to input w*64+p
            # We want bit p set if ((w*64+p) >> j) & 1 == 1
            arr = np.zeros(NUM_WORDS, dtype=np.uint64)
            for w in range(NUM_WORDS):
                base = w * 64
                for p in range(64):
                    inp = base + p
                    if (inp >> j) & 1:
                        arr[w] |= (1 << p)
            states.append(arr)
        return states

    initial_states = make_initial_states()

    def is_valid(network):
        """Check if network sorts all inputs using vectorized numpy ops."""
        state = [s.copy() for s in initial_states]
        for i, j in network:
            new_i = state[i] & state[j]
            new_j = state[i] | state[j]
            state[i] = new_i
            state[j] = new_j
        # Check sorted: for each k, state[k] & ~state[k+1] should be 0
        for k in range(N - 1):
            if np.any(state[k] & ~state[k + 1]):
                return False
        return True

    def is_valid_fast(network):
        """Faster validity check - early exit on first failure."""
        state = [s.copy() for s in initial_states]
        for i, j in network:
            new_i = state[i] & state[j]
            new_j = state[i] | state[j]
            state[i] = new_i
            state[j] = new_j
        for k in range(N - 1):
            bad = state[k] & ~state[k + 1]
            if np.any(bad):
                return False
        return True

    # Generate seed: odd-even transposition
    def make_seed():
        net = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                net.append((i, i + 1))
        return net

    # Generate a better structured seed: layered network
    def make_structured_seed():
        """Create a structured sorting network with multiple layers."""
        net = []
        # Layer 1: adjacent comparators
        for i in range(0, N - 1, 2):
            net.append((i, i + 1))
        for i in range(1, N - 1, 2):
            net.append((i, i + 1))
        # Layer 2: distance-2 comparators
        for i in range(0, N - 2, 2):
            net.append((i, i + 2))
        for i in range(1, N - 2, 2):
            net.append((i, i + 2))
        # Layer 3: distance-4
        for i in range(0, N - 4, 2):
            net.append((i, i + 4))
        for i in range(1, N - 4, 2):
            net.append((i, i + 4))
        # Layer 4: distance-8
        for i in range(0, N - 8, 2):
            net.append((i, i + 8))
        for i in range(1, N - 8, 2):
            net.append((i, i + 8))
        # Layer 5: long distance
        for i in range(0, N - 1, 1):
            if i + 16 < N:
                net.append((i, i + 16))
        # Final pass of adjacent to fix
        for i in range(0, N - 1, 2):
            net.append((i, i + 1))
        for i in range(1, N - 1, 2):
            net.append((i, i + 1))
        for i in range(0, N - 1, 2):
            net.append((i, i + 1))
        for i in range(1, N - 1, 2):
            net.append((i, i + 1))
        return net

    # Try both seeds, pick the smaller valid one
    seed1 = make_seed()
    seed2 = make_structured_seed()

    best_network = None
    for seed in [seed1, seed2]:
        if is_valid(seed):
            if best_network is None or len(seed) < len(best_network):
                best_network = seed[:]

    if best_network is None:
        return seed1

    random.seed(42)

    def try_remove_all(current_net, max_rounds=10):
        """Aggressively remove redundant comparators."""
        if time.time() > deadline:
            return current_net
        removed = True
        rounds = 0
        while removed and rounds < max_rounds:
            if time.time() > deadline:
                break
            removed = False
            rounds += 1
            n = len(current_net)
            indices = list(range(n))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > deadline:
                    break
                candidate = current_net[:idx] + current_net[idx + 1:]
                if is_valid(candidate):
                    current_net = candidate
                    removed = True
                    break
        return current_network

    # Aggressive removal
    current_network = try_remove_all(best_network, max_rounds=15)
    if len(current_network) < len(best_network):
        best_network = current_network[:]

    # Grow-and-shrink strategy
    def grow_and_shrink(current_net, num_add=3):
        """Add multiple comparators, then aggressively prune."""
        if time.time() > deadline:
            return current_net
        candidate = current_net[:]
        for _ in range(num_add):
            # Add biased toward useful distances
            r = random.random()
            if r < 0.25:
                dist = random.randint(2, 3)
            elif r < 0.50:
                dist = random.randint(4, 6)
            elif r < 0.75:
                dist = random.randint(7, 10)
            else:
                dist = random.randint(11, 16)
            new_i = random.randint(0, N - 1 - dist)
            new_j = new_i + dist
            pos = random.randint(0, len(candidate))
            candidate.insert(pos, (new_i, new_j))

        if not is_valid(candidate):
            return current_net

        # Aggressive pruning
        pruned = try_remove_all(candidate, max_rounds=8)
        return pruned

    # Simulated annealing
    initial_temp = 10.0
    cooling_rate = 0.9985
    min_temp = 0.05

    temperature = initial_temp
    stagnation = 0

    while time.time() < deadline and temperature > min_temp:
        if time.time() > deadline:
            break

        neighbor = current_network[:]
        strategy = random.random()
        n = len(neighbor)

        if strategy < 0.25:
            # Grow-and-shrink
            new_net = grow_and_shrink(current_network, num_add=random.randint(2, 5))
            if len(new_net) <= len(current_network):
                current_network = new_net
                if len(current_network) < len(best_network):
                    best_network = current_network[:]
                    stagnation = 0
                else:
                    stagnation += 1
                temperature *= cooling_rate
                continue

        elif strategy < 0.35:
            # Remove a random comparator
            if n > 1:
                idx = random.randint(0, n - 1)
                neighbor.pop(idx)

        elif strategy < 0.45:
            # Move a comparator to a different position
            if n > 2:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    comp = neighbor.pop(i)
                    if i < j:
                        j -= 1
                    neighbor.insert(j, comp)

        elif strategy < 0.55:
            # Swap two comparators
            if n > 1:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

        elif strategy < 0.70:
            # Replace a comparator with a nearby wire pair
            if n > 0:
                i = random.randint(0, n - 1)
                old_i, old_j = neighbor[i]
                new_i = max(0, min(N - 2, old_i + random.randint(-4, 4)))
                new_j = max(new_i + 1, min(N - 1, old_j + random.randint(-3, 3)))
                if new_i < new_j:
                    neighbor[i] = (new_i, new_j)

        elif strategy < 0.85:
            # Replace with a random long-distance comparator
            if n > 0:
                i = random.randint(0, n - 1)
                dist = random.randint(5, 16)
                new_i = random.randint(0, N - 1 - dist)
                new_j = new_i + dist
                neighbor[i] = (new_i, new_j)

        else:
            # Reorder a segment
            if n > 3:
                start = random.randint(0, n - 4)
                end = random.randint(start + 2, n - 1)
                segment = neighbor[start:end + 1]
                random.shuffle(segment)
                neighbor[start:end + 1] = segment

        # Check validity
        if not is_valid(neighbor):
            temperature *= cooling_rate
            stagnation += 1
            continue

        # Light cleanup
        if time.time() < deadline:
            neighbor = try_remove_all(neighbor, max_rounds=1)

        # Acceptance decision
        current_cost = len(current_network)
        neighbor_cost = len(neighbor)
        delta = neighbor_cost - current_cost

        if delta <= 0:
            current_network = neighbor
            if len(current_network) < len(best_network):
                best_network = current_network[:]
                stagnation = 0
            else:
                stagnation += 1
        else:
            if temperature > 0:
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                    current_network = neighbor
                    stagnation += 1
                else:
                    stagnation += 1

        temperature *= cooling_rate

        # Restart if stuck too long
        if stagnation > 200:
            temperature = min(temperature * 2.0, initial_temp)
            stagnation = 0

    # Final aggressive cleanup
    if time.time() < deadline:
        for _ in range(10):
            if time.time() > deadline:
                break
            final_network = try_remove_all(current_network, max_rounds=10)
            if len(final_network) < len(best_network):
                best_network = final_network[:]
                current_network = best_network[:]

    # One more grow-and-shrink at the end if time permits
    if time.time() < deadline - 5.0:
        for _ in range(3):
            if time.time() > deadline - 2.0:
                break
            new_net = grow_and_shrink(best_network, num_add=random.randint(3, 6))
            if len(new_net) < len(best_network):
                best_network = new_net[:]

    return best_network
# EVOLVE-BLOCK-END
