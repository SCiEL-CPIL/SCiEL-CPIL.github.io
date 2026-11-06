"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
TOTAL_INPUTS = 1 << N  # 262144
DEADLINE = 85.0  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Build a sorting network on N=18 wires by combining:
    (1) Greedy removal from a known-good odd-even transposition network.
    (2) Local-search repair: insert comparators to fix broken networks,
        exploring networks unreachable by pure removal.
    (3) Mutations (insert/swap) + repair.
    Verified bit-parallel over all 2^18 Boolean inputs using NumPy.
    """
    start_time = time.time()

    # --- Fast NumPy-based verifier ---
    # Build all 2^N inputs as a (TOTAL_INPUTS, N) uint8 array.
    # wire[i] is a uint8 array of length TOTAL_INPUTS.
    # After applying all comparators, check if sorted:
    #   for all i: wire[i] <= wire[i+1] elementwise.
    #
    # Speed: pre-compute the input matrix once.
    # Each compare-exchange: two array ops (min, max) on length-262144 arrays.
    # For a network of ~80 comparators, that's ~160 array ops.
    # Each op on 262144 uint8 = ~256KB, very fast with NumPy.

    # Pre-compute input matrix
    indices = np.arange(TOTAL_INPUTS, dtype=np.uint32)
    # input_matrix[k, i] = (k >> i) & 1
    input_matrix = np.zeros((TOTAL_INPUTS, N), dtype=np.uint8)
    for i in range(N):
        input_matrix[:, i] = ((indices >> i) & 1).astype(np.uint8)

    def verify_network(network):
        """Verify network sorts all inputs. Returns True if correct."""
        # Copy input matrix
        wires = input_matrix.copy()  # (TOTAL_INPUTS, N)
        for i, j in network:
            wi = wires[:, i]
            wj = wires[:, j]
            # min to i, max to j
            new_wi = np.minimum(wi, wj)
            new_wj = np.maximum(wi, wj)
            wires[:, i] = new_wi
            wires[:, j] = new_wj
        # Check sorted: wire[i] <= wire[i+1] for all i
        for i in range(N - 1):
            if np.any(wires[:, i] > wires[:, i + 1]):
                return False
        return True

    def verify_network_skip(network, skip_idx):
        """Verify network with comparator at skip_idx removed."""
        wires = input_matrix.copy()
        for idx, (i, j) in enumerate(network):
            if idx == skip_idx:
                continue
            wi = wires[:, i]
            wj = wires[:, j]
            new_wi = np.minimum(wi, wj)
            new_wj = np.maximum(wi, wj)
            wires[:, i] = new_wi
            wires[:, j] = new_wj
        for i in range(N - 1):
            if np.any(wires[:, i] > wires[:, i + 1]):
                return False
        return True

    # --- Start with odd-even transposition network ---
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))

    # Verify initial network
    assert verify_network(network), "Initial network should be correct"

    best_network = network[:]
    best_size = len(network)

    # --- Phase 1: Greedy removal ---
    removed = True
    while removed and (time.time() - start_time) < DEADLINE - 20:
        removed = False
        for idx in range(len(network)):
            if (time.time() - start_time) > DEADLINE - 15:
                break
            if verify_network_skip(network, idx):
                network.pop(idx)
                if len(network) < best_size:
                    best_network = network[:]
                    best_size = len(network)
                removed = True
                break  # Restart since indices changed

    # --- Phase 2: Local-search repair (structural change) ---
    # Start from a network slightly larger than best, try to find a smaller
    # valid network by:
    #   a) Randomly remove a comparator (may break correctness)
    #   b) If broken, try to fix by inserting a comparator
    #   c) If fixed and smaller, accept
    #
    # This explores networks NOT reachable by pure removal.

    def repair_network(broken_net, max_inserts=3):
        """Try to fix a broken network by inserting comparators.
        Returns (fixed_network, success)."""
        if verify_network(broken_net):
            return broken_net, True
        # Try inserting comparators at random positions
        for _ in range(max_inserts * 50):
            if (time.time() - start_time) > DEADLINE - 15:
                return None, False
            # Pick random position and random pair
            pos = random.randint(0, len(broken_net))
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            candidate = broken_net[:]
            candidate.insert(pos, (i, j))
            if verify_network(candidate):
                return candidate, True
        return None, False

    if (time.time() - start_time) < DEADLINE - 20:
        # Start from best_network, try to remove and repair
        current = best_network[:]
        for iteration in range(2000):
            if (time.time() - start_time) > DEADLINE - 15:
                break
            if len(current) <= 70:
                break
            # Random removal
            idx = random.randint(0, len(current) - 1)
            candidate = current[:]
            candidate.pop(idx)
            # Try to repair
            fixed, success = repair_network(candidate, max_inserts=2)
            if success and len(fixed) < len(current):
                current = fixed
                if len(current) < best_size:
                    best_network = current[:]
                    best_size = len(current)
            # If not successful, keep current

    # --- Phase 3: Mutation + repair ---
    if (time.time() - start_time) < DEADLINE - 15:
        current = best_network[:]
        for iteration in range(3000):
            if (time.time() - start_time) > DEADLINE - 10:
                break
            if len(current) <= 70:
                break
            mutation_type = random.random()
            if mutation_type < 0.5:
                # Remove a random comparator
                idx = random.randint(0, len(current) - 1)
                candidate = current[:]
                candidate.pop(idx)
            elif mutation_type < 0.8:
                # Insert a random comparator
                pos = random.randint(0, len(current))
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                candidate = current[:]
                candidate.insert(pos, (i, j))
            else:
                # Swap two comparators
                if len(current) < 2:
                    continue
                idx1 = random.randint(0, len(current) - 1)
                idx2 = random.randint(0, len(current) - 1)
                candidate = current[:]
                candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]

            if verify_network(candidate):
                if len(candidate) < len(current):
                    current = candidate
                    if len(current) < best_size:
                        best_network = current[:]
                        best_size = len(current)
            else:
                # Try to repair
                fixed, success = repair_network(candidate, max_inserts=1)
                if success and len(fixed) < len(current):
                    current = fixed
                    if len(current) < best_size:
                        best_network = current[:]
                        best_size = len(current)

    # --- Phase 4: Final greedy removal on best ---
    final_net = best_network[:]
    removed = True
    while removed and (time.time() - start_time) < DEADLINE - 5:
        removed = False
        for idx in range(len(final_net)):
            if (time.time() - start_time) > DEADLINE - 3:
                break
            if verify_network_skip(final_net, idx):
                final_net.pop(idx)
                removed = True
                break
    if len(final_net) < best_size:
        best_network = final_net[:]
        best_size = len(final_net)

    return best_network
# EVOLVE-BLOCK-END
