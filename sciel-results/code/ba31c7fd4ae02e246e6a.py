import time
import numpy as np

N = 17
NUM_BOOLEANS = 1 << N  # 131072
# Precompute the 17-bit masks once (row i has bit i set)
_MASKS = np.arange(N, dtype=np.uint32)

# Precompute bit masks as uint32 array for fast access
_BIT_MASKS = np.array([1 << i for i in range(N)], dtype=np.uint32)

def _check(network):
    """Return True iff `network` sorts every 2^N Boolean input (zero-one principle)."""
    if not network:
        return N == 1
    bits = np.arange(1, 1 << N, dtype=np.uint32)[:, None]  # shape (131072, 1)
    for (i, j) in network:
        bi = bits & _BIT_MASKS[i]
        bj = bits & _BIT_MASKS[j]
        # compare-exchange: wire i gets min, wire j gets max
        bits = np.where(bi < bj, bits, bits | bi | bj)
    # Check sorted: for each adjacent pair, no violation (bit_i=1 and bit_{i+1}=0)
    for i in range(N - 1):
        bi = bits & _BIT_MASKS[i]
        bj = bits & _BIT_MASKS[i + 1]
        if np.any((bi != 0) & (bj == 0)):
            return False
    return True


# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill

    # Seed: odd-even transposition network (known correct, 136 comparators for N=17)
    def odd_even():
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]

    net = odd_even()
    best_net = list(net)
    best_count = len(net)

    if not _check(net):
        return best_net

    # Phase 1: Aggressive single-removal local search with batching
    # Try removing comparators in random order; after each success, restart the order
    idx = 0
    order = None
    removals_this_round = 0
    while time.time() < deadline:
        if order is None or idx >= len(order):
            order = list(range(len(net)))
            np.random.shuffle(order)
            idx = 0
            removals_this_round = 0
            # If we've done a full pass with no removals, try a different strategy
            if removals_this_round == 0 and len(net) <= 100:
                # Try swapping adjacent comparators to enable further removals
                break

        pos = order[idx]
        idx += 1

        candidate = net[:pos] + net[pos + 1:]
        if _check(candidate):
            net = candidate
            best_net = list(net)
            best_count = len(net)
            removals_this_round += 1
            # After a successful removal, re-shuffle to avoid stale ordering
            order = None
            idx = 0

    # Phase 2: Try to swap comparators to create new removal opportunities
    if time.time() < deadline - 5.0:
        swap_attempts = 0
        while time.time() < deadline - 5.0:
            n = len(net)
            if n < 2:
                break
            # Pick two random positions to swap
            p1, p2 = np.random.choice(n, 2, replace=False)
            candidate = net[:]
            candidate[p1], candidate[p2] = candidate[p2], candidate[p1]
            
            # After swap, try to remove one of the swapped positions
            for pos in [p1, p2]:
                candidate2 = candidate[:pos] + candidate[pos + 1:]
                if _check(candidate2):
                    net = candidate2
                    best_net = list(net)
                    best_count = len(net)
                    break
            swap_attempts += 1
            if swap_attempts % 50 == 0:
                break  # Time check more frequently

    # Phase 3: Genetic-style layer restructuring
    # Represent network as layers (parallel groups), try to merge/restructure
    if time.time() < deadline - 2.0:
        def compute_depth(net):
            """Compute the depth (number of layers) of a comparator network."""
            if not net:
                return 0
            # Simple approach: greedily assign layers
            # A comparator can be in the same layer as another if they don't share wires
            depth = 0
            current_layer = []
            used_wires = set()
            for comp in net:
                i, j = comp
                if i not in used_wires and j not in used_wires:
                    current_layer.append(comp)
                    used_wires.add(i)
                    used_wires.add(j)
                else:
                    if current_layer:
                        depth += 1
                        current_layer = [comp]
                        used_wires = {i, j}
                    else:
                        depth += 1
                        current_layer = [comp]
                        used_wires = {i, j}
            if current_layer:
                depth += 1
            return depth

        # Try random insertions of new comparators that might enable more removals
        # This is a heuristic: sometimes adding a comparator allows removing 2+ others
        insert_attempts = 0
        while time.time() < deadline - 2.0 and insert_attempts < 200:
            n = len(net)
            if n == 0:
                break
            # Pick a random position to insert
            pos = np.random.randint(0, n + 1)
            # Pick a random comparator pair
            i, j = np.random.choice(N, 2, replace=False)
            if i > j:
                i, j = j, i
            
            candidate = net[:pos] + [(i, j)] + net[pos:]
            
            # Now try to remove some comparators from the candidate
            # Try removing up to 3 random comparators
            for _ in range(3):
                if len(candidate) <= best_count:
                    break
                m = len(candidate)
                rem_pos = np.random.randint(0, m)
                candidate2 = candidate[:rem_pos] + candidate[rem_pos + 1:]
                if _check(candidate2):
                    candidate = candidate2
                    if len(candidate) < best_count:
                        net = candidate
                        best_net = list(net)
                        best_count = len(net)
            insert_attempts += 1

    return best_net
# EVOLVE-BLOCK-END
