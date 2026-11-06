import numpy as np
import time
import random

N = 16

# EVOLVE-BLOCK-START
def construct():
    # Batcher's bitonic sort network for 16 wires: 60 comparators
    def bitonic_network(n):
        comparators = []
        def sort_stage(start, size, ascending):
            if size == 1:
                return
            half = size // 2
            sort_stage(start, half, not ascending)
            sort_stage(start + half, half, ascending)
            merge_stage(start, half, ascending)

        def merge_stage(start, size, ascending):
            if size == 1:
                return
            half = size // 2
            for i in range(half):
                a = start + i
                b = start + i + half
                if ascending:
                    comparators.append((a, b))
                else:
                    comparators.append((b, a))
            merge_stage(start, half, ascending)
            merge_stage(start + half, half, ascending)

        sort_stage(0, n, True)
        return comparators

    initial_net = bitonic_network(N)

    # Pre-compute inputs, popcounts, targets, and per-wire bit masks
    inputs = np.arange(2**N, dtype=np.uint32)
    popcounts = np.array([bin(x).count('1') for x in range(2**N)], dtype=np.uint8)
    targets = np.array([((1 << k) - 1) << (N - k) if k > 0 else 0 for k in popcounts], dtype=np.uint32)

    # Per-wire bit masks: mask[i][x] = 1 if wire i of input x is set
    wire_masks = np.array([((inputs >> i) & 1) for i in range(N)], dtype=np.uint8)  # shape (N, 65536)

    def apply_comparator(masks, i, j):
        """Apply compare-exchange on wires i,j to the array of masks (uint32)."""
        bit_i = np.uint32(1 << i)
        bit_j = np.uint32(1 << j)
        # cond: wire i has 1 and wire j has 0 -> swap
        wi = wire_masks[i]  # bool-like uint8
        wj = wire_masks[j]
        cond = (wi == 1) & (wj == 0)
        if np.any(cond):
            masks = np.where(cond, masks ^ bit_i ^ bit_j, masks)
        return masks

    def verify(net):
        masks = inputs.copy()
        for i, j in net:
            masks = apply_comparator(masks, i, j)
        return np.array_equal(masks, targets)

    def verify_net(masks_before, net_slice):
        """Verify a slice of net starting from pre-computed masks_before."""
        masks = masks_before
        for i, j in net_slice:
            masks = apply_comparator(masks, i, j)
        return np.array_equal(masks, targets)

    # Verify initial
    if not verify(initial_net):
        # Fallback to odd-even transposition
        initial_net = [(i, i + 1) for phase in range(N)
                       for i in range(phase % 2, N - 1, 2)]
        if not verify(initial_net):
            return initial_net

    current_net = initial_net
    best_net = initial_net
    best_len = len(current_net)

    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill

    random.seed(42)

    def try_removal(current_net, idx):
        """Try removing comparator at idx. Return True if valid."""
        candidate = current_net[:idx] + current_net[idx+1:]
        return verify(candidate)

    def try_double_removal(current_net, idx1, idx2):
        """Try removing two comparators (idx1 > idx2 to keep order)."""
        if idx1 > idx2:
            a, b = idx2, idx1
        else:
            a, b = idx1, idx2
        candidate = current_net[:a] + current_net[a+1:b] + current_net[b+1:]
        return candidate

    # Strategy: iteratively try to remove comparators, scanning from end to start
    # (later comparators are often redundant)
    while time.time() - start_time < deadline:
        net_len = len(current_net)
        if net_len <= 1:
            break

        removed_this_pass = False

        # Scan from end to start, trying to remove each comparator
        # This is more effective because later comparators are often redundant
        for idx in range(net_len - 1, -1, -1):
            if time.time() - start_time >= deadline:
                break

            if try_removal(current_net, idx):
                current_net = current_net[:idx] + current_net[idx+1:]
                removed_this_pass = True
                if len(current_net) < best_len:
                    best_net = current_net
                    best_len = len(current_net)
                # After a removal, restart scan from end (network changed)
                break

        if removed_this_pass:
            continue

        # No single removal found; try double removals
        net_len = len(current_net)
        found_double = False
        # Try pairs of comparators that are far apart (more likely both redundant)
        # Sample a subset of pairs to save time
        if net_len > 2:
            # Try removing pairs where both are in the latter half of the network
            half_start = net_len // 2
            indices = list(range(half_start, net_len))
            random.shuffle(indices)
            # Also include some from the first half
            indices += list(range(0, half_start))
            random.shuffle(indices)
            indices = indices[:min(40, len(indices))]

            pairs_tried = 0
            for i in range(len(indices)):
                if time.time() - start_time >= deadline:
                    break
                for j in range(i+1, len(indices)):
                    if time.time() - start_time >= deadline:
                        break
                    idx1, idx2 = indices[i], indices[j]
                    candidate = try_double_removal(current_net, idx1, idx2)
                    if candidate is not None and verify(candidate):
                        current_net = candidate
                        found_double = True
                        if len(current_net) < best_len:
                            best_net = current_net
                            best_len = len(current_net)
                        break
                    pairs_tried += 1
                    if pairs_tried > 200:
                        break
                if found_double:
                    break

        if found_double:
            continue

        # No removals found; try swap-then-remove to escape local minimum
        net_len = len(current_net)
        if net_len > 2:
            # Pick two random positions to swap
            i1 = random.randint(0, net_len - 2)
            i2 = random.randint(i1 + 1, net_len - 1)
            candidate = current_net.copy()
            candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
            if verify(candidate):
                # Try removing one comparator from the swapped network
                # Scan from end
                for idx in range(len(candidate) - 1, -1, -1):
                    if time.time() - start_time >= deadline:
                        break
                    if try_removal(candidate, idx):
                        candidate = candidate[:idx] + candidate[idx+1:]
                        current_net = candidate
                        if len(current_net) < best_len:
                            best_net = current_net
                            best_len = len(current_net)
                        break
                else:
                    # No removal, keep the swap
                    current_net = candidate

    return best_net
# EVOLVE-BLOCK-END
