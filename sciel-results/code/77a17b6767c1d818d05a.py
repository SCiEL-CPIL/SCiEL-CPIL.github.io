"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
import itertools

N = 16

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85

    # Precompute all 65536 boolean inputs as a bit-packed array
    # Each input is a 16-bit integer; we store them in a numpy array of uint64
    all_inputs = np.array(list(itertools.product([0, 1], repeat=N)), dtype=np.uint8)
    # Pack into uint64: each row becomes one uint64 with 16 bits
    # Actually, let's use a different approach: store as uint64 array of shape (65536,)
    # where each element has the 16 bits packed
    packed_inputs = np.zeros(1 << N, dtype=np.uint64)
    for k in range(1 << N):
        val = 0
        for w in range(N):
            if all_inputs[k, w]:
                val |= (1 << w)
        packed_inputs[k] = val

    def apply_comparator_packed(inputs, i, j):
        """Apply compare-exchange (i,j) to all inputs in parallel using bit ops.
        For each input, wire i gets min, wire j gets max.
        min bit at position i: bit_i AND bit_j
        max bit at position j: bit_i OR bit_j
        """
        mask_i = np.uint64(1) << i
        mask_j = np.uint64(1) << j
        bits_i = (inputs & mask_i) != 0
        bits_j = (inputs & mask_j) != 0
        # min goes to i: both must be 1
        min_bit = bits_i & bits_j
        # max goes to j: either is 1
        max_bit = bits_i | bits_j
        # Clear bits i and j, then set them
        result = inputs & ~(mask_i | mask_j)
        result = result | np.where(min_bit, mask_i, np.uint64(0))
        result = result | np.where(max_bit, mask_j, np.uint64(0))
        return result

    def verify_packed(network):
        """Verify network sorts all inputs using bit-parallel simulation."""
        x = packed_inputs.copy()
        for i, j in network:
            x = apply_comparator_packed(x, i, j)
        # Check sorted: for each input, bits should be non-decreasing
        # i.e., if bit k is 0 and bit l > k is 1, that's fine
        # Actually sorted means: all 1s come before all 0s (ascending)
        # So for each input, the set of 1-bits should be a prefix {0,1,...,m-1}
        # Check: for each wire w from 1 to N-1, if bit w is 1 then bit w-1 must be 1
        # Equivalently: the value should equal 2^m - 1 for some m
        # Simpler: check that for all w, bit[w] <= bit[w-1] ... no
        # Ascending sort: smallest values first. So bit 0 should be 0 if any 0 exists.
        # Actually: sorted ascending means 0s then 1s. So bit w = 0 for w < count_ones... 
        # No: ascending means wire 0 has smallest value. So if there are k ones,
        # wires 0..N-k-1 have 0 and wires N-k..N-1 have 1.
        # Check: for each w from 1 to N-1: if bit[w-1] == 1 then bit[w] == 1
        # i.e., no 1 followed by 0.
        for w in range(1, N):
            mask_w = np.uint64(1) << w
            mask_wm1 = np.uint64(1) << (w - 1)
            bit_w = (x & mask_w) != 0
            bit_wm1 = (x & mask_wm1) != 0
            # If bit_wm1 is 1, bit_w must be 1
            if np.any(bit_wm1 & ~bit_w):
                return False
        return True

    def verify_slow(network):
        """Slow verification as fallback."""
        x = all_inputs.copy()
        for i, j in network:
            x[i], x[j] = np.minimum(x[i], x[j]), np.maximum(x[i], x[j])
        return np.all(np.diff(x, axis=0) >= 0, axis=0).all()

    # Use the fast packed verification
    verify = verify_packed

    def odd_even_merge(n):
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        result = []
        for i in range(n // 2):
            result.append((i, i + n // 2))
        for (i, j) in odd_even_merge(n // 2):
            result.append((2 * i + 1, 2 * j + 1))
        for (i, j) in odd_even_merge(n // 2):
            result.append((2 * i, 2 * j))
        return result

    def batcher_oe(n):
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        result = []
        result.extend(batcher_oe(n // 2))
        result.extend([(i + n // 2, j + n // 2) for (i, j) in batcher_oe(n // 2)])
        result.extend(odd_even_merge(n))
        return result

    def odd_even_transposition():
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

    # Start with Batcher's odd-even merge (119 comparators)
    best = batcher_oe(N)
    if not verify(best):
        best = odd_even_transposition()

    net = list(best)

    # Aggressive deletion search with fast verification
    max_rounds = 500
    for round_num in range(max_rounds):
        if time.time() > deadline - 2:
            break
        if len(net) <= 60:
            break

        improved = False
        # Shuffle indices for randomness
        indices = list(range(len(net)))
        random.shuffle(indices)

        for idx in indices:
            if time.time() > deadline - 2:
                break
            candidate = net[:idx] + net[idx+1:]
            if verify(candidate):
                net = candidate
                improved = True
                if len(net) < len(best):
                    best = list(net)
                # Continue trying to delete more
                indices = list(range(len(net)))
                random.shuffle(indices)
                # Don't break; keep going

        if not improved:
            # Try a different strategy: swap two comparators or move one
            # Try moving a comparator to a different position
            moved = False
            for _ in range(20):
                if time.time() > deadline - 2:
                    break
                if len(net) < 2:
                    break
                a, b = random.sample(range(len(net)), 2)
                # Swap positions
                candidate = list(net)
                candidate[a], candidate[b] = candidate[b], candidate[a]
                if verify(candidate):
                    net = candidate
                    moved = True
                    if len(net) < len(best):
                        best = list(net)
                    break
                # Try moving a to position b
                candidate = list(net)
                comp = candidate.pop(a)
                candidate.insert(b, comp)
                if verify(candidate):
                    net = candidate
                    moved = True
                    if len(net) < len(best):
                        best = list(net)
                    break

            if not moved:
                # Try adding a random comparator then deleting
                for _ in range(10):
                    if time.time() > deadline - 2:
                        break
                    i, j = random.sample(range(N), 2)
                    if i > j:
                        i, j = j, i
                    candidate = list(net) + [(i, j)]
                    if verify(candidate):
                        # Now try to delete from this larger network
                        net = candidate
                        for idx in range(len(net)):
                            if time.time() > deadline - 2:
                                break
                            cand2 = net[:idx] + net[idx+1:]
                            if verify(cand2):
                                net = cand2
                                if len(net) < len(best):
                                    best = list(net)
                                break
                        break

    return best
# EVOLVE-BLOCK-END
