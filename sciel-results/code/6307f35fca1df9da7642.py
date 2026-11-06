import time
import numpy as np

N = 18
TIME_LIMIT = 52  # Safety margin: 90s wall clock, evaluator kills at 90s


def _odd_even_transposition():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _bit_reverse(x, bits):
    y = 0
    for i in range(bits):
        y = (y << 1) | (x & 1)
        x >>= 1
    return y


def _batcher_odd_even_split(n):
    """Generate a sorting network for n inputs using recursive odd-even mergesort."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    if n % 2 != 0:
        # Handle odd: sort n-1, then merge
        # This gets complex. For our use n is always a power of 2 or we handle it.
        # Actually let's just do standard odd-even merge sort which requires even n for the split.
        # We can pad or just assume even for the recursion steps?
        # Standard OEMS:
        # 1. Sort odd and even inputs in parallel (n/2 wires each)
        # 2. Odd-even merge the two sorted halves
        
        # Let's implement a generic OEMS that works for any n?
        # It's easier to just implement the standard even-n case and for odd n,
        # we can do: sort n, then...
        # Actually, let's just use the known Batcher network generation for powers of 2,
        # and for 18, we can use a hybrid or just stick to the OETS seed if this is complex.
        # However, Batcher networks are significantly smaller than OETS.
        # OETS for 18 is 18*18/2 = 162? No, 153.
        # Batcher for 18 is around 100-120?
        # Let's implement Batcher for even n. For odd n, we can add a dummy wire or just sort n-1 and merge.
        pass
        
    # Standard Odd-Even Mergesort
    comparators = []
    
    def odd_even_merge(a_size, b_size, a_offset, b_offset):
        # Merge two sorted arrays of size a_size and b_size
        # They are interleaved? No, in Batcher they are separate blocks.
        # Actually, the standard formulation interleaves them.
        # Let's use the simpler "recursive split" definition.
        pass
    
    # This is getting complicated to get right quickly.
    # Let's stick to the OETS seed but use a better initialization for the search.
    # The OETS seed is 153 comparators.
    # The search should reduce this significantly.
    # Let's just return OETS to be safe, or a known small network if I can remember one?
    # No, let's just use OETS. The search is the key.
    
    # Wait, I can generate a better starting point.
    # A simple "bitonic" style or just OETS is fine.
    # Let's stick to OETS for simplicity and reliability of the seed.
    return _odd_even_transposition()


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    num_blocks = 2**N // 64
    states = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
    # Fill states: bit-parallel representation of all 2^N inputs
    # For each input k (0..2^N-1), wire w has value (k>>w)&1
    # We pack 64 inputs per uint64 block.
    # states[w][block_idx] has bit (k%64) set if ((k>>w)&1)==1
    # k = block_idx * 64 + bit_idx
    
    # Optimization: Precompute the pattern for each wire w and each bit position?
    # Or just loop. 2^18 * 18 = ~4.7M operations. ~0.2s in Python? Maybe.
    # Let's vectorize the initialization.
    
    # For wire w, the bit is (k >> w) & 1.
    # k ranges from 0 to 2^N - 1.
    # We can create an array of all k's? 262k ints is fine.
    # But we need to pack them.
    
    # Let's try a faster init using numpy broadcasting on blocks?
    # Block b contains inputs k in [b*64, (b+1)*64).
    # For a fixed block b, the inputs are b*64 + i for i in 0..63.
    # Value for wire w is ((b*64 + i) >> w) & 1.
    
    # If w < 6, (b*64 + i) >> w = (b*64 >> w) + (i >> w) (if no carry? 64 is 2^6).
    # 64 * b = b * 2^6.
    # So (b * 2^6 + i) >> w.
    # If w < 6, this is (b * 2^(6-w) + (i >> w)) ... wait, no.
    # b * 2^6 >> w = b * 2^(6-w).
    # i >> w.
    # The bit 0 of this sum?
    # (b * 2^(6-w) + (i >> w)) & 1.
    # Since w < 6, 6-w >= 1. So b * 2^(6-w) is even.
    # So (b * 2^(6-w) + (i >> w)) & 1 == (i >> w) & 1.
    # So for w < 6, the pattern in each block is identical! It depends only on i.
    # Mask for wire w (w<6) is the same for all blocks.
    
    # If w >= 6, (b * 2^6 + i) >> w = b * 2^(6-w) + (i >> w).
    # Wait, if w=6, (b*64 + i) >> 6 = b + (i>>6) = b (since i<64).
    # So bit 0 is b & 1.
    # So for w=6, the mask alternates with block index b.
    
    # If w > 6, (b*64 + i) >> w.
    # Since i < 64, i >> w is 0.
    # So it is (b * 64) >> w = b >> (w-6).
    # So the value is constant across the 64 bits in the block!
    # It is either all 0s or all 1s.
    # Value is (b >> (w-6)) & 1.
    
    # This is a huge optimization for initialization.
    
    # Precompute masks for w < 6 (same for all blocks)
    base_masks = [0] * N
    for w in range(6):
        m = 0
        for i in range(64):
            if (i >> w) & 1:
                m |= (1 << i)
        base_masks[w] = m
    
    # For each block b and wire w:
    for b in range(num_blocks):
        for w in range(N):
            if w < 6:
                states[w][b] = base_masks[w]
            elif w == 6:
                if b & 1:
                    states[w][b] = (1 << 64) - 1
                else:
                    states[w][b] = 0
            else: # w > 6
                if (b >> (w - 6)) & 1:
                    states[w][b] = (1 << 64) - 1
                else:
                    states[w][b] = 0

    def _fast_verify(comparators):
        current = [s.copy() for s in states]
        for i, j in comparators:
            a = current[i]
            b = current[j]
            current[i] = a & b
            current[j] = a | b
        
        # Check sorted: for all w, wire w <= wire w+1
        # i.e., for all inputs, bit w is 0 whenever bit w+1 is 0? No.
        # Sorted means 0s then 1s.
        # Wire w contains 0s for the smallest inputs and 1s for the largest?
        # No, compare-exchange puts min on i and max on j.
        # So wire 0 is the min, wire N-1 is the max.
        # For a sorted sequence, if wire w is 1, then wire w+1 must be 1.
        # Because 1s are at the high end.
        # So we need: ~current[w] | current[w+1] is all 1s?
        # i.e., current[w] & ~current[w+1] is 0.
        for w in range(N - 1):
            if np.any(current[w] & ~current[w + 1]):
                return False
        return True

    best_network = _odd_even_transposition()
    best_len = len(best_network)
    
    if not _fast_verify(best_network):
        # Should not happen for OETS, but safe fallback
        return best_network

    current_network = list(best_network)
    
    def greedy_reduce(net):
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            idx = 0
            while idx < len(net):
                if time.time() > deadline - 3:
                    return net
                trial = net[:idx] + net[idx+1:]
                if _fast_verify(trial):
                    net = trial
                    changed = True
                else:
                    idx += 1
        return net

    current_network = greedy_reduce(current_network)
    if len(current_network) < best_len:
        best_network = current_network
        best_len = len(current_network)

    # Local Search
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 5:
        # Move 1: Swap adjacent
        if len(current_network) > 1:
            idx = rng.integers(len(current_network) - 1)
            trial = current_network.copy()
            trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
            if _fast_verify(trial):
                current_network = trial
                # Try to reduce after swap
                reduced = greedy_reduce(current_network)
                if len(reduced) < len(current_network):
                    current_network = reduced
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)

        # Move 2: Mutation (change wire pair)
        if len(current_network) > 0 and time.time() < deadline - 5:
            idx = rng.integers(len(current_network))
            i, j = current_network[idx]
            # Pick new pair
            a, b = rng.choice(N, 2, replace=False)
            if a > b: a, b = b, a
            if a == i and b == j: continue
            trial = current_network.copy()
            trial[idx] = (a, b)
            if _fast_verify(trial):
                current_network = trial
                reduced = greedy_reduce(current_network)
                if len(reduced) < len(current_network):
                    current_network = reduced
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)
                    
        # Move 3: Insert-Delete (2-opt style)
        # Insert a random comparator at a random position, then try to remove
        if len(current_network) < 200 and time.time() < deadline - 5:
             idx = rng.integers(len(current_network) + 1)
             a, b = rng.choice(N, 2, replace=False)
             if a > b: a, b = b, a
             trial = current_network[:idx] + [(a,b)] + current_network[idx:]
             # Don't verify trial directly, just try to reduce it
             # If it reduces to something shorter than current, it's a good move
             reduced = greedy_reduce(trial)
             if len(reduced) < len(current_network):
                 current_network = reduced
                 if len(current_network) < best_len:
                     best_network = list(current_network)
                     best_len = len(current_network)
             # If it didn't reduce enough, we might still want to keep the current network
             # The greedy_reduce on trial might have removed the inserted one and some others,
             # but if the result is not shorter, we discard.

    return best_network
