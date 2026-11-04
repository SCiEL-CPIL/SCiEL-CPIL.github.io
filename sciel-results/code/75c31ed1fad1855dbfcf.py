<<<<<<< SEARCH
    def time_ok():
        return time.time() < deadline

    def popcount_u64(arr):
        return int(np.sum(arr.view(np.uint64)))  # arr is uint64 already

    # --- Greedy construction ---
    blocks = []
    while total_uncov > 0 and time_ok():
        # find the first uncovered t-subset to seed the block
        seed_idx = None
        for w in range(words):
            if uncov[w]:
                # find lowest set bit
                word = uncov[w]
                bit = (word & -word).bit_length() - 1
                seed_idx = (w << 6) + bit
                break
        if seed_idx is None:
            break
=======
    def time_ok():
        return time.time() < deadline

    def popcount_u64(arr):
        # arr is np.uint64 array. We need to count set bits.
        # np.count_nonzero doesn't work on bits.
        # We can use a lookup table or bit manipulation.
        # For simplicity and speed in Python, we can convert to list and sum bit counts?
        # No, that's slow.
        # Let's use a helper that processes chunks or just use a simple loop if words are small.
        # v=19, t=5 -> C(19,5) = 11628 bits. words = 183.
        # Looping 183 times is fine.
        cnt = 0
        for w in arr:
            cnt += bin(int(w)).count('1')
        return cnt

    # --- Greedy construction ---
    blocks = []
    while total_uncov > 0 and time_ok():
        # find the first uncovered t-subset to seed the block
        seed_idx = None
        for w in range(words):
            if uncov[w]:
                # find lowest set bit
                word = int(uncov[w])
                bit = (word & -word).bit_length() - 1
                seed_idx = (w << 6) + bit
                break
        if seed_idx is None:
            break
>>>>>>> REPLACE

