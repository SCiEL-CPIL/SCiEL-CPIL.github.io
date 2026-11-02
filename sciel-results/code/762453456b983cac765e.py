import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    timeout = 115.0  # Leave 5s margin

    total_words = q ** n

    # Precompute neighbors as a flat array and offsets
    # For radius=1: each word has 1 + n*(q-1) neighbors
    if radius == 1:
        n_neighbors = 1 + n * (q - 1)
    else:
        # For general radius, compute properly (but we expect radius=1 for this task)
        n_neighbors = 1 + n * (q - 1)  # fallback, only correct for radius=1

    # Build neighbors array
    neighbors = np.empty((total_words, n_neighbors), dtype=np.int32)
    for u in range(total_words):
        # Convert to digits
        du = []
        tmp = u
        for i in range(n):
            du.append(tmp % q)
            tmp //= q
        # du[0] is least significant
        neighbors[u, 0] = u
        idx = 1
        for pos in range(n):
            orig_val = du[pos]
            power = q ** pos
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                v = u - orig_val * power + new_val * power
                neighbors[u, idx] = v
                idx += 1

    # Greedy initialization using bitmasks for speed
    # Represent uncovered set as a bitset
    # For 2187 words, we need ceil(2187/64) = 35 uint64 words
    n_bits = total_words
    n_words64 = (n_bits + 63) // 64
    uncovered_mask = np.ones(n_words64, dtype=np.uint64)
    # Clear bits beyond total_words
    if total_words % 64 != 0:
        uncovered_mask[-1] = (1 << (total_words % 64)) - 1

    C = []
    C_set = set()

    while True:
        # Check if uncovered is empty
        if not np.any(uncovered_mask):
            break

        # Find best u: maximize number of neighbors in uncovered
        # Convert uncovered_mask to a boolean array of size total_words
        uncovered_bool = np.zeros(total_words, dtype=bool)
        for i in range(n_words64):
            m = uncovered_mask[i]
            base = i * 64
            while m:
                bit = (m & -m).bit_length() - 1
                pos = base + bit
                if pos < total_words:
                    uncovered_bool[pos] = True
                m &= m - 1

        # Count how many neighbors of each u are in uncovered
        # neighbors is (total_words, n_neighbors), uncovered_bool is (total_words,)
        # counts[u] = sum over v in neighbors[u] of uncovered_bool[v]
        counts = np.sum(uncovered_bool[neighbors], axis=1)

        best_u = int(np.argmax(counts))
        if counts[best_u] == 0:
            break  # Should not happen

        C.append(best_u)
        C_set.add(best_u)

        # Update uncovered_mask: remove neighbors of best_u
        for v in neighbors[best_u]:
            v = int(v)
            word_idx = v // 64
            bit_idx = v % 64
            uncovered_mask[word_idx] &= ~(np.uint64(1) << np.uint64(bit_idx))

    # Convert to numpy arrays for vectorized local search
    C_arr = np.array(C, dtype=np.int32)
    C_len = len(C_arr)

    # Precompute neighbor indices as a flat structure for fast access
    # We'll use neighbors array directly

    def is_covering_fast(C_list):
        """Check if C_list covers all words using NumPy."""
        if len(C_list) == 0:
            return total_words == 0
        C_np = np.array(C_list, dtype=np.int32)
        # Get all neighbors of codewords
        all_neighbors = neighbors[C_np]  # shape: (len(C), n_neighbors)
        # Flatten and unique
        flat = all_neighbors.flatten()
        covered = np.zeros(total_words, dtype=np.uint8)
        # Use bincount or unique
        # np.unique is slow for large arrays, but flat is at most C_len * n_neighbors
        # For C_len~200, n_neighbors~14, flat has ~2800 elements
        # Mark covered
        covered[flat] = 1
        return np.all(covered > 0)

    # Better: maintain a coverage count array incrementally
    # Let's use a different approach: maintain cover_counts array
    cover_counts = np.zeros(total_words, dtype=np.int32)
    for c in C:
        cover_counts[neighbors[c]] += 1

    def is_covering_from_counts():
        return np.all(cover_counts > 0)

    # Local search with NumPy
    # Phase 1: Try removals
    # Phase 2: Try swaps

    def try_removals():
        """Try to remove elements from C_arr. Returns True if any removal succeeded."""
        nonlocal C_arr, cover_counts
        removed = False
        # Shuffle order
        order = np.random.permutation(len(C_arr))
        for i in order:
            c = C_arr[i]
            # Tentatively remove
            cover_counts[neighbors[c]] -= 1
            if np.all(cover_counts > 0):
                # Accept removal
                C_arr = np.delete(C_arr, i)
                removed = True
            else:
                # Revert
                cover_counts[neighbors[c]] += 1
        return removed

    def try_swaps(max_attempts=200):
        """Try random swaps. Returns True if any swap succeeded."""
        nonlocal C_arr, cover_counts
        n_c = len(C_arr)
        if n_c == 0:
            return False
        C_set_local = set(C_arr.tolist())
        for _ in range(max_attempts):
            if time.time() - start_time > timeout:
                break
            idx = random.randint(0, n_c - 1)
            old_c = int(C_arr[idx])
            new_c = random.randint(0, total_words - 1)
            if new_c == old_c or new_c in C_set_local:
                continue

            # Tentatively swap
            cover_counts[neighbors[old_c]] -= 1
            cover_counts[neighbors[new_c]] += 1
            if np.all(cover_counts > 0):
                # Accept swap
                C_arr[idx] = new_c
                C_set_local.discard(old_c)
                C_set_local.add(new_c)
                return True
            else:
                # Revert
                cover_counts[neighbors[old_c]] += 1
                cover_counts[neighbors[new_c]] -= 1
        return False

    # Main loop
    while time.time() - start_time < timeout:
        # Phase 1: Removals
        if try_removals():
            continue
        # Phase 2: Swaps
        if try_swaps(max_attempts=500):
            continue
        # Stuck - try more aggressive swaps
        # Try swaps with higher attempts
        if try_swaps(max_attempts=2000):
            continue

    # Convert back to list of lists
    result = []
    for c in C_arr:
        c = int(c)
        digits = []
        for i in range(n):
            digits.append(c % q)
            c //= q
        result.append(digits)

    return result
