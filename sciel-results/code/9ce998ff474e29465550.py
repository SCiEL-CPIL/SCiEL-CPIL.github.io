import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    n = 17
    min_dist = 6
    full_space = 1 << n  # 131072

    # Precompute popcount for all 17-bit values
    popcount_np = np.array([bin(i).count('1') for i in range(full_space)], dtype=np.uint8)

    # --- Construct the 256-word [17,8,6] QR code (shortened [18,9,6]) ---
    # The [18,9,6] quadratic residue code over GF(2) has generator polynomial
    # g(x) = x^9 + x^7 + x^6 + x^3 + 1 (one of the two QR codes of length 18)
    # We shorten by fixing coordinate 17 (the last bit) to 0.
    # Actually, let's use the standard construction.
    # The [18,9,6] QR code: generator polynomial g(x) = x^9 + x^7 + x^6 + x^3 + 1
    # This gives 2^9 = 512 codewords of length 18.
    # Shortening: set the last coordinate to 0, giving [17,8,6] with 256 codewords.
    # Wait, shortening a [18,9,6] by one coordinate gives [17,8,6] only if the
    # coordinate is not in all nonzero codewords... Actually, shortening reduces
    # dimension by 1 if the punctured coordinate is not identically zero.
    # Let me just enumerate: for each message m in 0..255, compute the 18-bit codeword,
    # keep only those with last bit = 0, then drop the last bit.

    gen_poly = (1 << 9) | (1 << 7) | (1 << 6) | (1 << 3) | (1 << 0)
    # g(x) = x^9 + x^7 + x^6 + x^3 + 1

    def encode_18(msg):
        """Encode a 9-bit message into an 18-bit codeword."""
        cw = 0
        for bit_pos in range(9):
            if (msg >> bit_pos) & 1:
                cw ^= (gen_poly << bit_pos)
        return cw

    qr_codewords = []
    seen = set()
    for msg in range(512):
        cw18 = encode_18(msg)
        # Check last bit (bit 17)
        if (cw18 >> 17) & 1 == 0:
            cw17 = cw18 & ((1 << 17) - 1)  # drop last bit
            if cw17 not in seen:
                qr_codewords.append(cw17)
                seen.add(cw17)

    # Verify: should have 256 codewords
    if len(qr_codewords) != 256:
        # Fallback: use a different generator or just use greedy from scratch
        # Let's try the other QR code generator: x^9 + x^8 + x^5 + x^2 + 1
        gen_poly2 = (1 << 9) | (1 << 8) | (1 << 5) | (1 << 2) | (1 << 0)
        qr_codewords = []
        seen = set()
        for msg in range(512):
            cw18 = 0
            for bit_pos in range(9):
                if (msg >> bit_pos) & 1:
                    cw18 ^= (gen_poly2 << bit_pos)
            if (cw18 >> 17) & 1 == 0:
                cw17 = cw18 & ((1 << 17) - 1)
                if cw17 not in seen:
                    qr_codewords.append(cw17)
                    seen.add(cw17)

    if len(qr_codewords) == 256:
        # Verify minimum distance
        qr_np = np.array(qr_codewords, dtype=np.uint32)
        xor_matrix = qr_np[:, None] ^ qr_np[None, :]
        dists = popcount_np[xor_matrix]
        np.fill_diagonal(dists, 100)
        if dists.min() < min_dist:
            # Invalid, fallback to greedy
            qr_codewords = [0]
    else:
        # Fallback: greedy from 0
        qr_codewords = [0]

    C_list = list(qr_codewords)
    C_np = np.array(C_list, dtype=np.uint32)
    C_set = set(C_list)
    current_size = len(C_list)

    # --- Greedy expansion with vectorized distance checks ---
    # Strategy: repeatedly try to add a new word that is at distance >= 6 from all existing words.
    # Use a "maximin" heuristic: among valid candidates, pick the one with maximum minimum distance.

    # Pre-generate a shuffled order of all 17-bit words for systematic search
    # But 131072 is manageable; let's use random batches for speed.

    # Phase 1: Add words one at a time using random batches
    # Each iteration: sample a batch, find valid ones, pick the best.
    # To be more efficient, we can also try to add multiple words at once if they're mutually far.

    # Let's use a smarter approach: maintain a "distance map" and use it to guide search.
    # For each word w in the space, compute min_dist_to_C(w).
    # This is expensive to update, but we can approximate.

    # Simpler: just do random batch greedy.
    # With 256 codewords, checking a batch of B candidates costs B*256 XOR+popcount.
    # NumPy vectorizes this well.

    # Let's also try a "repair" phase: if we get stuck, remove a word and try to add two.
    # But that's complex. Let's stick with greedy first.

    # Use a priority-queue-like approach: always try to add the word that maximizes min distance.
    # But computing min distance for all 131072 words against C is 131072*|C| ops.
    # With |C|=256, that's ~33M ops. NumPy can do this in ~100ms.
    # We can afford to do this a few times.

    # Phase 1: Full-space scan to find all valid words, then greedily add the best ones.
    # This is O(full_space * |C|) per full scan.

    # Let's do iterative full-space scans.
    # In each scan, compute min_dist for all words not in C.
    # Pick the word with max min_dist that is >= 6.
    # Add it.
    # Repeat until no more valid words or time runs out.

    # But full-space scan each time is expensive as |C| grows.
    # Alternative: maintain a candidate pool and update incrementally.

    # Let's do a hybrid:
    # 1. Initial full scan to find all words with min_dist >= 6 to QR code.
    # 2. From that pool, greedily add the best ones (maximin).
    # 3. After each addition, update the pool by removing words that conflict with the new word.

    # Step 1: Find all words valid against the QR code.
    all_words = np.arange(full_space, dtype=np.uint32)
    # Exclude existing codewords
    mask_not_in_C = np.ones(full_space, dtype=bool)
    for cw in C_list:
        mask_not_in_C[cw] = False

    # Compute min distance from each word to C
    # This is full_space x |C| matrix of XORs -> popcounts -> min along axis 1
    # Memory: 131072 * 256 * 1 byte = ~32 MB. OK.
    xor_full = all_words[:, None] ^ C_np[None, :]
    dist_full = popcount_np[xor_full]
    min_dists_full = dist_full.min(axis=1)

    # Valid candidates: not in C and min_dist >= 6
    valid_mask = mask_not_in_C & (min_dists_full >= min_dist)
    valid_candidates = np.where(valid_mask)[0]

    # Phase 2: Greedily add from valid_candidates using maximin
    # We'll maintain the valid_candidates list and min_dists for each.
    # When we add a word, we need to update min_dists for all remaining candidates.
    # This is O(|valid| * 1) per addition if we just check distance to the new word.

    # Let's do it iteratively:
    # While there are valid candidates and time remains:
    #   Pick the candidate with max min_dist
    #   Add it to C
    #   For all remaining valid candidates, update min_dist = min(old_min_dist, dist_to_new_word)
    #   Remove candidates whose min_dist < 6

    # To make this efficient, keep valid_candidates as a list and min_dists as an array.
    # But updating min_dists for all candidates each time is O(|valid|).
    # With |valid| ~ 10000 and ~84 additions, that's ~840k ops. Very fast.

    # Let's implement this.
    valid_list = list(valid_candidates)
    min_dists_valid = min_dists_full[valid_mask].copy()

    # Sort by min_dist descending for initial ordering (optional, but helps with maximin)
    # Actually, we'll just pick the max each time.
    # To speed up, we can use a max-heap, but Python's heapq is min-heap.
    # For simplicity, just scan for max each time. |valid| ~ 10000, 84 scans = 840k ops.

    while valid_list and time.time() < deadline - 5.0:
        # Find the candidate with max min_dist
        max_idx = 0
        max_val = min_dists_valid[0]
        for i in range(1, len(valid_list)):
            if min_dists_valid[i] > max_val:
                max_val = min_dists_valid[i]
                max_idx = i

        new_word = valid_list[max_idx]
        C_list.append(new_word)
        C_np = np.append(C_np, new_word)
        C_set.add(new_word)
        current_size += 1

        # Update min_dists for remaining candidates
        # dist to new_word
        dists_to_new = popcount_np[np.array(valid_list, dtype=np.uint32) ^ new_word]
        # Update: min_dists_valid = min(min_dists_valid, dists_to_new)
        min_dists_valid = np.minimum(min_dists_valid, dists_to_new)

        # Remove candidates with min_dist < 6
        keep_mask = min_dists_valid >= min_dist
        valid_list = [valid_list[i] for i in range(len(valid_list)) if keep_mask[i]]
        min_dists_valid = min_dists_valid[keep_mask]

        if current_size >= 340:
            break

    # Phase 3: If we have time, try random restarts or local search to find more.
    # But the above greedy should find most of the addable words.
    # Let's try a few random batches to see if we missed any.
    if time.time() < deadline - 3.0:
        # Try adding more via random batches
        C_np = np.array(C_list, dtype=np.uint32)
        for _ in range(50):  # up to 50 more iterations
            if time.time() > deadline - 3.0:
                break
            batch_size = 10000
            batch = np.random.randint(0, full_space, size=batch_size, dtype=np.uint32)
            # Filter out existing
            # Compute min dist to C
            xor_batch = batch[:, None] ^ C_np[None, :]
            dists_batch = popcount_np[xor_batch]
            min_dists_batch = dists_batch.min(axis=1)
            valid_batch_mask = min_dists_batch >= min_dist
            # Also check not in C (distance 0 would be caught, but let's be safe)
            valid_batch_indices = np.where(valid_batch_mask)[0]
            if len(valid_batch_indices) == 0:
                continue
            # Pick the best
            best_idx = valid_batch_indices[np.argmax(min_dists_batch[valid_batch_indices])]
            new_word = batch[best_idx]
            if new_word not in C_set:
                C_list.append(new_word)
                C_np = np.append(C_np, new_word)
                C_set.add(new_word)
                current_size += 1
                if current_size >= 340:
                    break

    return C_list
