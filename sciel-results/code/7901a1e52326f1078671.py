import time
import random
import numpy as np
import itertools


def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1()
    return _fallback_construct(q, n, radius)


def _fallback_construct(q, n, radius):
    all_words = list(itertools.product(range(q), repeat=n))
    if radius != 1:
        return [list(w) for w in all_words]
    total = len(all_words)
    covered = [False] * total
    codewords = []
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    covers = {}
    for i, w in enumerate(all_words):
        w_list = list(w)
        nbrs = [i]
        for pos in range(n):
            orig = w_list[pos]
            for val in range(q):
                if val != orig:
                    w_list[pos] = val
                    nbrs.append(word_to_idx[tuple(w_list)])
            w_list[pos] = orig
        covers[i] = set(nbrs)
    while not all(covered):
        best_idx = -1
        best_count = -1
        for i in range(total):
            cnt = sum(1 for j in covers[i] if not covered[j])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_idx == -1:
            break
        codewords.append(list(all_words[best_idx]))
        for j in covers[best_idx]:
            covered[j] = True
    return codewords


def _construct_ternary_8_1():
    q = 3
    n = 8
    radius = 1
    total = q ** n  # 6561

    start_time = time.time()
    deadline = start_time + 130  # safety margin for 150s limit

    # Precompute neighbor indices for all 6561 words using vectorized approach
    # Each word idx -> base-3 representation
    # Neighbors: self + all single-position changes (17 total)
    all_indices = np.arange(total, dtype=np.int64)
    
    # Compute digit values for all positions
    # digits[i, pos] = value at position pos for word i
    digits = np.zeros((total, n), dtype=np.int64)
    temp = all_indices.copy()
    for pos in range(n - 1, -1, -1):
        digits[:, pos] = temp % q
        temp //= q

    # Precompute neighbor list for each word as a flat array
    # Each word has 17 neighbors
    num_nbrs = 1 + n * (q - 1)  # 17
    neighbors = np.zeros((total, num_nbrs), dtype=np.int64)
    neighbors[:, 0] = all_indices  # self
    
    nbr_offset = 1
    for pos in range(n):
        for val in range(q):
            # For each word, change position pos to val (if different)
            # We'll compute all and filter later, but easier to build directly
            pass
    
    # Build neighbors more carefully
    neighbors = np.zeros((total, num_nbrs), dtype=np.int64)
    neighbors[:, 0] = all_indices
    nbr_col = 1
    for pos in range(n):
        orig_digits = digits[:, pos]
        for val in range(q):
            if val == 0:
                continue  # skip if val equals original? No, we need all changes
            # Actually we need to handle: for each word, for each pos, for each val != orig
            # Let's build it differently
            pass

    # Simpler: build neighbors in a loop but vectorize the index computation
    neighbors = np.zeros((total, num_nbrs), dtype=np.int64)
    neighbors[:, 0] = all_indices
    nbr_col = 1
    for pos in range(n):
        orig = digits[:, pos]
        for val in range(q):
            mask = orig != val
            if not np.any(mask):
                continue
            # Compute new index for words where digit at pos changes to val
            # new_idx = idx + (val - orig) * q^pos
            power = q ** pos
            new_idx = all_indices + (val - orig) * power
            neighbors[mask, nbr_col] = new_idx[mask]
            nbr_col += 1

    # Verify: nbr_col should equal num_nbrs
    # Each pos has (q-1) = 2 changes, so 8*2 = 16, plus self = 17. Correct.

    # Build a set-based lookup for fast membership (for swap search)
    # But for vectorized approach, we'll use cover_count array

    # Initial code: product of tetracode (9 words) x all 4-digit tails (81 words) = 729
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=4):
            word = h + list(tail)
            idx = 0
            for v in word:
                idx = idx * q + v
            initial_code.append(idx)
    
    code = np.array(initial_code, dtype=np.int64)
    code_set = set(code.tolist())

    # Cover count: for each of 6561 words, how many codewords are within distance 1
    cover_count = np.zeros(total, dtype=np.int64)
    for c in code:
        cover_count[neighbors[c]] += 1

    best_size = len(code)
    best_code = code.copy()

    def is_time_up():
        return time.time() > deadline - 5

    # Local search: alternating prune and kick phases
    # Prune: remove any codeword that is redundant (all its neighbors have cover_count > 1)
    # Kick: remove a random codeword, find a replacement that covers all newly-uncovered words

    iteration = 0
    while not is_time_up():
        iteration += 1

        # --- PRUNE PHASE ---
        # Try to remove redundant codewords
        # A codeword c is removable if all neighbors of c have cover_count >= 2
        # (since removing c decreases cover_count by 1 for each neighbor)
        changed = True
        while changed and not is_time_up():
            changed = False
            # For each codeword, check if all its neighbors have cover_count >= 2
            # Vectorized: for each codeword c, min(cover_count[neighbors[c]]) >= 2
            # neighbors[c] has shape (len(code), 17)
            if len(code) == 0:
                break
            nbrs_of_code = neighbors[code]  # shape (len(code), 17)
            cover_of_nbrs = cover_count[nbrs_of_code]  # shape (len(code), 17)
            min_cover = cover_of_nbrs.min(axis=1)  # shape (len(code),)
            removable_mask = min_cover >= 2
            
            if not np.any(removable_mask):
                break
            
            # Remove all removable codewords at once (but be careful: removing one affects others)
            # To be safe, remove one at a time, but that's slow. 
            # Alternative: remove all that are removable given current cover_count,
            # then recompute. This is valid because if c is removable (all nbrs have cover>=2),
            # and d is also removable, removing both is fine as long as no word becomes uncovered.
            # Actually, removing c decreases cover_count for nbrs(c). If d is in nbrs(c) and d was removable,
            # after removing c, d's cover_count decreases, so d might no longer be removable.
            # So we must be careful. Let's remove one at a time but in a smart order.
            
            # For efficiency, let's just remove one at a time but break early if time is tight
            removable_indices = np.where(removable_mask)[0]
            random.shuffle(removable_indices.tolist())
            
            for ri in removable_indices:
                if is_time_up():
                    break
                c = code[ri]
                # Double-check: all neighbors of c have cover_count >= 2
                if np.any(cover_count[neighbors[c]] < 2):
                    continue
                # Remove c
                cover_count[neighbors[c]] -= 1
                code = np.delete(code, ri)
                code_set.discard(c)
                changed = True
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code.copy()
        
        # --- KICK PHASE ---
        # Try to swap a codeword to enable further pruning
        if len(code) > 450 and not is_time_up():
            # Pick a random codeword to remove
            idx_c = random.randint(0, len(code) - 1)
            c = code[idx_c]
            
            # Temporarily remove
            cover_count[neighbors[c]] -= 1
            
            # Find uncovered words (cover_count == 0)
            uncovered = np.where(cover_count == 0)[0]
            
            if len(uncovered) == 0:
                # No uncovered words, keep the removal
                code = np.delete(code, idx_c)
                code_set.discard(c)
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code.copy()
            else:
                # Need to find a replacement word that covers all uncovered words
                # A word w covers word u if Hamming distance(w, u) <= 1
                # So w must be within distance 1 of all uncovered words
                # This means w must be in the intersection of the "closed balls" of radius 1 around each uncovered word
                
                # For efficiency, start with the set of all words that are within distance 1 of the first uncovered word
                # Then intersect with others
                # But with many uncovered words, this is expensive.
                
                # Alternative: iterate over all 6561 words and check if they cover all uncovered
                # But that's 6561 * len(uncovered) checks, which could be slow if many uncovered.
                
                # Better: for each candidate word w, check if all uncovered words are in neighbors[w]
                # neighbors[w] is a set of 17 indices. So check if set(uncovered) is subset of set(neighbors[w])
                
                # To speed up, first filter: w must be close to at least one uncovered word
                # Actually, let's just iterate over all words not in code_set and check
                
                found_swap = False
                # Try all possible replacement words
                # For speed, we can precompute for each word the set of uncovered it covers
                # But let's just do a direct check
                
                # Optimization: if len(uncovered) is large, it's unlikely any single word covers all
                # In that case, revert and try a different kick
                if len(uncovered) > 8:
                    # Too many uncovered, revert
                    cover_count[neighbors[c]] += 1
                else:
                    # Check each candidate word
                    for w in range(total):
                        if w in code_set:
                            continue
                        # Check if w covers all uncovered words
                        # w covers u if u is in neighbors[w] or u == w
                        # neighbors[w] includes w itself
                        nbrs_w = neighbors[w]
                        # Check if all uncovered are in nbrs_w
                        # Use set intersection
                        uncovered_set = set(uncovered.tolist())
                        if uncovered_set.issubset(set(nbrs_w.tolist())):
                            # Perform swap
                            code[idx_c] = w
                            code_set.add(w)
                            cover_count[neighbors[w]] += 1
                            found_swap = True
                            break
                    
                    if not found_swap:
                        # Revert
                        cover_count[neighbors[c]] += 1
        
        # Periodically reset to best if we've gone too long without improvement
        if len(code) > best_size and iteration % 100 == 0:
            code = best_code.copy()
            code_set = set(code.tolist())
            cover_count = np.zeros(total, dtype=np.int64)
            for c in code:
                cover_count[neighbors[c]] += 1

    # Final prune
    if not is_time_up():
        changed = True
        while changed and not is_time_up():
            changed = False
            if len(code) == 0:
                break
            nbrs_of_code = neighbors[code]
            cover_of_nbrs = cover_count[nbrs_of_code]
            min_cover = cover_of_nbrs.min(axis=1)
            removable_mask = min_cover >= 2
            if not np.any(removable_mask):
                break
            removable_indices = np.where(removable_mask)[0]
            random.shuffle(removable_indices.tolist())
            for ri in removable_indices:
                if is_time_up():
                    break
                c = code[ri]
                if np.any(cover_count[neighbors[c]] < 2):
                    continue
                cover_count[neighbors[c]] -= 1
                code = np.delete(code, ri)
                code_set.discard(c)
                changed = True
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code.copy()

    # Return best code found
    final_code = best_code if len(best_code) <= len(code) else code
    # Convert to list of lists
    result = []
    for c in final_code:
        word = [0] * n
        temp = c
        for pos in range(n - 1, -1, -1):
            word[pos] = temp % q
            temp //= q
        result.append(word)
    return result
