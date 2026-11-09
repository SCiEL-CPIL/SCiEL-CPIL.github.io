# EVOLVE-BLOCK-START
import itertools
import time


def construct(q, n, radius):
    deadline = time.time() + 140  # safety margin before 150s kill
    total = q ** n  # 6561 for q=3, n=8

    # Precompute all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute neighborhoods (words within Hamming distance <= radius)
    # For radius=1, each word covers itself + n*(q-1) neighbors
    if radius == 1:
        neighborhoods = []
        for w in all_words:
            idx = word_to_idx[w]
            covered = {idx}
            for pos in range(n):
                for val in range(q):
                    if val != w[pos]:
                        nw = list(w)
                        nw[pos] = val
                        covered.add(word_to_idx[tuple(nw)])
            neighborhoods.append(covered)
    else:
        # Generic neighbor computation for other radii
        neighborhoods = []
        for w in all_words:
            idx = word_to_idx[w]
            covered = {idx}
            for delta in itertools.product(range(q), repeat=n):
                dist = sum(1 for a, b in zip(w, delta) if a != b)
                if 0 < dist <= radius:
                    covered.add(word_to_idx[delta])
            neighborhoods.append(covered)

    # Convert neighborhoods to bitmask integers for fast set operations
    # 6561 bits fits in a Python int
    neigh_masks = []
    for nb in neighborhoods:
        mask = 0
        for i in nb:
            mask |= (1 << i)
        neigh_masks.append(mask)

    # Build initial code: direct sum of ternary tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(q) for b in range(q)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_code.append(h + list(tail))

    # Convert to indices
    code_indices = [word_to_idx[tuple(c)] for c in initial_code]
    code_set = set(code_indices)

    # Compute covered set as bitmask
    covered_mask = 0
    for idx in code_set:
        covered_mask |= neigh_masks[idx]

    # All-words mask
    full_mask = (1 << total) - 1

    def covered_count():
        return bin(covered_mask).count('1')

    def is_complete():
        return covered_mask == full_mask

    # Greedy improvement: try to remove codewords that are redundant
    # A codeword is redundant if all words it uniquely covers are also covered by others
    # More practically: try removing each codeword and check if coverage is maintained

    # For efficiency, track for each word which codewords cover it
    # word_coverers[i] = set of codeword indices in code_set that cover word i
    # This is expensive to maintain, so we'll use a different approach:
    # Try removing each codeword, check if still complete, keep if yes

    # First pass: remove redundant codewords
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        new_code = list(code_set)
        for i in range(len(new_code)):
            if time.time() > deadline - 10:
                break
            ci = new_code[i]
            # Try removing ci
            new_mask = covered_mask & ~neigh_masks[ci]
            # Check if still complete
            if new_mask == full_mask:
                # Redundant, remove it
                code_set.discard(ci)
                covered_mask = new_mask
                new_code.pop(i)
                improved = True
                # Restart since indices changed
                break

    # Second pass: try swapping - replace a codeword with one that covers more uncovered words
    # For each codeword, find if there's a better replacement
    # This is expensive, so do limited iterations
    for iteration in range(50):
        if time.time() > deadline - 15:
            break
        progress = False
        for ci in list(code_set):
            if time.time() > deadline - 15:
                break
            # What does ci uniquely cover? (words only covered by ci)
            # This requires knowing which other codewords cover each word
            # Simplified: try swapping ci with a random word and see if coverage improves
            # Better: find the word that ci covers that is covered by fewest codewords
            # For now, try a limited set of candidates
            
            # Find words covered by ci
            ci_mask = neigh_masks[ci]
            # Words that would become uncovered if we remove ci:
            # We need to check if any word in ci_mask is covered only by ci
            # This is hard without word_coverers tracking
            
            # Alternative: try replacing ci with a nearby word
            # Get the word for ci
            ci_word = all_words[ci]
            # Try neighbors of ci
            for pos in range(n):
                if time.time() > deadline - 15:
                    break
                for val in range(q):
                    if val == ci_word[pos]:
                        continue
                    nw = list(ci_word)
                    nw[pos] = val
                    ni = word_to_idx[tuple(nw)]
                    if ni in code_set:
                        continue
                    # Try swap: remove ci, add ni
                    new_mask = (covered_mask & ~neigh_masks[ci]) | neigh_masks[ni]
                    if new_mask == full_mask:
                        # Valid swap, but does it help? Only if we can later remove more
                        # For now, accept if it maintains coverage
                        # Actually, this doesn't reduce count, so skip
                        pass
        if not progress:
            break

    # Final: try one more round of removals after swaps
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for ci in list(code_set):
            if time.time() > deadline - 5:
                break
            new_mask = covered_mask & ~neigh_masks[ci]
            if new_mask == full_mask:
                code_set.discard(ci)
                covered_mask = new_mask
                improved = True
                break

    # Convert back to list of codewords
    result = [list(all_words[idx]) for idx in sorted(code_set)]
    return result
# EVOLVE-BLOCK-END
