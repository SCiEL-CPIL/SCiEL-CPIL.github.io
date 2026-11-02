"""Football pool: ternary covering code K_3(7,1)."""
import time
import random
from collections import defaultdict


def _cover_mask_for_word(word, q, n):
    """Return a frozenset of tuples for all words within distance 1 of word."""
    covered = [tuple(word)]
    for i in range(n):
        orig = word[i]
        for v in range(q):
            if v == orig:
                continue
            new_word = word.copy()
            new_word[i] = v
            covered.append(tuple(new_word))
    return frozenset(covered)


def _covers_word(word_tuple, codewords, q, n):
    """Check if word_tuple is within distance 1 of any codeword."""
    for cw in codewords:
        dist = sum(1 for i in range(n) if word_tuple[i] != cw[i])
        if dist <= 1:
            return True
    return False


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s

    # Enumerate all 2187 words
    all_words = []
    for i in range(q ** n):
        w = []
        x = i
        for _ in range(n):
            w.append(x % q)
            x //= q
        all_words.append(tuple(w))

    n_total = len(all_words)  # 2187

    # Precompute cover sets for all words
    cover_sets = []
    for w in all_words:
        cs = [w]
        for i in range(n):
            orig = w[i]
            for v in range(q):
                if v == orig:
                    continue
                nw = list(w)
                nw[i] = v
                cs.append(tuple(nw))
        cover_sets.append(frozenset(cs))

    # Build word index map
    word_index = {w: i for i, w in enumerate(all_words)}

    # Convert cover sets to lists of indices for faster operations
    cover_indices = []
    for cs in cover_sets:
        idxs = []
        for w in cs:
            idxs.append(word_index[w])
        cover_indices.append(idxs)

    # Greedy set cover: start with all words, then greedily select
    # We want to find a small set S such that union of cover_sets[S] = all words
    # Greedy: repeatedly pick the word whose cover covers the most uncovered words

    uncovered = [True] * n_total  # all words initially uncovered
    n_uncovered = n_total

    codewords = []
    codeword_indices = []

    # Precompute cover as list of indices
    # Greedy set cover
    while n_uncovered > 0:
        if time.time() > deadline - 5.0:
            break

        best_idx = -1
        best_count = 0
        for i in range(n_total):
            if not uncovered[i]:
                continue
            count = 0
            for j in cover_indices[i]:
                if uncovered[j]:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = i

        if best_idx == -1:
            break

        codewords.append(list(all_words[best_idx]))
        codeword_indices.append(best_idx)

        # Mark covered
        for j in cover_indices[best_idx]:
            if uncovered[j]:
                uncovered[j] = False
                n_uncovered -= 1

    # Now try to remove redundant codewords
    # A codeword is redundant if removing it still leaves all words covered
    def is_redundant(cw_idx, cw_list):
        """Check if cw at position cw_idx is redundant."""
        # For each word covered by cw, check if it's covered by another cw
        for j in cover_indices[cw_idx]:
            found = False
            for k in range(len(cw_list)):
                if k == cw_idx:
                    continue
                if j in cover_indices[cw_list[k]]:
                    found = True
                    break
            if not found:
                return False
        return True

    # Use a set for faster lookup of covered-by-codeword
    # Rebuild: for each word, which codewords cover it?
    # This is expensive, so use a different approach:
    # For removal, check each codeword: does removing it leave all its covered words still covered?

    # More efficient: maintain for each word, the set of codeword indices that cover it
    # But that's 2187 * |S| entries. Let's use a different strategy.

    # Strategy: iterate through codewords, try to remove each one
    # For removal check: for each word in cover_indices[cw], is there another cw' that covers it?

    # Build reverse: for each word, list of codeword indices that cover it
    word_to_cws = defaultdict(list)
    for ci, cwi in enumerate(codeword_indices):
        for j in cover_indices[cwi]:
            word_to_cws[j].append(ci)

    # Try to remove redundant codewords
    changed = True
    while changed:
        changed = False
        for ci in range(len(codeword_indices)):
            if time.time() > deadline - 3.0:
                break
            cwi = codeword_indices[ci]
            redundant = True
            for j in cover_indices[cwi]:
                # Check if j is covered by any other codeword
                other_cover = False
                for ci2 in word_to_cws[j]:
                    if ci2 != ci:
                        other_cover = True
                        break
                if not other_cover:
                    redundant = False
                    break
            if redundant:
                # Remove codeword ci
                codeword_indices.pop(ci)
                codewords.pop(ci)
                # Update word_to_cws
                for j in cover_indices[cwi]:
                    if ci in word_to_cws[j]:
                        word_to_cws[j].remove(ci)
                changed = True
                break  # restart from beginning since indices changed
        if time.time() > deadline - 3.0:
            break

    # Local search: try to replace a codeword with another to reduce size
    # Actually, let's try a different approach: hill climbing on the set
    # Try removing a codeword and adding a different one if it helps

    # Simpler local search: try swapping each codeword with a random word to see if we can then remove another
    # For now, let's do a more aggressive reduction pass

    # Second pass: more careful redundancy check
    # Rebuild word_to_cws from scratch
    word_to_cws = defaultdict(list)
    for ci, cwi in enumerate(codeword_indices):
        for j in cover_indices[cwi]:
            word_to_cws[j].append(ci)

    # Try removing in multiple passes
    for _pass in range(10):
        if time.time() > deadline - 2.0:
            break
        removed_any = False
        for ci in range(len(codeword_indices)):
            if time.time() > deadline - 2.0:
                break
            cwi = codeword_indices[ci]
            redundant = True
            for j in cover_indices[cwi]:
                if len(word_to_cws[j]) <= 1:
                    redundant = False
                    break
            if redundant:
                codeword_indices.pop(ci)
                codewords.pop(ci)
                for j in cover_indices[cwi]:
                    if ci in word_to_cws[j]:
                        word_to_cws[j].remove(ci)
                removed_any = True
                break
        if not removed_any:
            break

    # Verify coverage
    # Check all words are covered
    covered_all = set()
    for cwi in codeword_indices:
        for j in cover_indices[cwi]:
            covered_all.add(j)

    if len(covered_all) != n_total:
        # Add back missing words
        for j in range(n_total):
            if j not in covered_all:
                codewords.append(list(all_words[j]))
                codeword_indices.append(j)

    # Final redundancy cleanup
    if time.time() < deadline - 1.0:
        # Quick pass
        changed = True
        while changed:
            changed = False
            for ci in range(len(codeword_indices)):
                if time.time() > deadline - 0.5:
                    break
                cwi = codeword_indices[ci]
                redundant = True
                for j in cover_indices[cwi]:
                    # Check if any other codeword covers j
                    found_other = False
                    for ci2 in range(len(codeword_indices)):
                        if ci2 == ci:
                            continue
                        if j in cover_indices[codeword_indices[ci2]]:
                            found_other = True
                            break
                    if not found_other:
                        redundant = False
                        break
                if redundant:
                    codeword_indices.pop(ci)
                    codewords.pop(ci)
                    changed = True
                    break

    return codewords
