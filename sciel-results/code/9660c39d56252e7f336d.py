"""Rewritten program: greedy + local-search heuristic for the ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

# Deadline guard: must finish within 120 s wall-clock; use a safety margin.
DEADLINE_S = 115.0


def _all_words(q, n):
    # Enumerate all q^n words as integers 0..q^n-1 in base q.
    return np.arange(q ** n, dtype=np.int64)


def _int_to_word(x, q, n):
    # Convert integer (base q) to list of n digits.
    word = []
    for _ in range(n):
        word.append(int(x % q))
        x //= q
    return word


def _word_to_int(word, q, n):
    v = 0
    for d in word:
        v = v * q + d
    return v


def _neighbors_int(word, q, n):
    # Return list of integers for all words within Hamming distance <=1 of word.
    # word is a list of n ints in 0..q-1.
    base = _word_to_int(word, q, n)
    neigh = {base}
    for i in range(n):
        for d in range(q):
            if d == word[i]:
                continue
            w2 = list(word)
            w2[i] = d
            neigh.add(_word_to_int(w2, q, n))
    return neigh


def _build_cover_table(q, n):
    # Precompute neighbor sets for all words as a dict int -> set of ints (within dist<=1).
    # For q=3,n=7 that's 2187 words, each with 1 + n*(q-1) = 1+14=15 neighbors.
    cover = {}
    for x in range(q ** n):
        w = _int_to_word(x, q, n)
        cover[x] = _neighbors_int(w, q, n)
    return cover


def _greedy_cover(q, n, cover, rng):
    # Greedy set cover: repeatedly pick the word covering the most currently uncovered words.
    total = q ** n
    uncovered = set(range(total))
    chosen = []
    # Precompute the set of words that cover each word (i.e., words within dist<=1).
    # cover[x] is the set of words within dist<=1 of x. So word x covers all y such that x in cover[y].
    # We need reverse: for each candidate codeword c, which words does it cover? That's cover[c].
    # So candidate c covers exactly cover[c].
    # Greedy: pick c with max |cover[c] ∩ uncovered|.
    while uncovered:
        best_c = None
        best_count = -1
        # To speed up, we can sample or use a heap, but for 2187 words it's fine to scan.
        for c in range(total):
            cnt = 0
            # Count how many of cover[c] are in uncovered
            for y in cover[c]:
                if y in uncovered:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_c = c
        if best_c is None or best_count <= 0:
            break
        chosen.append(best_c)
        for y in cover[best_c]:
            uncovered.discard(y)
    return chosen


def _local_search(q, n, cover, chosen, rng, time_limit):
    # Local search: try to remove a codeword if coverage remains; try to swap one for another to reduce size.
    # We represent chosen as a set of ints.
    chosen_set = set(chosen)
    start_time = time.time()
    
    # Helper: check if a set of codewords covers all words.
    def is_covering(code_set):
        if not code_set:
            return False
        # Union of cover[c] for c in code_set must be all words.
        # Efficient: iterate words and check if any codeword is within dist 1.
        # For q=3,n=7, 2187 words, each check is O(|code_set| * n * q) worst case.
        # Better: precompute for each word the set of codewords that cover it.
        # But let's do a simple check first.
        for x in range(q ** n):
            # Check if x is within dist 1 of any c in code_set
            w = _int_to_word(x, q, n)
            found = False
            for c in code_set:
                wc = _int_to_word(c, q, n)
                dist = sum(1 for i in range(n) if w[i] != wc[i])
                if dist <= 1:
                    found = True
                    break
            if not found:
                return False
        return True
    
    # For efficiency, let's use a different approach: maintain a coverage count array.
    # cov[x] = number of codewords within dist 1 of x.
    def compute_cov(code_set):
        cov = np.zeros(q ** n, dtype=np.int32)
        for c in code_set:
            for y in cover[c]:
                cov[y] += 1
        return cov
    
    def try_remove(code_set):
        # Try to remove a codeword if coverage remains.
        cov = compute_cov(code_set)
        for c in list(code_set):
            # If we remove c, which words lose coverage?
            # Words y where c is the only codeword within dist 1.
            # That is, y in cover[c] and cov[y] == 1.
            can_remove = True
            for y in cover[c]:
                if cov[y] == 1:
                    can_remove = False
                    break
            if can_remove:
                code_set.discard(c)
                return True
        return False
    
    def try_swap(code_set, rng):
        # Try to swap one codeword for another to reduce size (i.e., remove one and add one that covers more).
        # Actually, to reduce size, we want to remove one without adding. That's try_remove.
        # To improve, we might swap to enable future removals.
        # Let's just do random perturbations: remove a random codeword, then repair.
        if len(code_set) < 2:
            return False
        c_remove = rng.choice(list(code_set))
        code_set.discard(c_remove)
        # Repair: greedily add codewords until covered.
        cov = compute_cov(code_set)
        uncovered = [x for x in range(q ** n) if cov[x] == 0]
        if not uncovered:
            return True
        # Add codewords to cover uncovered
        for x in uncovered:
            # Find a codeword that covers x and maximizes new coverage
            best_c = None
            best_gain = -1
            for c in range(q ** n):
                if c in code_set:
                    continue
                gain = 0
                for y in cover[c]:
                    if cov[y] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c is not None and best_gain > 0:
                code_set.add(best_c)
                cov = compute_cov(code_set)
        return True
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try removals
        while try_remove(chosen_set):
            improved = True
        # Try swaps/perturbations
        if (time.time() - start_time) < time_limit:
            if try_swap(chosen_set, rng):
                improved = True
                # After swap, try removals again
                while try_remove(chosen_set):
                    improved = True
    
    return list(chosen_set)


def construct(q, n, radius):
    start_time = time.time()
    rng = random.Random(42)
    
    # Handle trivial cases
    if n == 0:
        return [[]]
    if q == 1:
        return [[0] * n]
    
    # For small spaces, we can do exact or near-exact search
    total = q ** n
    
    # Precompute cover table
    cover = _build_cover_table(q, n)
    
    # Greedy initial solution
    chosen = _greedy_cover(q, n, cover, rng)
    
    # Local search to improve
    time_left = DEADLINE_S - (time.time() - start_time)
    if time_left > 1.0:
        chosen = _local_search(q, n, cover, chosen, rng, time_left - 0.5)
    
    # Convert back to list of lists
    result = [_int_to_word(c, q, n) for c in chosen]
    
    # Ensure distinctness
    seen = set()
    unique_result = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique_result.append(w)
    
    # Check deadline
    if time.time() - start_time > DEADLINE_S:
        # Return what we have
        pass
    
    return unique_result
# EVOLVE-BLOCK-END
