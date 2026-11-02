import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0

    total_space = q ** n

    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    masks = [0] * total_space
    for i in range(total_space):
        m = 1 << i
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                m |= (1 << (i + diff * powers[pos]))
        masks[i] = m

    full_mask = (1 << total_space) - 1

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def build_counts(selected):
        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            m = masks[w]
            mm = m
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit
        return counts

    def polish(selected, time_limit):
        if not selected:
            return selected

        counts = build_counts(selected)

        # Phase 1: remove redundant words
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]

                removable = True
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        removable = False
                        break
                    mm ^= lowbit

                if removable:
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(idx)
                    changed = True

        # Phase 2: try swaps - remove one word, add another that covers uncovered words
        # Actually this doesn't reduce count. Let's try: for each word, see if removing it and adding a different word keeps coverage but... no, that keeps same count.
        # Better: try to find a word whose removal is "almost" redundant, then see if we can add one word to cover the gap while removing two others.
        # This is complex. Let's just do multiple rounds of removal.

        # Phase 3: more aggressive - try removing each word and checking if the remaining set still covers everything
        # This is O(n * 2187) which is fine for small n
        for _ in range(10):
            if time.time() > time_limit:
                break
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)
            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]
                removable = True
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        removable = False
                        break
                    mm ^= lowbit
                if removable:
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(idx)
                    changed = True
            if not changed:
                break

        return selected

    def run_greedy(randomized=False, time_limit=None):
        if time_limit is None:
            time_limit = deadline - 5

        remaining = full_mask
        selected = []

        while remaining and time.time() < time_limit:
            best_gain = -1
            best_word = -1
            tie_words = []

            for w in range(total_space):
                gain = (masks[w] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
                    tie_words = [w]
                elif gain == best_gain and gain > 0:
                    tie_words.append(w)

            if best_word < 0:
                break

            if randomized and len(tie_words) > 1:
                best_word = random.choice(tie_words)

            selected.append(best_word)
            remaining &= ~masks[best_word]

        return selected

    best_size = total_space
    best_selected = list(range(total_space))

    if time.time() < deadline - 10:
        sol = run_greedy(randomized=False)
        if len(sol) < best_size:
            best_size = len(sol)
            best_selected = sol[:]
            polished = polish(best_selected[:], deadline - 5)
            if len(polished) < best_size:
                best_size = len(polished)
                best_selected = polished[:]

    while time.time() < deadline - 15:
        sol = run_greedy(randomized=True, time_limit=deadline - 15)
        if len(sol) < best_size:
            best_size = len(sol)
            best_selected = sol[:]
            polished = polish(best_selected[:], deadline - 5)
            if len(polished) < best_size:
                best_size = len(polished)
                best_selected = polished[:]

    if time.time() < deadline - 3:
        polished = polish(best_selected[:], deadline - 2)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
