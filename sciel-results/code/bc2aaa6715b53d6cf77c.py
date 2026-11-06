"""Combined program: Reed-Muller seed + greedy add + local search with replace-one-with-two."""
import random
import time

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 45.0  # Safety margin for 60s/90s limit

    # Start with Reed-Muller code (32 words, distance 8)
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)

    rng = random.Random(42)
    best_size = len(words)
    best_words = list(words)

    def can_add(w, code):
        for c in code:
            if (w ^ c).bit_count() < MIN_DIST:
                return False
        return True

    def try_add_random(code, rng, attempts=300):
        for _ in range(attempts):
            if time.time() - start_time > time_limit:
                return False
            w = rng.getrandbits(N)
            if can_add(w, code):
                code.append(w)
                return True
        return False

    def try_flip(code, rng):
        idx = rng.randrange(len(code))
        bit = rng.randrange(N)
        new_w = code[idx] ^ (1 << bit)
        for j, c in enumerate(code):
            if j != idx and (new_w ^ c).bit_count() < MIN_DIST:
                return False
        code[idx] = new_w
        return True

    def try_double_flip(code, rng):
        idx = rng.randrange(len(code))
        b1 = rng.randrange(N)
        b2 = rng.randrange(N)
        if b1 == b2:
            b2 = (b2 + 1) % N
        new_w = code[idx] ^ (1 << b1) ^ (1 << b2)
        for j, c in enumerate(code):
            if j != idx and (new_w ^ c).bit_count() < MIN_DIST:
                return False
        code[idx] = new_w
        return True

    def try_swap(code, rng):
        idx = rng.randrange(len(code))
        old_w = code[idx]
        code.pop(idx)
        for _ in range(200):
            if time.time() - start_time > time_limit:
                code.insert(idx, old_w)
                return False
            w = rng.getrandbits(N)
            if can_add(w, code):
                code.append(w)
                return True
        code.insert(idx, old_w)
        return False

    def try_replace_one_with_two(code, rng, attempts=100):
        """Remove one codeword, try to add two new ones. Net gain +1 if successful."""
        if len(code) < 2:
            return False
        idx = rng.randrange(len(code))
        removed = code.pop(idx)
        added_count = 0
        added_words = []
        for _ in range(attempts):
            if time.time() - start_time > time_limit:
                # Restore
                code.insert(idx, removed)
                for w in added_words:
                    code.append(w)
                return False
            w = rng.getrandbits(N)
            if can_add(w, code):
                code.append(w)
                added_words.append(w)
                added_count += 1
                if added_count == 2:
                    return True
        # Restore if we didn't get two
        code.insert(idx, removed)
        for w in added_words:
            code.append(w)
        return False

    def kick(code, rng, k=3):
        if time.time() - start_time > time_limit:
            return
        removed = []
        for _ in range(min(k, len(code))):
            idx = rng.randrange(len(code))
            removed.append(code.pop(idx))
        for w in removed:
            if can_add(w, code):
                code.append(w)
        try_add_random(code, rng, attempts=100)

    # Phase 1: Greedy add
    greedy_deadline = start_time + time_limit * 0.25
    while time.time() < greedy_deadline:
        if not try_add_random(words, rng, attempts=200):
            break

    if len(words) > best_size:
        best_size = len(words)
        best_words = list(words)

    # Phase 2: Local search
    while time.time() - start_time < time_limit:
        if time.time() - start_time > time_limit:
            break
        move = rng.random()
        if move < 0.25:
            try_add_random(words, rng, attempts=100)
        elif move < 0.45:
            try_flip(words, rng)
        elif move < 0.60:
            try_double_flip(words, rng)
        elif move < 0.75:
            try_swap(words, rng)
        elif move < 0.90:
            try_replace_one_with_two(words, rng, attempts=150)
        else:
            kick(words, rng, k=rng.randint(2, 5))

        if len(words) > best_size:
            best_size = len(words)
            best_words = list(words)

    if len(best_words) >= len(words):
        return best_words
    return words
