"""Binary code of length 17, distance >= 6. Greedy + local search."""
import time
import random
import numpy as np

N = 17
MIN_D = 6
START = time.time()
DEADLINE = START + 80.0  # safety margin before 90s kill

def _qr18():
    # Extended quadratic-residue [18,9,6] code, 512 words, as 18-bit ints.
    q = 17
    # Legendre symbol mod 17
    quad = {x: 1 for x in range(1, q)}
    for x in range(1, q):
        quad[x] = 1 if pow(x, (q - 1) // 2, q) == 1 else -1
    # positions 1..16, parity at 17
    def weight(v):
        return sum(quad[(x * v) % q] for x in range(1, q))
    rows = []
    for v in range(1, q):
        bits = 0
        for x in range(1, q):
            if quad[(x * v) % q] == 1:
                bits |= 1 << (x - 1)
        rows.append(bits)
    # even-weight coset (the QR code), 256 words
    ev = [r for r in rows if bin(r).count("1") % 2 == 0]
    # parity bit so each word has even total weight (extend to length 18)
    words = []
    for r in ev:
        p = bin(r).count("1") % 2
        words.append((r << 1) | p)  # parity in bit 17
    # add all-ones (weight 18, even)
    words.append((1 << 18) - 1)
    return words

def _shorten(w18):
    return w18 & ((1 << 17) - 1)  # drop coordinate 18

def _dists(a, b):
    return bin(a ^ b).count("1")

def _valid(words):
    n = len(words)
    for i in range(n):
        for j in range(i + 1, n):
            if _dists(words[i], words[j]) < MIN_D:
                return False
    return True

def construct():
    global DEADLINE
    DEADLINE = time.time() + 80.0
    rng = random.Random(12345)

    # --- seed: shortened [18,9,6] QR code (256 words) ---
    qr = _qr18()
    code = [_shorten(w) for w in qr]
    code = list(dict.fromkeys(code))
    if not _valid(code):
        # fallback: keep only valid subset via greedy
        code = []
        for w in qr:
            w17 = _shorten(w)
            if all(_dists(w17, c) >= MIN_D for c in code):
                code.append(w17)

    best = len(code)
    best_code = list(code)

    # --- local search: 1-for-1 swap and 1-for-2 swap ---
    def try_add(cand, words):
        return all(_dists(cand, w) >= MIN_D for w in words)

    # candidate pool: all 17-bit words with weight in a useful range
    # For distance 6, words of weight 8 or 9 are good. Use a random pool.
    pool = []
    # generate candidate words of weight 8 and 9 (combinatorial), but that's huge.
    # Instead use random words biased to mid-weight.
    for _ in range(200000):
        w = rng.getrandbits(17)
        wt = bin(w).count("1")
        if 6 <= wt <= 11:
            pool.append(w)

    def local_search(words):
        words = list(words)
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            # 1-for-1: replace one word with a candidate
            for _ in range(4000):
                if time.time() >= DEADLINE:
                    break
                i = rng.randrange(len(words))
                cand = pool[rng.randrange(len(pool))]
                if cand in words:
                    continue
                rest = words[:i] + words[i + 1:]
                if try_add(cand, rest):
                    words[i] = cand
                    improved = True
            # 1-for-2: remove two words, add three candidates
            for _ in range(1500):
                if time.time() >= DEADLINE:
                    break
                i, j = rng.sample(range(len(words)), 2)
                cands = [pool[rng.randrange(len(pool))] for _ in range(3)]
                rest = [w for k, w in enumerate(words) if k not in (i, j)]
                new_words = list(rest)
                added = 0
                for c in cands:
                    if c in new_words:
                        continue
                    if try_add(c, new_words):
                        new_words.append(c)
                        added += 1
                if added >= 2:
                    words = new_words
                    improved = True
        return words

    # run local search for the remaining budget
    while time.time() < DEADLINE:
        candidate = local_search(best_code)
        if len(candidate) > best:
            best = len(candidate)
            best_code = candidate

    if time.time() > DEADLINE - 2.0:
        return best_code
    return best_code
