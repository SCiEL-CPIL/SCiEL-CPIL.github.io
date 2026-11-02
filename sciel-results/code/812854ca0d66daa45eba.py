import time, sys
from itertools import permutations

_DEADLINE = 150.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - time.time()

def _classic(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word

def _perm_index(n):
    # map each permutation tuple to 0..n!-1
    perms = list(permutations(range(1, n + 1)))
    idx = {p: i for i, p in enumerate(perms)}
    return perms, idx

def _beam(n, time_budget=140.0):
    perms, idx = _perm_index(n)
    total = n ** 0  # placeholder
    fact = 1
    for i in range(2, n + 1):
        fact *= i
    all_mask = (1 << fact) - 1
    # state: (last_tuple, mask, word_len, path)  -- store path as list for reconstruction
    # To save memory, store path as a list of symbols; beam width small.
    start = (1,)  # last (n-1)? for n>=2 start with (1,) then build
    # We'll build word starting from [1]. last window of length n-1 initially needs n-1 symbols.
    # Represent state by (suffix tuple of length up to n-1, mask, word list)
    # For n=1 handle separately.
    if n == 1:
        return [1]
    # initial word [1]; suffix length 1
    beam = [((1,), 0, [1])]
    beam_width = 200000
    while True:
        if _remaining() < 5.0:
            break
        new_beam = []
        for (sfx, mask, word) in beam:
            for sym in range(1, n + 1):
                new_word = word + [sym]
                if len(new_word) >= n:
                    w = tuple(new_word[-n:])
                    if w in idx:
                        bit = 1 << idx[w]
                        if not (mask & bit):
                            new_mask = mask | bit
                            if new_mask == all_mask:
                                return new_word
                        else:
                            new_mask = mask
                    else:
                        new_mask = mask
                else:
                    new_mask = mask
                new_sfx = tuple(new_word[-(n - 1):]) if len(new_word) >= n - 1 else tuple(new_word)
                new_beam.append((new_sfx, new_mask, new_word))
        # prune: keep best by (most covered, shortest)
        # dedupe by (sfx, mask) keeping shortest word
        best = {}
        for (sfx, mask, word) in new_beam:
            key = (sfx, mask)
            if key not in best or len(word) < len(best[key][2]):
                best[key] = (sfx, mask, word)
        cand = list(best.values())
        cand.sort(key=lambda x: (-x[1].bit_count(), len(x[2])))
        beam = cand[:beam_width]
        if not beam:
            break
    return None

def construct(n):
    if n == 1:
        return "1"
    if n <= 4:
        # optimal via beam (fast)
        res = _beam(n, time_budget=20.0)
        if res is not None:
            return "".join(map(str, res))
    # try beam for all n with deadline guard
    res = _beam(n)
    if res is not None:
        return "".join(map(str, res))
    return "".join(map(str, _classic(n)))
