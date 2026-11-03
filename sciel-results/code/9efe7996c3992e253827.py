"""Rewritten superpermutation solver: direct greedy path construction."""
import time
import random
import itertools
from math import factorial

# EVOLVE-BLOCK-START

def construct(n):
    """Return a superpermutation over symbols 1..n (as a string of digits)."""
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234123142312431214321342132431213243214321"

    start_time = time.time()
    TIME_LIMIT = 140  # safety margin before 150s kill

    total = factorial(n)
    n1 = n - 1
    n1total = factorial(n1)

    nperms = list(itertools.permutations(range(n)))
    code_of = {}
    for rank, p in enumerate(nperms):
        c = 0
        for x in p:
            c = c * n + x
        code_of[c] = rank

    prefix_code = [0] * total
    suffix_code = [0] * total
    for rank, p in enumerate(nperms):
        pc = 0
        for x in p[:n1]:
            pc = pc * n1 + x
        prefix_code[rank] = pc
        sc = 0
        for x in p[n1:]:
            sc = sc * n1 + x
        suffix_code[rank] = sc

    prefix_add = {}
    for rank, p in enumerate(nperms):
        prefix_add[(prefix_code[rank], p[n1])] = rank

    # transitions per prefix context
    trans = {}
    for pc_code in range(n1total):
        digits = [0] * n1
        tmp = pc_code
        for k in range(n1 - 1, -1, -1):
            digits[k] = tmp % n1
            tmp //= n1
        present = set(digits)
        opts = []
        for b in range(n):
            if b in present:
                continue
            r = prefix_add.get((pc_code, b))
            if r is not None:
                opts.append((b, r, suffix_code[r]))
        trans[pc_code] = opts

    rng = random.Random(987654321)
    ident = tuple(range(n))
    covered = 1  # first perm rank 0
    word = list(ident)
    pc = 0
    for x in ident[:n1]:
        pc = pc * n1 + x

    step = 0
    max_steps = total * 6
    while covered.bit_count() < total and step < max_steps:
        if time.time() - start_time > TIME_LIMIT:
            break
        opts = trans[pc]
        if not opts:
            break
        adding = [o for o in opts if not (covered >> o[1]) & 1]
        choice = rng.choice(adding) if adding else rng.choice(opts)
        b, r, ns = choice
        covered |= (1 << r)
        word.append(b)
        pc = ns
        step += 1

    if covered.bit_count() < total:
        return _classic_to_str(_classic_superperm(n))

    word = _polish(word, n, code_of, start_time, TIME_LIMIT)
    return "".join(str(x + 1) for x in word)

def _classic_superperm(n):
    word = [0]
    for m in range(2, n + 1):
        seen = set()
        perms = []
        for i in range(len(word) - (m - 1) + 1):
            w = tuple(word[i:i + m - 1])
            if len(set(w)) == m - 1 and w not in seen:
                seen.add(w)
                perms.append(w)
        new = []
        for p in perms:
            block = list(p) + [m - 1] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _classic_to_str(word):
    return "".join(str(x + 1) for x in word)

def _polish(word, n, code_of, start_time, TIME_LIMIT):
    L = len(word)
    if L < n:
        return word

    def covered_count(w):
        cnt = 0
        seen = set()
        for i in range(len(w) - n + 1):
            win = w[i:i + n]
            if len(set(win)) == n:
                c = 0
                for x in win:
                    c = c * n + x
                r = code_of.get(c)
                if r is not None and r not in seen:
                    seen.add(r)
                    cnt += 1
        return cnt

    current = list(word)
    cur_count = covered_count(current)
    improved = True
    while improved and time.time() - start_time < TIME_LIMIT:
        improved = False
        for pos in range(len(current) - 1, -1, -1):
            if time.time() - start_time > TIME_LIMIT:
                break
            if len(current) - 1 < n:
                continue
            trial = current[:pos] + current[pos + 1:]
            if covered_count(trial) == cur_count:
                current = trial
                improved = True
                break
    return current

# EVOLVE-BLOCK-END
