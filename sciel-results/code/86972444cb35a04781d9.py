import time
import random

def construct():
    t0 = time.time()
    TLIMIT = 86.0

    def elapsed():
        return time.time() - t0

    MAXV = 256

    def validate(seq):
        n = len(seq)
        if n == 0:
            return True
        prev = 0
        for v in seq:
            if v < 1 or v > 256 or v <= prev:
                return False
            prev = v
        prefix = [0]
        acc = 0
        for v in seq:
            acc += v
            prefix.append(acc)
        bits = 0
        for i in range(n):
            for j in range(i + 1, n + 1):
                s = prefix[j] - prefix[i]
                if (bits >> s) & 1:
                    return False
                bits |= (1 << s)
        return True

    def greedy_fast(start_val, seed_off):
        rng = random.Random(seed_off)
        seq = []
        prefix = [0]
        bits = 0
        last = 0
        for _ in range(MAXV):
            valid = []
            cur_end = prefix[-1]
            for x in range(last + 1, MAXV + 1):
                np = cur_end + x
                nb = 0
                ok = True
                for p in prefix:
                    s = np - p
                    if (bits >> s) & 1 or (nb >> s) & 1:
                        ok = False
                        break
                    nb |= (1 << s)
                if ok:
                    valid.append((x, np, nb))
            if not valid:
                break
            if len(valid) == 1:
                x, np, nb = valid[0]
            else:
                n = len(valid)
                k = min(n, 3 + rng.randrange(6))
                idx = rng.randrange(k)
                x, np, nb = valid[idx]
            seq.append(x)
            prefix.append(np)
            bits |= nb
            last = x
        return seq

    def build_state(seq):
        prefix = [0]
        acc = 0
        for v in seq:
            acc += v
            prefix.append(acc)
        bits = 0
        n = len(seq)
        for i in range(n):
            for j in range(i + 1, n + 1):
                s = prefix[j] - prefix[i]
                bits |= (1 << s)
        last = seq[-1] if seq else 0
        return prefix, bits, last

    def extend_dfs(seq, target_len, deadline):
        n0 = len(seq)
        if n0 >= target_len:
            return seq
        prefix, bits, last = build_state(seq)
        work_seq = list(seq)
        work_prefix = list(prefix)
        work_bits = bits
        work_last = last
        found = [None]

        def dfs(depth):
            if elapsed() > deadline or found[0] is not None:
                return
            if depth >= target_len:
                found[0] = list(work_seq)
                return
            if depth + (MAXV - work_last) < target_len:
                return
            cur_end = work_prefix[-1]
            cands = []
            for x in range(work_last + 1, MAXV + 1):
                np = cur_end + x
                nb = 0
                ok = True
                for p in work_prefix:
                    s = np - p
                    if (work_bits >> s) & 1 or (nb >> s) & 1:
                        ok = False
                        break
                    nb |= (1 << s)
                if ok:
                    cands.append((x, np, nb))
            for x, np, nb in cands:
                if elapsed() > deadline or found[0] is not None:
                    return
                has_next = True
                if depth + 1 < target_len:
                    has_next = False
                    for nx in range(x + 1, min(x + 12, MAXV + 1)):
                        np2 = np + nx
                        ok2 = True
                        nb2 = 0
                        for p in work_prefix:
                            s = np2 - p
                            if (work_bits >> s) & 1 or (nb >> s) & 1 or (nb2 >> s) & 1:
                                ok2 = False
                                break
                            nb2 |= (1 << s)
                        s = np2 - np
                        if (work_bits >> s) & 1 or (nb >> s) & 1 or (nb2 >> s) & 1:
                            ok2 = False
                        if ok2:
                            has_next = True
                            break
                if not has_next:
                    continue
                work_seq.append(x)
                work_prefix.append(np)
                old_bits = work_bits
                work_bits = old_bits | nb
                dfs(depth + 1)
                work_bits = old_bits
                work_prefix.pop()
                work_seq.pop()
                if found[0] is not None:
                    return
        dfs(n0)
        return found[0]

    best_seq = [1, 2, 4, 8, 16, 32, 64, 128]
    best_len = len(best_seq)

    deadline1 = t0 + 10.0
    for start in range(1, 26):
        if elapsed() > deadline1:
            break
        s = greedy_fast(start, seed_off=start * 31 + 1)
        if len(s) > best_len:
            best_len = len(s)
            best_seq = list(s)

    deadline2 = t0 + 42.0
    attempt = 0
    while elapsed() < deadline2:
        attempt += 1
        start = random.randrange(1, 30)
        s = greedy_fast(start, seed_off=1000 + attempt * 17)
        if len(s) > best_len:
            best_len = len(s)
            best_seq = list(s)
        if attempt % 60 == 0:
            tgt = best_len + 1
            r = extend_dfs(best_seq, tgt, deadline=min(t0 + 5.0, deadline2))
            if r is not None and len(r) > best_len:
                best_len = len(r)
                best_seq = list(r)

    deadline3 = TLIMIT - 2.0
    for extra in range(1, 14):
        if elapsed() > deadline3:
            break
        tgt = best_len + extra
        if best_len + (MAXV - best_seq[-1]) < tgt:
            break
        remaining = max(0.6, deadline3 - elapsed())
        r = extend_dfs(best_seq, tgt, deadline=t0 + remaining)
        if r is not None and len(r) > best_len:
            best_len = len(r)
            best_seq = list(r)
        else:
            for _ in range(2):
                if elapsed() > deadline3:
                    break
                ss = greedy_fast(random.randrange(1, 30), seed_off=5000 + random.randrange(9999))
                if len(ss) >= best_len:
                    ss = ss[:best_len]
                    r = extend_dfs(ss, tgt, deadline=t0 + max(0.6, deadline3 - elapsed()))
                    if r is not None and len(r) > best_len:
                        best_len = len(r)
                        best_seq = list(r)
                        break

    if not validate(best_seq):
        fb = [1, 2, 4, 8, 16, 32, 64, 128]
        assert validate(fb)
        best_seq = fb

    return best_seq
