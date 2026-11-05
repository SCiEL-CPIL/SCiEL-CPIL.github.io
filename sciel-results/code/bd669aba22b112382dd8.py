import time
import random

def construct():
    start = time.time()
    deadline = start + 82  # safety margin before 90s kill

    def time_left():
        return deadline - time.time()

    def is_valid(seq):
        """Full O(n^2) distinct-consecutive-sum check."""
        L = len(seq)
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        seen = set()
        for i in range(L):
            pi = P[i]
            for j in range(i + 1, L + 1):
                s = P[j] - pi
                if s in seen:
                    return False
                seen.add(s)
        return True

    def full_sums(seq):
        L = len(seq)
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        s = set()
        for i in range(L):
            pi = P[i]
            for j in range(i + 1, L + 1):
                s.add(P[j] - pi)
        return s

    def insert_at(seq, sums, x, pos):
        """Return (new_seq, new_sums) if inserting x before index pos is valid, else None.
        x must strictly fit between neighbours."""
        L = len(seq)
        if pos > 0 and seq[pos - 1] >= x:
            return None
        if pos < L and x >= seq[pos]:
            return None
        # new sums are x + (sum of l left-neighbours) + (sum of r right-neighbours)
        left = [0]
        acc = 0
        for k in range(pos - 1, -1, -1):
            acc += seq[k]
            left.append(acc)
        right = [0]
        acc = 0
        for k in range(pos, L):
            acc += seq[k]
            right.append(acc)
        for a in left:
            base = x + a
            for b in right:
                s = base + b
                if s in sums:
                    return None
        new_sums = set(sums)
        for a in left:
            base = x + a
            for b in right:
                new_sums.add(base + b)
        new_seq = seq[:pos] + [x] + seq[pos:]
        return new_seq, new_sums

    def try_insert_any(seq, sums, x):
        """Try to insert x at any valid position; return (new_seq,new_sums) or None."""
        L = len(seq)
        for pos in range(L + 1):
            r = insert_at(seq, sums, x, pos)
            if r is not None:
                return r
        return None

    def local_search(seq, sums, budget):
        """Repeatedly add any candidate at any position until stuck or budget hit."""
        seq = list(seq)
        sums = set(sums)
        while time_left() > budget:
            cur_set = set(seq)
            improved = False
            for x in range(1, 257):
                if x in cur_set:
                    continue
                r = try_insert_any(seq, sums, x)
                if r is not None:
                    seq, sums = r
                    cur_set.add(x)
                    improved = True
            if not improved:
                break
        return seq, sums

    best_seq = None
    best_len = 0

    def consider(seq):
        nonlocal best_seq, best_len
        if seq and len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # ---- Phase 1: greedy seeds with position-aware insertion ----
    for seed in range(40):
        if time_left() < 3:
            break
        if seed == 0:
            order = list(range(1, 257))
        elif seed == 1:
            order = list(range(256, 0, -1))
        elif seed == 2:
            order = []
            for i in range(128):
                order.append(i + 1)
                order.append(256 - i)
        elif seed == 3:
            order = [2 ** i for i in range(1, 9) if 2 ** i <= 256]
            order += [x for x in range(1, 257) if x not in order]
        else:
            order = list(range(1, 257))
            random.shuffle(order)
        seq = []
        sums = set()
        for x in order:
            r = try_insert_any(seq, sums, x)
            if r is not None:
                seq, sums = r
        consider(seq)

    # ---- Phase 2: local search (add at any position) from best seed ----
    if best_seq:
        seq, sums = local_search(best_seq, full_sums(best_seq), budget=3)
        consider(seq)

    # ---- Phase 3: perturbation + local search (remove/swap) to escape plateaus ----
    for it in range(200):
        if time_left() < 4:
            break
        seq = list(best_seq)
        sums = full_sums(seq)
        # random perturbation: remove 1-2 elements
        k = random.randint(1, 2)
        for _ in range(k):
            if seq:
                v = random.choice(seq)
                seq.remove(v)
                # recompute sums cheaply
        sums = full_sums(seq)
        seq, sums = local_search(seq, sums, budget=2)
        consider(seq)

    # ---- Final safety: ensure returned sequence is valid ----
    if best_seq is None:
        best_seq = [1]
    best_seq = sorted(best_seq)
    if not is_valid(best_seq):
        # defensive rebuild
        seq = []
        sums = set()
        for x in range(1, 257):
            r = try_insert_any(seq, sums, x)
            if r is not None:
                seq, sums = r
        best_seq = sorted(seq) if seq else [1]

    return best_seq
