import itertools
import random
import time

V = 19
K = 8
T = 5

# Precompute all 5-subsets and their bitmask representation
all_5_subs = list(itertools.combinations(range(V), T))
N_SUBS = len(all_5_subs)  # 11628
sub_mask = [0] * N_SUBS
for i, s in enumerate(all_5_subs):
    m = 0
    for x in s:
        m |= (1 << x)
    sub_mask[i] = m

# For each point, the list of 5-subset indices containing it
point_subs = [[] for _ in range(V)]
for i, s in enumerate(all_5_subs):
    for x in s:
        point_subs[x].append(i)


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin below 120s kill

    def block_mask(block):
        m = 0
        for p in block:
            m |= (1 << p)
        return m

    def covered_subs_mask(block):
        bm = block_mask(block)
        covered = 0
        for p in block:
            for si in point_subs[p]:
                m = sub_mask[si]
                if (m & bm) == bm:
                    covered |= (1 << si)
        return covered

    # Fast block generator: greedily build a k-block.
    # `weights` is a list over V (higher = more desired).
    # `uncovered` is a bitmask of currently uncovered 5-subsets.
    def generate_block(weights, uncovered):
        selected = []
        selected_mask = 0
        covered = 0
        for _ in range(k):
            best_point = -1
            best_score = -1
            for p in range(v):
                if (selected_mask >> p) & 1:
                    continue
                # new 5-subsets added by p given current selected set
                new_mask = 0
                for si in point_subs[p]:
                    m = sub_mask[si]
                    if (m & selected_mask) == selected_mask and (m & covered) == 0:
                        new_mask |= (1 << si)
                gain = bin(new_mask & uncovered).count('1')
                score = gain * 1000000 + weights[p]
                if score > best_score:
                    best_score = score
                    best_point = p
            selected.append(best_point)
            selected_mask |= (1 << best_point)
            for si in point_subs[best_point]:
                m = sub_mask[si]
                if (m & selected_mask) == selected_mask:
                    covered |= (1 << si)
        return selected

    # ---- Phase 1: dynamic greedy with density-driven generation ----
    uncovered = (1 << N_SUBS) - 1  # all subsets uncovered
    blocks = []
    while True:
        if time.time() > deadline - 30.0:
            break
        if uncovered == 0:
            break
        # weights biased toward uncovered regions
        weights = [0] * v
        u = uncovered
        while u:
            lsb = u & (-u)
            si = lsb.bit_length() - 1
            u ^= lsb
            m = sub_mask[si]
            mm = m
            while mm:
                lsb2 = mm & (-mm)
                p = lsb2.bit_length() - 1
                weights[p] += 1
                mm ^= lsb2
        # generate a few candidate blocks, pick the one covering most uncovered
        best_block = None
        best_gain = -1
        for trial in range(5):
            cand = generate_block(weights, uncovered)
            covered = covered_subs_mask(cand)
            gain = bin(covered & uncovered).count('1')
            if gain > best_gain:
                best_gain = gain
                best_block = cand
        if best_block is None or best_gain <= 0:
            break
        blocks.append(best_block)
        covered = covered_subs_mask(best_block)
        uncovered &= ~covered

    if uncovered != 0:
        # fallback: fill remaining with plain greedy over all blocks
        for b in itertools.combinations(range(v), k):
            if uncovered == 0:
                break
            if time.time() > deadline - 5.0:
                break
            covered = covered_subs_mask(b)
            if (covered & uncovered) != 0:
                blocks.append(list(b))
                uncovered &= ~covered
    if uncovered != 0:
        # last resort: brute force all blocks to guarantee validity
        for b in itertools.combinations(range(v), k):
            if uncovered == 0:
                break
            covered = covered_subs_mask(b)
            uncovered &= ~covered
            blocks.append(list(b))

    # ---- Phase 2: coverage-based pruning ----
    counts = [0] * N_SUBS
    for b in blocks:
        covered = covered_subs_mask(b)
        c = covered
        while c:
            lsb = c & (-c)
            si = lsb.bit_length() - 1
            c ^= lsb
            counts[si] += 1

    changed = True
    while changed and time.time() < deadline - 10.0:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            b = blocks[i]
            bm = block_mask(b)
            unique = False
            for p in b:
                for si in point_subs[p]:
                    m = sub_mask[si]
                    if (m & bm) == bm and counts[si] == 1:
                        unique = True
                        break
                if unique:
                    break
            if not unique:
                covered = covered_subs_mask(b)
                c = covered
                while c:
                    lsb = c & (-c)
                    si = lsb.bit_length() - 1
                    c ^= lsb
                    counts[si] -= 1
                blocks.pop(i)
                changed = True

    # ---- Phase 3: swap improvement ----
    if time.time() < deadline - 5.0:
        for i in range(len(blocks)):
            if time.time() > deadline - 5.0:
                break
            b = blocks[i]
            bm = block_mask(b)
            unique_subs = []
            for p in b:
                for si in point_subs[p]:
                    m = sub_mask[si]
                    if (m & bm) == bm and counts[si] == 1:
                        unique_subs.append(si)
            if not unique_subs:
                continue
            # try a few random replacement blocks
            for _ in range(30):
                if time.time() > deadline - 5.0:
                    break
                cand = list(random.sample(range(v), k))
                cm = block_mask(cand)
                ok = True
                for si in unique_subs:
                    if (sub_mask[si] & cm) != sub_mask[si]:
                        ok = False
                        break
                if ok and cand != b:
                    # swap
                    old_cov = covered_subs_mask(b)
                    c = old_cov
                    while c:
                        lsb = c & (-c)
                        si = lsb.bit_length() - 1
                        c ^= lsb
                        counts[si] -= 1
                    new_cov = covered_subs_mask(cand)
                    c = new_cov
                    while c:
                        lsb = c & (-c)
                        si = lsb.bit_length() - 1
                        c ^= lsb
                        counts[si] += 1
                    blocks[i] = cand
                    break

    return blocks
