"""Global set-cover greedy with incremental gain updates for covering designs."""
import itertools
import time


def construct(v, k, t):
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s kill

    points = range(v)
    all_t = list(itertools.combinations(points, t))
    all_tm1 = list(itertools.combinations(points, t - 1))
    all_blocks = list(itertools.combinations(points, k))

    # Index t-subsets and (t-1)-subsets
    t_index = {ts: i for i, ts in enumerate(all_t)}
    tm1_index = {ts: i for i, ts in enumerate(all_tm1)}
    n_t = len(all_t)
    n_tm1 = len(all_tm1)

    # For each candidate block, precompute its (t-1)-subsets (indices) and t-subsets (indices)
    # A t-subset of a block is "newly covered" when exactly one of its (t-1)-subsets is available.
    block_tm1 = []
    block_t = []
    for b in all_blocks:
        tm1_list = [tm1_index[ts] for ts in itertools.combinations(b, t - 1)]
        t_list = [t_index[ts] for ts in itertools.combinations(b, t)]
        block_tm1.append(tm1_list)
        block_t.append(t_list)

    # State
    covered = [False] * n_t          # is t-subset covered
    cnt = [0] * n_tm1                # how many blocks cover each (t-1)-subset
    active = [True] * len(all_blocks)
    gain = [0] * len(all_blocks)     # current gain of each block

    # Initialize gains: with cnt=0, every (t-1)-subset is available, so gain = C(k,t)
    base_gain = len(block_t[0])
    for i in range(len(all_blocks)):
        gain[i] = base_gain

    blocks = []
    n_uncovered = n_t

    while n_uncovered > 0:
        if time.time() > deadline:
            break

        # Pick the active block with the maximum gain
        best = -1
        best_gain = -1
        for i in range(len(all_blocks)):
            if active[i] and gain[i] > best_gain:
                best_gain = gain[i]
                best = i
        if best < 0:
            break

        blocks.append(all_blocks[best])
        active[best] = False

        # Update coverage: for each t-subset in this block, if exactly one of its
        # (t-1)-subsets was available (cnt==1) before this block, it becomes covered.
        for t_idx in block_t[best]:
            if covered[t_idx]:
                continue
            ts = all_t[t_idx]
            avail = 0
            for j in range(t):
                sub = ts[:j] + ts[j+1:]
                if cnt[tm1_index[sub]] == 1:
                    avail += 1
            if avail == 1:
                covered[t_idx] = True
                n_uncovered -= 1

        # Update cnt for this block's (t-1)-subsets
        for tm1_idx in block_tm1[best]:
            cnt[tm1_idx] += 1

        # Update gains of remaining active blocks
        for i in range(len(all_blocks)):
            if not active[i]:
                continue
            new_gain = 0
            for t_idx in block_t[i]:
                if covered[t_idx]:
                    continue
                ts = all_t[t_idx]
                avail = 0
                for j in range(t):
                    sub = ts[:j] + ts[j+1:]
                    if cnt[tm1_index[sub]] == 1:
                        avail += 1
                if avail == 1:
                    new_gain += 1
            gain[i] = new_gain

    return [list(b) for b in blocks]
