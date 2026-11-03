def greedy_init(seed):
    rng = random.Random(seed)
    counts = np.zeros(N, dtype=np.int32)
    chosen = []
    chosen_mask = np.zeros(N, dtype=bool)
    gain = np.full(N, 17, dtype=np.int32)
    prio = [rng.random() for _ in range(N)]
    heap = [(-gain[w], prio[w], w) for w in range(N)]
    heapq.heapify(heap)
    # affected set tracking not needed; just lazily recompute on pop.
    while len(chosen) < N:
        if is_deadline():
            break
        neg_g, p, w = heapq.heappop(heap)
        if chosen_mask[w]:
            continue
        if -neg_g != gain[w]:
            continue
        if gain[w] <= 0:
            break
        # add w
        chosen.append(w)
        chosen_mask[w] = True
        b = balls[w]
        counts[b] += 1
        # recompute gain for candidates whose ball intersects b
        # affected candidates: for each u in b, for each v in rev[u]
        affected = set()
        for u in b:
            affected.update(rev[u])
        for v in affected:
            if chosen_mask[v]:
                continue
            gain[v] = np.count_nonzero((counts == 0)[balls[v]])
            heapq.heappush(heap, (-gain[v], prio[v], v))
        if full_coverage(counts):
            break
    return chosen, counts
