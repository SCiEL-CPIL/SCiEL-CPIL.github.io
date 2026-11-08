import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    n = adjacency.shape[0]
    if n == 0:
        return np.array([])

    nwords = (n + 63) // 64

    # Build neighbor masks.
    neighbor_masks = np.zeros((n, nwords), dtype=np.uint64)
    for w in range(nwords):
        lo = w * 64
        hi = min(lo + 64, n)
        bits = adjacency[:, lo:hi]
        for j in range(hi - lo):
            col = bits[:, j].astype(np.uint64)
            neighbor_masks[:, w] |= (col << j)

    degrees = adjacency.sum(axis=1)

    # Self bits.
    self_bits = np.zeros((n, nwords), dtype=np.uint64)
    for v in range(n):
        w = v // 64
        b = v % 64
        self_bits[v, w] = np.uint64(1 << b)

    blocked_mask = neighbor_masks | self_bits

    # Flat list for speed.
    blocked_flat = [list(map(int, blocked_mask[v])) for v in range(n)]

    # Last word valid bits mask.
    last_w = nwords - 1
    valid_bits_last = n - last_w * 64
    if valid_bits_last < 64:
        last_word_mask = (1 << valid_bits_last) - 1
    else:
        last_word_mask = (1 << 64) - 1

    def make_avail():
        avail = [0] * nwords
        for w in range(nwords):
            avail[w] = (1 << 64) - 1
        avail[last_w] &= last_word_mask
        return avail

    def greedy_size_fast(order):
        avail = make_avail()
        count = 0
        for v in order:
            w = v >> 6
            b = v & 63
            bit = 1 << b
            if avail[w] & bit:
                count += 1
                bm = blocked_flat[v]
                for ww in range(nwords):
                    avail[ww] &= ~bm[ww]
        return count

    def greedy_construct_order():
        avail = make_avail()
        order = []
        for _ in range(n):
            best_v = -1
            best_cost = 10**9
            for v in range(n):
                w = v >> 6
                b = v & 63
                bit = 1 << b
                if not (avail[w] & bit):
                    continue
                bm = blocked_flat[v]
                cost = 0
                for ww in range(nwords):
                    cost += bin(avail[ww] & bm[ww]).count('1')
                if cost < best_cost:
                    best_cost = cost
                    best_v = v
            if best_v == -1:
                break
            order.append(best_v)
            bm = blocked_flat[best_v]
            for ww in range(nwords):
                avail[ww] &= ~bm[ww]
        return order

    def local_search(order, max_iters, deadline, rng, use_sa=True):
        order = list(order)
        current_size = greedy_size_fast(order)
        best_order = order[:]
        best_size = current_size

        temp = 1.0
        cooling = 0.9999

        for it in range(max_iters):
            if time.time() > deadline:
                break
            i = rng.integers(0, n)
            v = order[i]
            j = rng.integers(0, n)
            if i == j or i == j - 1:
                continue
            order.pop(i)
            if j > i:
                j -= 1
            order.insert(j, v)
            new_size = greedy_size_fast(order)
            if new_size >= current_size:
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_order = order[:]
            else:
                accepted = False
                if use_sa:
                    delta = current_size - new_size
                    if delta > 0:
                        if rng.random() < np.exp(-delta / temp):
                            accepted = True
                            current_size = new_size
                if not accepted:
                    order.pop(j)
                    order.insert(i, v)
            temp *= cooling

        return best_order, best_size

    rng = np.random.default_rng(12345)

    initial_orders = []
    initial_orders.append(greedy_construct_order())
    initial_orders.append(list(np.argsort(degrees)))
    initial_orders.append(list(rng.permutation(n)))
    initial_orders.append(list(np.argsort(-degrees)))
    noisy = degrees + rng.normal(0, 1, n)
    initial_orders.append(list(np.argsort(noisy)))

    best_order = initial_orders[0]
    best_size = greedy_size_fast(best_order)

    for order in initial_orders:
        sz = greedy_size_fast(order)
        if sz > best_size:
            best_size = sz
            best_order = order[:]

    num_restarts = 8
    steps_per_restart = 20000

    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        if r < len(initial_orders):
            start_order = initial_orders[r]
        else:
            start_order = list(rng.permutation(n))

        order, sz = local_search(start_order, steps_per_restart, deadline - 5, rng, use_sa=True)
        if sz > best_size:
            best_size = sz
            best_order = order[:]

    if time.time() < deadline - 2:
        order, sz = local_search(best_order, 50000, deadline - 1, rng, use_sa=True)
        if sz > best_size:
            best_size = sz
            best_order = order[:]

    scores = np.zeros(n, dtype=float)
    for rank, v in enumerate(best_order):
        scores[v] = float(n - rank)

    return scores
