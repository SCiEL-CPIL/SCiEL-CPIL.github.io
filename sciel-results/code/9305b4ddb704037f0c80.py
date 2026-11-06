"""Multi-restart search for 32 integers in 1..512 maximizing distinct 3-AP differences.

Combines:
  * structural idea (inspiration): greedy density-guided initialization +
    diverse move operators (1/2/3-swap) + multi-restart population.
  * refinement machinery (current): tight explicit deadline guard, fast bitset
    scoring, and a final local-search polish.
"""

def construct():
    import time
    import random

    start_time = time.time()
    # Hard wall-clock limit is 150s; keep a generous safety margin so we
    # always return a valid result before the evaluator kills the process.
    deadline = start_time + 130.0

    # Bitset representation: value x in 1..512 -> bit (x-1).
    # A 3-AP difference d exists iff (b & (b >> d) & (b >> (2*d))) != 0.
    def score_bitset(b):
        count = 0
        for d in range(1, 256):
            if (b & (b >> d) & (b >> (2 * d))):
                count += 1
        return count

    def set_from_bitset(b):
        s = []
        i = 0
        while b:
            if b & 1:
                s.append(i + 1)
            b >>= 1
            i += 1
        return s

    def bitset_from_set(S):
        b = 0
        for x in S:
            b |= 1 << (x - 1)
        return b

    # Density of each position across all 3-APs: how many (a,a+d,a+2d)
    # triples contain position p. Used for greedy seeding.
    def compute_density():
        density = [0] * 513
        for d in range(1, 256):
            for a in range(1, 513 - 2 * d):
                density[a] += 1
                density[a + d] += 1
                density[a + 2 * d] += 1
        return density

    density = compute_density()

    def greedy_init(rng, target=32):
        """Build a seed set by repeatedly adding the element with best marginal gain
        (sampled over a random candidate subset for speed)."""
        current = set()
        bit = 0
        score = 0
        candidates = list(range(1, 513))
        rng.shuffle(candidates)
        for _ in range(target):
            if not candidates:
                break
            sample_size = min(60, len(candidates))
            sample = rng.sample(candidates, sample_size)
            best_elem = None
            best_gain = -1
            for elem in sample:
                if elem in current:
                    continue
                nb = bit | (1 << (elem - 1))
                gain = score_bitset(nb) - score
                if gain > best_gain:
                    best_gain = gain
                    best_elem = elem
            if best_elem is None:
                best_elem = candidates.pop()
                current.add(best_elem)
                bit |= 1 << (best_elem - 1)
                score = score_bitset(bit)
                continue
            current.add(best_elem)
            bit |= 1 << (best_elem - 1)
            score += best_gain
            candidates.remove(best_elem)
        return list(current), bit, score

    def anneal(init_set, init_bit, init_score, rng, budget):
        """Simulated annealing with mixed 1/2/3-swap moves from a given start."""
        cur = list(init_set)
        bit = init_bit
        score = init_score
        best = list(cur)
        best_bit = bit
        best_score = score
        temp = 30.0
        cooling = 0.99997
        min_temp = 0.01
        e = 2.718281828459045

        while temp > min_temp:
            if time.time() > budget:
                break
            r = rng.random()
            if r < 0.6:
                i = rng.randrange(32)
                old = cur[i]
                new = rng.randint(1, 512)
                if new == old or new in cur:
                    continue
                nb = (bit & ~(1 << (old - 1))) | (1 << (new - 1))
                ns = score_bitset(nb)
                delta = ns - score
                if delta > 0 or rng.random() < (e ** (delta / temp)):
                    cur[i] = new
                    bit, score = nb, ns
                    if score > best_score:
                        best_score, best, best_bit = score, list(cur), nb
            elif r < 0.8:
                i = rng.randrange(32)
                j = rng.randrange(32)
                if i == j:
                    continue
                o1, o2 = cur[i], cur[j]
                n1 = rng.randint(1, 512)
                n2 = rng.randint(1, 512)
                if n1 in cur or n2 in cur or n1 == n2:
                    continue
                nb = bit
                nb &= ~(1 << (o1 - 1))
                nb &= ~(1 << (o2 - 1))
                nb |= (1 << (n1 - 1))
                nb |= (1 << (n2 - 1))
                ns = score_bitset(nb)
                delta = ns - score
                if delta > 0 or rng.random() < (e ** (delta / temp)):
                    cur[i], cur[j] = n1, n2
                    bit, score = nb, ns
                    if score > best_score:
                        best_score, best, best_bit = score, list(cur), nb
            else:
                i = rng.randrange(32)
                j = rng.randrange(32)
                k = rng.randrange(32)
                if i == j or i == k or j == k:
                    continue
                o1, o2, o3 = cur[i], cur[j], cur[k]
                n1 = rng.randint(1, 512)
                n2 = rng.randint(1, 512)
                n3 = rng.randint(1, 512)
                if n1 in cur or n2 in cur or n3 in cur:
                    continue
                if len({n1, n2, n3}) < 3:
                    continue
                nb = bit
                nb &= ~(1 << (o1 - 1))
                nb &= ~(1 << (o2 - 1))
                nb &= ~(1 << (o3 - 1))
                nb |= (1 << (n1 - 1))
                nb |= (1 << (n2 - 1))
                nb |= (1 << (n3 - 1))
                ns = score_bitset(nb)
                delta = ns - score
                if delta > 0 or rng.random() < (e ** (delta / temp)):
                    cur[i], cur[j], cur[k] = n1, n2, n3
                    bit, score = nb, ns
                    if score > best_score:
                        best_score, best, best_bit = score, list(cur), nb
            temp *= cooling
        return best, best_bit, best_score

    def polish(bit, score, rng, budget):
        """Pure hill-climbing local search to squeeze out last improvements."""
        best_bit, best_score = bit, score
        for _ in range(20000):
            if time.time() > budget:
                break
            cur = set_from_bitset(best_bit)
            x_out = rng.choice(cur)
            x_in = rng.randint(1, 512)
            if x_in in cur:
                continue
            nb = (best_bit & ~(1 << (x_out - 1))) | (1 << (x_in - 1))
            ns = score_bitset(nb)
            if ns > best_score:
                best_score, best_bit = ns, nb
        return best_bit, best_score

    # Multi-restart population: each restart gets a fresh greedy seed and its
    # own annealing budget; we keep the global best across all restarts.
    global_best_bit = 0
    global_best_score = -1
    rng = random.Random(12345)

    # First restart gets a slightly larger budget to establish a strong baseline.
    first_budget = start_time + 45.0
    seed_set, seed_bit, seed_score = greedy_init(rng)
    b, bb, bs = anneal(seed_set, seed_bit, seed_score, rng, first_budget)
    global_best_bit, global_best_score = bb, bs

    # Subsequent restarts share the remaining time.
    while time.time() < deadline - 20.0:
        seed_set, seed_bit, seed_score = greedy_init(rng)
        budget = min(deadline - 8.0, time.time() + 18.0)
        b, bb, bs = anneal(seed_set, seed_bit, seed_score, rng, budget)
        if bs > global_best_score:
            global_best_score, global_best_bit = bs, bb

    # Final local-search polish on the global best.
    if time.time() < deadline - 5.0:
        global_best_bit, global_best_score = polish(
            global_best_bit, global_best_score, rng, deadline - 3.0
        )

    return set_from_bitset(global_best_bit)
