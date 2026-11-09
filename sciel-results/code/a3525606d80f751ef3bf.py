import random
import time
import itertools

def _max_sidon_size(S):
    """Exhaustively compute largest subset of S with all unordered pair sums
    (including diagonals, i.e., a+b for a<=b in subset) unique."""
    S = list(S)
    n = len(S)
    best = 0
    # Precompute all pair sums for the full set for quick reference not needed;
    # just check each subset.
    for mask in range(1 << n):
        # count bits
        if bin(mask).count('1') <= best:
            continue
        subset = [S[i] for i in range(n) if mask & (1 << i)]
        k = len(subset)
        # check all unordered pair sums including diagonals unique
        sums = set()
        ok = True
        for i in range(k):
            for j in range(i, k):
                s = subset[i] + subset[j]
                if s in sums:
                    ok = False
                    break
                sums.add(s)
            if not ok:
                break
        if ok:
            best = k
    return best

def _collision_heuristic(S):
    """Lower bound on how 'non-Sidon' the set is: count sum collisions.
    More collisions => likely smaller max Sidon subset."""
    S = list(S)
    sums = {}
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            sums[s] = sums.get(s, 0) + 1
    # total number of pairs that are in a collision
    coll = 0
    for c in sums.values():
        if c > 1:
            coll += c
    return coll

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    best_set = list(range(12))
    best_val = _max_sidon_size(best_set)
    
    rng = random.Random(12345)
    
    # Initial diverse candidates
    candidates = []
    # 1) seed
    candidates.append(list(range(12)))
    # 2) dense low range
    candidates.append(list(range(0, 13)))
    # 3) two clusters
    candidates.append(list(range(0, 6)) + list(range(20, 26)))
    # 4) structured collision: pairs sharing sums
    #    {0,1,2,3, 10,11,12,13, 20,21,22,23}
    candidates.append(list(range(0,4)) + list(range(10,14)) + list(range(20,24)))
    # 5) arithmetic progression with common difference that creates collisions
    candidates.append([i*2 for i in range(12)])
    # 6) quadratic residues-ish
    candidates.append(sorted(set((i*i) % 129 for i in range(129)) | set(range(12))))
    candidates[5] = sorted(candidates[5])[:12]
    
    # Evaluate initial candidates
    for cand in candidates:
        if len(cand) != 12 or len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        v = _max_sidon_size(cand)
        if v < best_val:
            best_val = v
            best_set = list(cand)
    
    # Local search: swap elements to increase collisions
    def _try_improve(current):
        # Try random swaps and perturbations
        for _ in range(200):
            new = list(current)
            op = rng.random()
            if op < 0.5:
                # swap one element
                i = rng.randrange(12)
                new[i] = rng.randint(0, 128)
            elif op < 0.8:
                # replace two elements
                i, j = rng.sample(range(12), 2)
                new[i] = rng.randint(0, 128)
                new[j] = rng.randint(0, 128)
            else:
                # shift a cluster
                i = rng.randrange(12)
                new[i] = (new[i] + rng.randint(-3, 3)) % 129
            if len(set(new)) != 12:
                continue
            if any(x < 0 or x > 128 for x in new):
                continue
            # Use collision heuristic as fast filter
            h = _collision_heuristic(new)
            hc = _collision_heuristic(current)
            if h < hc - 2:  # need significant improvement in collisions
                continue
            v = _max_sidon_size(new)
            if v < best_val:
                best_val = v
                best_set = list(new)
                current = list(new)
        return current
    
    # Main search loop with deadline guard
    current = list(best_set)
    iterations = 0
    while time.time() < deadline:
        current = _try_improve(current)
        iterations += 1
        # Occasionally restart from a random candidate
        if iterations % 5 == 0:
            if rng.random() < 0.3:
                # random set
                current = rng.sample(range(129), 12)
            elif rng.random() < 0.5:
                # two-cluster random
                c1 = rng.sample(range(0, 40), 6)
                c2 = rng.sample(range(60, 129), 6)
                current = sorted(c1 + c2)
    
    return best_set
