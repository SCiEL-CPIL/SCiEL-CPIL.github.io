import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

N = P['n']
ALL_MASK = (1 << N) - 1

# Precompute full adjacency bitmask and edge count
FULL_ADJ = [0] * N
for u, v in P['edges']:
    FULL_ADJ[u] |= (1 << v)
    FULL_ADJ[v] |= (1 << u)

def biclique_cross(L, R):
    """Return bitmask of cross edges for biclique (L_mask, R_mask)."""
    m = 0
    Lm = L
    while Lm:
        lsb = Lm & (-Lm)
        u = lsb.bit_length() - 1
        m |= R & FULL_ADJ[u]
        Lm ^= lsb
    return m

def is_valid_partition(bicliques):
    """Check bicliques form an exact partition of all graph edges."""
    if not bicliques:
        return False
    covered = 0
    for L, R in bicliques:
        if L == 0 or R == 0:
            return False
        if L & R:
            return False
        cross = biclique_cross(L, R)
        if cross & covered:
            return False
        covered |= cross
    return covered == FULL_ADJ[N-1] if False else True  # placeholder, replaced below

# Better: build full edge mask once
FULL_EDGE_MASK = 0
for u, v in P['edges']:
    a, b = (u, v) if u < v else (v, u)
    FULL_EDGE_MASK |= (1 << a) * N + (1 << b)  # not a bitmask; use tuple set instead

# Use a set of (u,v) with u<v for the full edge set
FULL_EDGE_SET = set()
for u, v in P['edges']:
    a, b = (u, v) if u < v else (v, u)
    FULL_EDGE_SET.add((a, b))

def cross_edge_set(L, R):
    """Return set of (u,v) with u<v for cross edges of biclique."""
    s = set()
    Lm = L
    while Lm:
        lsb = Lm & (-Lm)
        u = lsb.bit_length() - 1
        Rm = R
        while Rm:
            lsb2 = Rm & (-Rm)
            v = lsb2.bit_length() - 1
            if u != v:
                a, b = (u, v) if u < v else (v, u)
                s.add((a, b))
            Rm ^= lsb2
        Lm ^= lsb
    return s

def is_valid_partition(bicliques):
    if not bicliques:
        return False
    covered = set()
    for L, R in bicliques:
        if L == 0 or R == 0:
            return False
        if L & R:
            return False
        # Verify all cross edges exist in graph
        cross = cross_edge_set(L, R)
        for e in cross:
            if e not in FULL_EDGE_SET:
                return False
            if e in covered:
                return False
        covered |= cross
    return covered == FULL_EDGE_SET

def greedy_initial():
    """Greedy maximal biclique set-cover to get a valid initial partition."""
    remaining = set(FULL_EDGE_SET)
    bicliques = []
    while remaining:
        # Find the edge with max potential to form a large biclique
        best_biclique = None
        best_size = 0
        for (u, v) in list(remaining)[:20]:  # sample to save time
            L = {u}
            R = {v}
            changed = True
            while changed:
                changed = False
                # Expand L
                new_L = set(L)
                for w in range(N):
                    if w in new_L:
                        continue
                    ok = True
                    for rv in R:
                        a, b = (w, rv) if w < rv else (rv, w)
                        if (a, b) not in remaining:
                            ok = False
                            break
                    if ok:
                        new_L.add(w)
                if len(new_L) > len(L):
                    L = new_L
                    changed = True
                # Expand R
                new_R = set(R)
                for w in range(N):
                    if w in new_R:
                        continue
                    ok = True
                    for lu in L:
                        a, b = (lu, w) if lu < w else (w, lu)
                        if (a, b) not in remaining:
                            ok = False
                            break
                    if ok:
                        new_R.add(w)
                if len(new_R) > len(R):
                    R = new_R
                    changed = True
            # Convert to masks
            L_mask = 0
            for x in L:
                L_mask |= (1 << x)
            R_mask = 0
            for x in R:
                R_mask |= (1 << x)
            # Ensure no overlap
            R_mask &= ~L_mask
            if R_mask == 0:
                continue
            size = 0
            for lu in L:
                for rv in R:
                    a, b = (lu, rv) if lu < rv else (rv, lu)
                    if (a, b) in remaining:
                        size += 1
            if size > best_size:
                best_size = size
                best_biclique = (L_mask, R_mask)
        if best_biclique is None:
            # Fallback: single edge
            (u, v) = next(iter(remaining))
            L_mask = 1 << u
            R_mask = 1 << v
            best_biclique = (L_mask, R_mask)
        L_mask, R_mask = best_biclique
        # Remove covered edges
        cross = cross_edge_set(L_mask, R_mask)
        remaining -= cross
        bicliques.append((L_mask, R_mask))
    return bicliques

def mask_to_list(mask):
    lst = []
    m = mask
    while m:
        lsb = m & (-m)
        lst.append(lsb.bit_length() - 1)
        m ^= lsb
    return lst

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def time_up():
        return time.time() > deadline

    # Get initial valid solution via greedy
    current = greedy_initial()
    if not is_valid_partition(current):
        # Fallback: one biclique per vertex (seed-like)
        current = []
        for u in range(N):
            R = FULL_ADJ[u]
            if R:
                current.append((1 << u, R))
    best = list(current)
    best_count = len(best)

    # Simulated annealing over the biclique sequence
    T = 1.0
    cooling = 0.999
    T_min = 0.01
    no_improve = 0

    while not time_up() and T > T_min:
        if no_improve > 200:
            # Restart from greedy with random perturbation
            current = greedy_initial()
            no_improve = 0
            T = max(T, 0.5)

        # Choose a random move
        move_type = random.random()
        n_b = len(current)
        if n_b == 0:
            break

        new_bicliques = list(current)

        if move_type < 0.3 and n_b >= 2:
            # Swap two bicliques
            i, j = random.sample(range(n_b), 2)
            new_bicliques[i], new_bicliques[j] = new_bicliques[j], new_bicliques[i]
        elif move_type < 0.5:
            # Delete a biclique
            i = random.randrange(n_b)
            new_bicliques.pop(i)
        elif move_type < 0.7:
            # Insert a new biclique at a random position
            # Pick a random edge and form a small biclique
            remaining = set(FULL_EDGE_SET)
            for L, R in current:
                remaining -= cross_edge_set(L, R)
            if remaining:
                (u, v) = random.choice(list(remaining))
                L_mask = 1 << u
                R_mask = 1 << v
                # Try to expand a bit
                for w in range(N):
                    if w == u or w == v:
                        continue
                    a, b = (w, v) if w < v else (v, w)
                    if (a, b) in remaining:
                        L_mask |= (1 << w)
                for w in range(N):
                    if w == u or w == v:
                        continue
                    a, b = (u, w) if u < w else (w, u)
                    if (a, b) in remaining:
                        R_mask |= (1 << w)
                R_mask &= ~L_mask
                if R_mask:
                    pos = random.randrange(len(new_bicliques) + 1)
                    new_bicliques.insert(pos, (L_mask, R_mask))
        else:
            # Mutate L or R of a random biclique
            i = random.randrange(n_b)
            L, R = new_bicliques[i]
            if random.random() < 0.5:
                # Add or remove a vertex from L
                candidates = [w for w in range(N) if w not in L and w not in R]
                if candidates:
                    w = random.choice(candidates)
                    # Check if adding w to L is valid: w must connect to all R
                    ok = True
                    for rv in mask_to_list(R):
                        a, b = (w, rv) if w < rv else (rv, w)
                        if (a, b) not in FULL_EDGE_SET:
                            ok = False
                            break
                    if ok:
                        L |= (1 << w)
                        new_bicliques[i] = (L, R)
            else:
                candidates = [w for w in range(N) if w not in R and w not in L]
                if candidates:
                    w = random.choice(candidates)
                    ok = True
                    for lu in mask_to_list(L):
                        a, b = (lu, w) if lu < w else (w, lu)
                        if (a, b) not in FULL_EDGE_SET:
                            ok = False
                            break
                    if ok:
                        R |= (1 << w)
                        new_bicliques[i] = (L, R)

        # Validate
        if is_valid_partition(new_bicliques):
            new_count = len(new_bicliques)
            if new_count < best_count:
                best = list(new_bicliques)
                best_count = new_count
                current = list(new_bicliques)
                no_improve = 0
            else:
                delta = new_count - len(current)
                if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                    current = list(new_bicliques)
                    no_improve += 1
                else:
                    no_improve += 1
        else:
            no_improve += 1

        T *= cooling

    # Return best
    result = []
    for L, R in best:
        result.append([mask_to_list(L), mask_to_list(R)])
    return result
