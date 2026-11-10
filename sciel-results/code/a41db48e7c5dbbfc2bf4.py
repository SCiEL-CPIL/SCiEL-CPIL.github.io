"""Edge-coloring of K16 avoiding rainbow 4-cycles; maximize distinct colors used."""
import time
import random

N = 16
M = N * (N - 1) // 2  # 120 edges
MAXC = 120

# Precompute all 4-cycles (3 per 4-vertex subset) as edge-index triples
# Edge index: for i<j, idx = i*N - i*(i+1)//2 + (j - i - 1)
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

# Build list of 4-cycles: each is a tuple of 4 edge indices
# For a 4-set {a,b,c,d} with a<b<c<d, the three 4-cycles are:
#   (a,b,c,d): edges ab, bc, cd, da
#   (a,b,d,c): edges ab, bd, dc, ca
#   (a,c,b,d): edges ac, cb, bd, da
CYCLES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                CYCLES.append((edge_idx(a, b), edge_idx(b, c), edge_idx(c, d), edge_idx(d, a)))
                CYCLES.append((edge_idx(a, b), edge_idx(b, d), edge_idx(d, c), edge_idx(c, a)))
                CYCLES.append((edge_idx(a, c), edge_idx(c, b), edge_idx(b, d), edge_idx(d, a)))

NUM_CYCLES = len(CYCLES)

# For each edge, list of (cycle_index, position) it participates in
EDGE_CYCLES = [[] for _ in range(M)]
for ci, cyc in enumerate(CYCLES):
    for pos, e in enumerate(cyc):
        EDGE_CYCLES[e].append((ci, pos))

def count_violations(colors):
    """Count number of 4-cycles with 4 distinct colors."""
    v = 0
    for cyc in CYCLES:
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        if len((c0, c1, c2, c3)) == 4:
            v += 1
    return v

def is_valid(colors):
    return count_violations(colors) == 0

def distinct_count(colors):
    return len(set(colors))

def make_valid_seed(rng):
    """Build a valid coloring with some diversity using a random 2-coloring of a
    carefully chosen structure, then try to expand."""
    # Start with a random assignment using 2 colors based on parity of a random
    # permutation - this tends to be valid or near-valid.
    perm = list(range(N))
    rng.shuffle(perm)
    # Map original vertex to position
    pos = [0] * N
    for i, v in enumerate(perm):
        pos[v] = i
    colors = [0] * M
    for i in range(N):
        for j in range(i + 1, N):
            # color by parity of (pos[i] + pos[j])
            colors[edge_idx(i, j)] = (pos[i] + pos[j]) % 2
    return colors

def try_grow_diversity(colors, rng, deadline, target_colors=40):
    """Greedy: try to introduce new colors on edges, accepting only if still valid."""
    used = set(colors)
    best = list(colors)
    best_distinct = len(used)
    # Candidate new colors
    all_colors = list(range(MAXC))
    # We'll iterate: pick a random edge, try assigning a random new color,
    # check validity incrementally (full check is fine for M=120, ~2184 cycles)
    attempts = 0
    while time.time() < deadline and len(used) < target_colors:
        # Pick an edge
        e = rng.randrange(M)
        # Pick a new color not yet used (or a random color)
        if rng.random() < 0.7:
            # Try a brand new color
            new_c = rng.randrange(MAXC)
            if new_c in used:
                continue
        else:
            new_c = rng.randrange(MAXC)
        old_c = colors[e]
        colors[e] = new_c
        if is_valid(colors):
            used.add(new_c)
            if len(used) > best_distinct:
                best_distinct = len(used)
                best = list(colors)
        else:
            colors[e] = old_c
        attempts += 1
    return best, best_distinct

def local_search(colors, rng, deadline):
    """Local search: try single-edge color changes to increase distinct count
    while maintaining validity. Also try to fix violations if any."""
    best = list(colors)
    best_distinct = distinct_count(colors)
    cur = list(colors)
    cur_viol = count_violations(cur)
    cur_distinct = best_distinct
    used = set(cur)
    stall = 0
    while time.time() < deadline:
        e = rng.randrange(M)
        old_c = cur[e]
        # Decide target color
        r = rng.random()
        if cur_viol > 0:
            # Try to reduce violations: pick a color from a neighbor edge
            # or a random color
            if r < 0.5:
                new_c = rng.randrange(MAXC)
            else:
                # copy a color from a random other edge
                new_c = cur[rng.randrange(M)]
        else:
            # valid: try to increase diversity
            if r < 0.6:
                new_c = rng.randrange(MAXC)
            else:
                new_c = rng.randrange(MAXC)
        if new_c == old_c:
            continue
        cur[e] = new_c
        new_viol = count_violations(cur)
        new_distinct = distinct_count(cur) if new_viol == 0 else 0
        # Accept if it improves: fewer violations, or same violations but more distinct
        if new_viol < cur_viol:
            cur_viol = new_viol
            if new_viol == 0:
                cur_distinct = new_distinct
                used = set(cur)
                if cur_distinct > best_distinct:
                    best_distinct = cur_distinct
                    best = list(cur)
                stall = 0
            else:
                stall = 0
        elif new_viol == cur_viol and cur_viol == 0:
            if new_distinct > cur_distinct:
                cur_distinct = new_distinct
                used = set(cur)
                if cur_distinct > best_distinct:
                    best_distinct = cur_distinct
                    best = list(cur)
                stall = 0
            else:
                # accept with small probability to escape (simulated anneal-ish)
                if rng.random() < 0.1:
                    cur_distinct = new_distinct
                    used = set(cur)
                else:
                    cur[e] = old_c
        else:
            cur[e] = old_c
        stall += 1
        if stall > 5000:
            # restart from best
            cur = list(best)
            cur_viol = count_violations(cur)
            cur_distinct = distinct_count(cur)
            used = set(cur)
            stall = 0
    return best, best_distinct

def construct():
    start = time.time()
    deadline = start + 80  # safety margin before 90s kill
    rng = random.Random(12345)

    best = None
    best_distinct = 0

    # Try multiple random seeds
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        seed_colors = make_valid_seed(rng)
        # Fix any violations with a quick local repair
        seed_colors, d = local_search(seed_colors, rng, min(deadline, time.time() + 5))
        if d > best_distinct:
            best_distinct = d
            best = seed_colors
        # Now grow diversity
        grown, d2 = try_grow_diversity(list(seed_colors), rng, min(deadline, time.time() + 10), target_colors=60)
        if d2 > best_distinct:
            best_distinct = d2
            best = grown
        # Further local search on the grown coloring
        final, d3 = local_search(list(grown), rng, deadline)
        if d3 > best_distinct:
            best_distinct = d3
            best = final

    if best is None:
        # Fallback: all zeros is always valid
        return [0] * M

    # Ensure validity
    if not is_valid(best):
        # repair
        repaired, _ = local_search(list(best), rng, deadline)
        if is_valid(repaired):
            best = repaired
        else:
            return [0] * M

    # Clamp colors to 0..119
    best = [c % MAXC for c in best]
    # Final validity check
    if not is_valid(best):
        return [0] * M
    return best
