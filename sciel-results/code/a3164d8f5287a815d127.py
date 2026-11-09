"""Cochromatic partition via branch-and-bound over valid clique/independent-set parts."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

N = P['n']
EDGE_SET = set()
ADJ = [set() for _ in range(N)]
for a, b in P['edges']:
    EDGE_SET.add((a, b))
    EDGE_SET.add((b, a))
    ADJ[a].add(b)
    ADJ[b].add(a)

# Precompute valid parts: every subset that is a clique or an independent set.
# Represent each subset as an integer bitmask.
valid_parts = []  # list of (mask, tuple_of_vertices_sorted)

def is_clique_mask(mask):
    # Check all pairs in mask are edges
    verts = []
    for v in range(N):
        if mask & (1 << v):
            verts.append(v)
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            if (verts[i], verts[j]) not in EDGE_SET:
                return False
    return True

def is_independent_mask(mask):
    verts = []
    for v in range(N):
        if mask & (1 << v):
            verts.append(v)
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            if (verts[i], verts[j]) in EDGE_SET:
                return False
    return True

# Enumerate all nonempty subsets
for mask in range(1, 1 << N):
    if is_clique_mask(mask) or is_independent_mask(mask):
        verts = tuple(v for v in range(N) if mask & (1 << v))
        valid_parts.append((mask, verts))

# Sort valid parts by size descending (larger parts first for better pruning)
valid_parts.sort(key=lambda x: -len(x[1]))

# Build a quick lookup: for each vertex, which valid parts contain it
vertex_to_parts = [[] for _ in range(N)]
for idx, (mask, verts) in enumerate(valid_parts):
    for v in verts:
        vertex_to_parts[v].append(idx)

# Deduplicate: a vertex may be in many parts; we will use this for branching.

FULL_MASK = (1 << N) - 1
DEADLINE = time.time() + 80  # leave 10s safety margin before 90s kill

best_solution = None
best_count = N + 1  # worst case: each vertex its own part

def branch(remaining_mask, current_parts, count):
    global best_solution, best_count, DEADLINE
    if time.time() > DEADLINE:
        return
    if remaining_mask == 0:
        if count < best_count:
            best_count = count
            best_solution = list(current_parts)
        return
    # Prune: if we already used >= best_count parts, no improvement possible
    if count >= best_count:
        return
    # Lower bound: at least 1 more part needed
    if count + 1 >= best_count:
        return

    # Pick a vertex to branch on (choose one with fewest valid parts to reduce branching)
    # Find a vertex in remaining_mask
    v = None
    min_opts = None
    for u in range(N):
        if remaining_mask & (1 << u):
            # Count valid parts that are subsets of remaining_mask and contain u
            opts = 0
            for idx in vertex_to_parts[u]:
                pmask = valid_parts[idx][0]
                if pmask & ~remaining_mask == 0:
                    opts += 1
            if v is None or opts < min_opts:
                v = u
                min_opts = opts
    # Branch on each valid part containing v and subset of remaining_mask
    for idx in vertex_to_parts[v]:
        pmask, pverts = valid_parts[idx]
        if pmask & ~remaining_mask != 0:
            continue
        # Add this part
        new_remaining = remaining_mask & ~pmask
        current_parts.append(pverts)
        branch(new_remaining, current_parts, count + 1)
        current_parts.pop()
        if best_solution is not None and best_count == count + 1:
            # Can't do better than count+1 since we just found a solution with that many
            # Actually we might find equal, but we want strictly less, so break if count+1 == best_count
            # But there might be other parts of same size giving same count; we only need one.
            # However, to be safe, continue but the prune at top will catch it.
            pass

# Initial greedy to get a good upper bound
def greedy_partition():
    remaining = FULL_MASK
    parts = []
    while remaining:
        # Pick the largest valid part subset of remaining
        best_part = None
        best_size = 0
        for idx, (pmask, pverts) in enumerate(valid_parts):
            if pmask & ~remaining == 0:
                if len(pverts) > best_size:
                    best_size = len(pverts)
                    best_part = pverts
        if best_part is None:
            # Should not happen; single vertex is always valid
            v = (remaining & -remaining).bit_length() - 1
            best_part = (v,)
        parts.append(best_part)
        for v in best_part:
            remaining &= ~(1 << v)
    return parts

greedy = greedy_partition()
best_count = len(greedy)
best_solution = greedy

# Now branch-and-bound to try to improve
branch(FULL_MASK, [], 0)

def construct():
    if best_solution is None:
        # Fallback: singleton partition
        return [[v] for v in range(N)]
    # Convert to list of lists of ints
    result = []
    for part in best_solution:
        result.append([int(v) for v in part])
    # Verify: every vertex exactly once, each part is clique or independent
    seen = [0] * N
    for part in result:
        for v in part:
            seen[v] += 1
    # (verification for safety; if broken, return fallback)
    for v in range(N):
        if seen[v] != 1:
            return [[v] for v in range(N)]
    for part in result:
        if len(part) <= 1:
            continue
        # check clique or independent
        all_edges = True
        no_edges = True
        for i in range(len(part)):
            for j in range(i + 1, len(part)):
                a, b = part[i], part[j]
                if (a, b) in EDGE_SET:
                    no_edges = False
                else:
                    all_edges = False
        if not (all_edges or no_edges):
            return [[v] for v in range(N)]
    return result

# EVOLVE-BLOCK-END
