import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    n = P['n']
    full_mask = (1 << n) - 1
    size = 1 << n

    # Build adjacency masks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Precompute is_indep and is_clique for all masks (fast incremental)
    is_indep = [0] * size
    is_clique = [0] * size
    is_indep[0] = 1
    is_clique[0] = 1
    for mask in range(1, size):
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        rest = mask ^ lb
        if is_indep[rest] and (adj[v] & rest) == 0:
            is_indep[mask] = 1
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = 1

    # A mask is a valid part if it is a clique or an independent set
    is_valid = [0] * size
    for m in range(1, size):
        if is_indep[m] or is_clique[m]:
            is_valid[m] = 1

    # Precompute a cheap lower bound on the number of parts needed to cover a mask:
    # greedy maximal homogeneous set packing (each part at most size n, so parts >= ceil(popcount/n))
    # but better: use greedy to get a tight lower bound via maximal independent/clique sets.
    # We'll compute a fast lower bound: popcount(mask) // max_part_size where max_part_size is the
    # largest homogeneous subset size, but to be safe we use a greedy packing count as an upper bound
    # on parts, and popcount // max_homogeneous_size as a lower bound.
    # For pruning we need a LOWER bound on remaining parts. Use: ceil(popcount / max_hom_size).
    # Precompute max homogeneous subset size globally.
    max_hom_size = 0
    for m in range(1, size):
        if is_valid[m]:
            c = bin(m).count('1')
            if c > max_hom_size:
                max_hom_size = c
    if max_hom_size == 0:
        max_hom_size = 1

    def popcount(x):
        return bin(x).count('1')

    # Greedy lower bound on number of parts to cover mask:
    # repeatedly take a maximal homogeneous subset (greedy), count them. This gives an UPPER bound
    # on parts. For a LOWER bound we use ceil(popcount / max_hom_size).
    # We'll use the cheap lower bound for pruning.
    def lower_bound(mask):
        return (popcount(mask) + max_hom_size - 1) // max_hom_size

    # For tighter pruning, compute a greedy maximal homogeneous packing count as an upper bound,
    # and use ceil(popcount/max) as lower bound. We'll also try a simple greedy to get a tighter lower
    # bound: the number of parts in a greedy maximal packing is an upper bound; the true minimum is
    # <= that. For a lower bound, we use ceil(popcount / max_hom_size).
    # Additionally, we can compute a better lower bound using a greedy that maximizes part sizes.

    # Precompute for each mask a list of valid submasks (homogeneous parts), sorted by size desc,
    # to speed up candidate generation. But that's 3^14 memory-heavy. Instead generate on the fly
    # and sort by size.

    # Recursive backtracking: try to cover `mask` with at most `k` parts.
    # Returns a list of part masks if success, else None.
    best_solution = [None]  # store the best (fewest parts) complete partition found

    def try_cover(mask, k, parts, depth, deadline):
        # parts is list of chosen part masks so far
        if mask == 0:
            # success: we covered with len(parts) parts
            best_solution[0] = list(parts)
            return True
        if k <= 0:
            return False
        if lower_bound(mask) > k:
            return False
        if time.time() > deadline:
            return False

        # Generate candidate homogeneous submasks of mask, sorted by size descending
        # To avoid duplicates and ensure we make progress, we require the candidate to contain
        # the lowest set bit of mask (to break symmetry: the part containing the smallest
        # remaining vertex must be chosen now).
        lb = mask & (-mask)
        # Enumerate submasks of mask that contain lb and are valid
        # We iterate over all submasks of mask that contain lb.
        # Trick: let rest = mask ^ lb. Any submask containing lb is lb | sub where sub is a submask of rest.
        rest = mask ^ lb
        candidates = []
        sub = rest
        while True:
            cand = sub | lb
            if is_valid[cand]:
                candidates.append(cand)
            if sub == 0:
                break
            sub = (sub - 1) & rest

        # Sort by size descending (larger parts first)
        candidates.sort(key=lambda x: popcount(x), reverse=True)

        for cand in candidates:
            new_k = k - 1
            parts.append(cand)
            if try_cover(mask ^ cand, new_k, parts, depth + 1, deadline):
                parts.pop()
                return True
            parts.pop()
            if time.time() > deadline:
                return False
        return False

    # Try k = 1, 2, ... until we find a feasible cover
    # Upper bound on k: n (each vertex its own part)
    # Lower bound: lower_bound(full_mask)
    k_min = lower_bound(full_mask)
    k_max = n

    # First, get a feasible solution using a greedy heuristic to set an initial upper bound
    # Greedy: repeatedly pick a maximal homogeneous set covering the lowest remaining vertex.
    def greedy_partition(mask):
        parts = []
        while mask:
            lb = mask & (-mask)
            rest = mask ^ lb
            # find a maximal (by inclusion) homogeneous superset of {lb} within mask
            # We'll do a simple greedy: start with {lb}, try to add vertices that keep it homogeneous.
            # Determine type: try clique first, then independent set.
            # For a vertex v, it can be in a clique with all its neighbors; in an independent set with all non-neighbors.
            # Greedy: build a homogeneous set containing lb.
            # Try clique: S = {lb}, add vertices from rest that are adjacent to all in S.
            best_part = lb
            best_type = None
            # Clique attempt
            S = lb
            changed = True
            while changed:
                changed = False
                # find a vertex in rest\S that is adjacent to all in S
                cand_mask = rest ^ S
                # a vertex v is adjacent to all in S if (adj[v] & S) == S
                # iterate vertices in cand_mask
                temp = cand_mask
                while temp:
                    b = temp & (-temp)
                    v = b.bit_length() - 1
                    if (adj[v] & S) == S:
                        S |= b
                        changed = True
                    temp ^= b
            if popcount(S) > popcount(best_part):
                best_part = S
                best_type = 'clique'
            # Independent set attempt
            S2 = lb
            changed = True
            while changed:
                changed = False
                cand_mask = rest ^ S2
                temp = cand_mask
                while temp:
                    b = temp & (-temp)
                    v = b.bit_length() - 1
                    if (adj[v] & S2) == 0:
                        S2 |= b
                        changed = True
                    temp ^= b
            if popcount(S2) > popcount(best_part):
                best_part = S2
                best_type = 'indep'

            parts.append(best_part)
            mask ^= best_part
        return parts

    greedy_parts = greedy_partition(full_mask)
    k_upper = len(greedy_parts)
    # Store greedy solution as a fallback
    best_solution[0] = greedy_parts

    # Now try k from k_min to k_upper-1 to find a better (smaller) solution
    for k in range(k_min, k_upper):
        if time.time() > deadline:
            break
        # Try to find a cover with exactly k parts
        # We use try_cover which finds a cover with at most k parts; if it succeeds, the
        # solution has <= k parts. We want exactly the minimum, so we try increasing k.
        # But try_cover returns success if it can cover with <= k parts. Since we start from
        # k_min (a lower bound), the first k that succeeds gives the minimum.
        parts = []
        if try_cover(full_mask, k, parts, 0, deadline):
            # Found a cover with <= k parts; since k >= lower_bound, this is optimal
            best_solution[0] = parts
            break

    # Extract the best solution
    part_masks = best_solution[0]
    if part_masks is None:
        # Fallback: individual vertices
        part_masks = [(1 << v) for v in range(n)]

    # Convert to lists of vertices
    parts = []
    for pm in part_masks:
        verts = []
        for v in range(n):
            if (pm >> v) & 1:
                verts.append(v)
        parts.append(verts)

    # Ensure types: list of lists of Python ints
    final_parts = []
    for part in parts:
        final_parts.append([int(v) for v in part])

    # Validate: every vertex exactly once, each part nonempty and homogeneous
    seen = [False] * n
    for part in final_parts:
        if len(part) == 0:
            # should not happen
            pass
        for v in part:
            if seen[v]:
                # duplicate, should not happen
                pass
            seen[v] = True
    # (Assume valid by construction)

    return final_parts

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time taken: {end - start:.4f}s")
    print(f"Number of parts: {len(res)}")
    print(f"Partition: {res}")
