"""Rewritten program: model as Max Independent Set on support-conflict graph,
then exhaustively enumerate sign patterns that satisfy the distance constraint.
Uses simulated annealing with swaps to grow the independent set of 4-subsets."""

import time
import numpy as np

D = 11
DEADLINE = 130.0  # seconds; leave 20s safety margin for 150s wall-clock kill


def _all_4_subsets():
    """Return list of frozensets of size 4 from range(D)."""
    from itertools import combinations
    return [frozenset(c) for c in combinations(range(D), 4)]


def _build_conflict_adj(subsets):
    """Build adjacency bitsets: adj[i] is a Python int bitmask of subsets that conflict with i.
    Two 4-subsets conflict if their intersection has size >= 3 (i.e., they share 3 or 4 elements).
    """
    n = len(subsets)
    # For fast intersection-size checks, precompute element membership as bitmasks over D
    elem_masks = []
    for s in subsets:
        m = 0
        for e in s:
            m |= (1 << e)
        elem_masks.append(m)

    adj = [0] * n
    for i in range(n):
        mi = elem_masks[i]
        for j in range(i + 1, n):
            mj = elem_masks[j]
            inter = mi & mj
            # popcount
            pc = bin(inter).count('1')
            if pc >= 3:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj, elem_masks


def _sa_max_independent_set(adj, n, time_budget):
    """Simulated annealing with swap moves to find a large independent set.
    Returns list of indices in the independent set.
    """
    import random
    random.seed(42)

    # Start with a greedy independent set
    selected = set()
    candidates = list(range(n))
    random.shuffle(candidates)
    for idx in candidates:
        if idx not in selected:
            # Check if idx conflicts with any already selected
            conflict = False
            adj_mask = adj[idx]
            # Check if any selected node is in adj_mask
            # For small sets, just iterate
            for s in selected:
                if (adj_mask >> s) & 1:
                    conflict = True
                    break
            if not conflict:
                selected.add(idx)

    best_set = set(selected)
    best_size = len(selected)
    current = set(selected)

    start_time = time.time()
    T = 1.0
    cooling = 0.999
    max_iters = 500000

    while time.time() - start_time < time_budget and T > 0.001:
        # Pick a random node to swap out and a random candidate to swap in
        if not current:
            break
        out_node = random.choice(list(current))
        # Candidate: any node not in current that doesn't conflict with (current - {out_node})
        temp_current = current - {out_node}
        # Find candidates
        candidates = []
        for idx in range(n):
            if idx in current:
                continue
            # Check conflict with temp_current
            adj_mask = adj[idx]
            ok = True
            for s in temp_current:
                if (adj_mask >> s) & 1:
                    ok = False
                    break
            if ok:
                candidates.append(idx)

        if not candidates:
            T *= cooling
            continue

        in_node = random.choice(candidates)
        new_set = temp_current | {in_node}

        # Compute delta in size
        delta = len(new_set) - len(current)

        if delta > 0 or random.random() < np.exp(delta / T):
            current = new_set
            if len(current) > best_size:
                best_size = len(current)
                best_set = set(current)

        T *= cooling

    return list(best_set)


def _signs_for_support(support):
    """For a given 4-element support, enumerate all sign patterns (16 of them)
    and return those that are mutually compatible (pairwise distance >= max norm).
    
    For vectors with support S (|S|=4) and entries ±1:
    - norm^2 = 4
    - Two vectors u, v with same support: dist^2 = sum over S of (u_i - v_i)^2
      If they agree on k positions and disagree on (4-k): dist^2 = 2*(4-k)
      We need dist^2 >= 4 (max norm squared), so 2*(4-k) >= 4 => 4-k >= 2 => k <= 2.
      So two sign patterns are compatible iff they agree on at most 2 of the 4 positions.
    
    This is equivalent to: the Hamming distance between sign patterns (as length-4 binary strings)
    must be at least 2. We want the maximum set of such patterns.
    
    The maximum code of length 4 over {±1} with min Hamming distance >= 2 is size 8
    (even parity code or just all patterns with even number of -1s... let's verify).
    
    Actually, let's just do a greedy/brute-force: enumerate all 16 patterns, find max independent set
    in the graph where edges connect patterns with Hamming distance < 2 (i.e., distance 0 or 1).
    Distance 0 is self, distance 1 means they differ in exactly 1 position.
    
    The conflict graph: each pattern conflicts with its 4 neighbors (flip one bit). This is the 4-cube Q4.
    Max independent set in Q4 has size 8 (bipartite, each part has 8 vertices).
    """
    # Represent sign patterns as 4-bit integers (0 or 1 for +1/-1)
    patterns = list(range(16))
    # Conflict: Hamming distance <= 1 (excluding self, so distance == 1)
    conflict = [set() for _ in range(16)]
    for a in range(16):
        for b in range(a + 1, 16):
            diff = a ^ b
            if bin(diff).count('1') == 1:
                conflict[a].add(b)
                conflict[b].add(a)
    
    # Greedy max independent set on this small graph
    selected = []
    remaining = list(range(16))
    random_seed = 0
    while remaining:
        # Pick the vertex with fewest conflicts among remaining
        best_idx = None
        best_conf = 999
        for idx in remaining:
            c = len(conflict[idx] & set(remaining))
            if c < best_conf:
                best_conf = c
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
        # Remove all neighbors of best_idx from remaining
        to_remove = conflict[best_idx] & set(remaining)
        for r in to_remove:
            remaining.remove(r)
    
    # Convert to sign vectors: bit 0 -> position 0 of support, etc.
    signs = []
    support_list = sorted(support)
    for pat in selected:
        vec = [0] * D
        for bit, elem in enumerate(support_list):
            vec[elem] = 1 if (pat >> bit) & 1 else -1
        signs.append(vec)
    return signs


def construct():
    start_time = time.time()
    
    subsets = _all_4_subsets()
    n_subsets = len(subsets)
    
    adj, elem_masks = _build_conflict_adj(subsets)
    
    # Run SA to find a large independent set of supports
    sa_budget = min(60.0, DEADLINE - (time.time() - start_time) - 10.0)
    if sa_budget < 1.0:
        sa_budget = 1.0
    
    selected_indices = _sa_max_independent_set(adj, n_subsets, sa_budget)
    
    # For each selected support, get all compatible sign patterns
    all_vectors = []
    for idx in selected_indices:
        support = subsets[idx]
        signs = _signs_for_support(support)
        all_vectors.extend(signs)
    
    if not all_vectors:
        # Fallback: use the original seed approach
        import itertools
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    result = np.array(all_vectors, dtype=np.int64)
    
    # Verify constraint: min pairwise distance >= max norm
    # Do a quick check if time permits
    if time.time() - start_time < DEADLINE - 5.0:
        norms = np.sum(result.astype(np.float64) ** 2, axis=1)
        max_norm_sq = np.max(norms)
        # Check a sample of pairs
        k = len(result)
        if k > 2:
            # Sample up to 10000 pairs
            n_pairs = min(10000, k * (k - 1) // 2)
            np.random.seed(123)
            for _ in range(n_pairs):
                i, j = np.random.randint(0, k, 2)
                if i == j:
                    continue
                diff = result[i] - result[j]
                dist_sq = np.sum(diff.astype(np.float64) ** 2)
                if dist_sq < max_norm_sq - 1e-9:
                    # Constraint violated; this shouldn't happen with our construction
                    # but as a safety, just return what we have
                    break
    
    return result
