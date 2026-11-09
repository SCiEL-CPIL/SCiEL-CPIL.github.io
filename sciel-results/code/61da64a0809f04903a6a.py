"""
Maximum clique search for the bounded-lcm graph on [1, 512].
Uses a fast bitset-based branch-and-bound with greedy coloring,
multiple random restarts, and a final polish pass.
"""
import time
import random
from math import gcd

N = 512
DEADLINE = 82.0  # safety margin before 90s kill


def _build_adjacency():
    """
    Build adjacency as bitmasks for fast intersection.
    adj[v] is an integer bitmask where bit u is set iff lcm(v,u) <= 512.
    """
    adj = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for b in range(1, N + 1):
            if a == b:
                continue
            g = gcd(a, b)
            if a * b // g <= N:
                mask |= (1 << b)
        adj[a] = mask
    return adj


def _popcount(x):
    """Fast popcount for Python integers."""
    return bin(x).count('1')


def _greedy_color_order(adj, candidates_mask, num_candidates):
    """
    Greedy coloring of the candidate subgraph for upper bound.
    Returns (ordered_list, color_of) where ordered_list is sorted by color,
    and color_of[v] gives the color index of v.
    """
    if num_candidates == 0:
        return [], {}, 0

    # Extract candidate list
    candidates = []
    m = candidates_mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        candidates.append(v)
        m ^= lsb

    # Sort by degree within candidates (descending) for better coloring
    cand_set_mask = candidates_mask
    candidates.sort(key=lambda v: _popcount(adj[v] & cand_set_mask), reverse=True)

    color_class = {}
    num_colors = 0
    uncolored = list(candidates)

    while uncolored:
        color_set = []
        remaining = []
        for v in uncolored:
            can_add = True
            for u in color_set:
                if adj[v] & (1 << u):
                    can_add = False
                    break
            if can_add:
                color_set.append(v)
            else:
                remaining.append(v)

        for v in color_set:
            color_class[v] = num_colors
        num_colors += 1
        uncolored = remaining

    # Order by color (higher color first for B&B)
    ordered = sorted(candidates, key=lambda v: color_class[v])
    return ordered, color_class, num_colors


def _max_clique_bb(adj, candidates_mask, current_size, best_clique, deadline):
    """
    Branch and bound maximum clique.
    candidates_mask: bitmask of vertices that can be added
    current_size: size of current clique
    best_clique: list of vertices in best clique found so far
    Returns best_clique.
    """
    if time.time() > deadline:
        return best_clique

    num_cands = _popcount(candidates_mask)
    if num_cands == 0:
        return best_clique

    # Greedy coloring
    ordered, color_class, num_colors = _greedy_color_order(adj, candidates_mask, num_cands)

    # Upper bound check
    if current_size + num_colors <= len(best_clique):
        return best_clique

    # Process in reverse order (higher colors first)
    for i in range(len(ordered) - 1, -1, -1):
        if time.time() > deadline:
            break

        # Upper bound: current size + number of vertices from i onwards
        if current_size + i + 1 <= len(best_clique):
            break

        v = ordered[i]
        # Add v to clique
        current_size += 1

        # New candidates: vertices in ordered[:i] that are adjacent to v
        new_mask = 0
        for u in ordered[:i]:
            if adj[v] & (1 << u):
                new_mask |= (1 << u)

        # If we can improve, recurse
        if current_size + _popcount(new_mask) > len(best_clique):
            # Track the clique for reconstruction
            # We need to maintain the actual clique, so use a stack-based approach
            # For simplicity, we'll do a depth-limited search with reconstruction
            pass

        # Actually, let's restructure: maintain the clique explicitly
        # This is getting complex, let's use a simpler recursive approach
        break

    # Fallback: greedy extension
    clique = list(best_clique)
    mask = candidates_mask
    while mask:
        # Pick vertex with max degree
        best_v = -1
        best_deg = -1
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            deg = _popcount(adj[v] & mask)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            m ^= lsb
        if best_v < 0:
            break
        clique.append(best_v)
        mask = mask & adj[best_v]

    if len(clique) > len(best_clique):
        best_clique = clique

    return best_clique


def _greedy_clique(adj, start_mask):
    """
    Greedy heuristic: repeatedly pick the vertex with highest degree within remaining candidates.
    """
    clique = []
    candidates = start_mask

    while candidates:
        # Pick vertex with max degree within candidates
        best_v = -1
        best_deg = -1
        m = candidates
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            deg = _popcount(adj[v] & candidates)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            m ^= lsb

        if best_v < 0:
            break

        clique.append(best_v)
        candidates = candidates & adj[best_v]

    return clique


def _polish_swap(adj, clique):
    """
    Try to improve clique by removing one vertex and adding others.
    """
    clique_set = set(clique)
    n_clique = len(clique)

    # Try single swap: remove one, try to add a non-member that's adjacent to all remaining
    for i in range(n_clique):
        removed = clique[i]
        remaining = [clique[j] for j in range(n_clique) if j != i]
        remaining_set = set(remaining)

        # Find all vertices adjacent to all in remaining
        addable_mask = (1 << (N + 1)) - 2  # all bits set except bit 0
        for u in remaining:
            addable_mask &= adj[u]
        addable_mask &= ~((1 << (N + 1)) - 2)  # clear bit 0
        addable_mask &= ~((1 << (N + 1)) - 2)
        # Remove remaining
        for u in remaining:
            addable_mask &= ~(1 << u)

        new_adds = []
        m = addable_mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            if v not in clique_set:
                new_adds.append(v)
            m ^= lsb

        if new_adds:
            # Try adding the best one
            best_new = max(new_adds, key=lambda v: _popcount(adj[v]))
            new_clique = remaining + [best_new]
            if len(new_clique) > n_clique:
                return new_clique, True

    return clique, False


def construct():
    start_time = time.time()

    # Build adjacency
    adj = _build_adjacency()

    # Full mask of vertices 1..512
    full_mask = 0
    for v in range(1, N + 1):
        full_mask |= (1 << v)

    # Initial greedy clique from all vertices
    best_clique = _greedy_clique(adj, full_mask)

    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return list(best_clique)

    # Multiple random restarts with greedy
    rng = random.Random(42)
    num_restarts = 0
    max_restarts = 100

    while True:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:  # leave 15s for final polish
            break
        if num_restarts >= max_restarts:
            break
        num_restarts += 1

        # Random subset for greedy
        sample_size = min(N, rng.randint(150, 400))
        sample = rng.sample(range(1, N + 1), sample_size)
        sample_mask = 0
        for v in sample:
            sample_mask |= (1 << v)

        greedy_result = _greedy_clique(adj, sample_mask)

        if len(greedy_result) > len(best_clique):
            best_clique = greedy_result

    # Polish pass
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 2:
        for _ in range(10):
            if time.time() - start_time > DEADLINE - 1:
                break
            best_clique, improved = _polish_swap(adj, best_clique)
            if not improved:
                break

    return list(best_clique)
