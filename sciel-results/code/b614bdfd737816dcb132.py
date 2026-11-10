import math
import time
from itertools import combinations

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255,
    with gcd(a, b) > 1 for all distinct pairs.
    Maximizes cardinality.
    """
    N = 255
    deadline = time.time() + 58.0

    # Candidate pool: numbers in [1,255] divisible by 3, 5, or 17
    candidates = [i for i in range(1, N + 1)
                  if i % 3 == 0 or i % 5 == 0 or i % 17 == 0]
    n = len(candidates)
    idx_of = {v: i for i, v in enumerate(candidates)}

    # Ensure 255 is in the pool
    if 255 not in idx_of:
        candidates.append(255)
        idx_of[255] = n
        n += 1

    # Build adjacency using bitsets (Python ints)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    def bit_count(x):
        return bin(x).count('1')

    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    # Branch and bound maximum clique (Tomita algorithm)
    def max_clique(P, best_clique):
        # P is a set of vertices (as a bitmask)
        # best_clique is the current best clique (list of vertex indices)
        nonlocal best_size, best_clique_found
        if P == 0:
            if len(best_clique) > best_size:
                best_size = len(best_clique)
                best_clique_found = list(best_clique)
            return

        # Bound: if we can't beat best, prune
        if len(best_clique) + popcount(P) <= best_size:
            return

        # Ordering: color bound
        # Simple greedy coloring of P for upper bound
        # We'll use a simple approach: repeatedly pick a vertex, remove it and its neighbors
        # For a tighter bound, use greedy coloring
        # Let's implement a basic color bound
        remaining = P
        color = 0
        color_classes = []
        while remaining != 0:
            color += 1
            # Greedy: pick vertices that can share a color (non-adjacent)
            # Simple: take first vertex, then add non-adjacent ones
            c = 0
            first = (remaining & -remaining).bit_length() - 1
            c |= (1 << first)
            remaining &= ~(1 << first)
            temp = remaining
            while temp != 0:
                v = (temp & -temp).bit_length() - 1
                if (adj[v] & c) == 0:
                    c |= (1 << v)
                    remaining &= ~(1 << v)
                temp &= ~(1 << v)
            color_classes.append(c)

        # Upper bound is the number of colors
        if len(best_clique) + color <= best_size:
            return

        # Branch: pick a vertex from the last color class
        # Order vertices by color (higher color = more promising)
        # We'll branch on vertices in the last color class first
        for c_idx in range(len(color_classes) - 1, -1, -1):
            if len(best_clique) + (c_idx + 1) <= best_size:
                break
            # Branch on vertices in color_classes[c_idx]
            cc = color_classes[c_idx]
            while cc != 0:
                if time.time() > deadline:
                    return
                v = (cc & -cc).bit_length() - 1
                cc &= ~(1 << v)
                # Include v
                best_clique.append(v)
                # New candidates: neighbors of v that are in P
                new_P = P & adj[v]
                max_clique(new_P, best_clique)
                best_clique.pop()
                # Exclude v from P
                P &= ~(1 << v)
                if len(best_clique) + popcount(P) <= best_size:
                    break

    best_size = 0
    best_clique_found = []

    # Start with all vertices
    full_mask = (1 << n) - 1
    max_clique(full_mask, [])

    if best_size == 0:
        # Fallback: all multiples of 3
        result = [i for i in range(1, N + 1) if i % 3 == 0]
        if 255 not in result:
            result.append(255)
        return result

    result = [candidates[i] for i in best_clique_found]

    # Ensure 255 is present
    if 255 not in result:
        result.append(255)

    return result
