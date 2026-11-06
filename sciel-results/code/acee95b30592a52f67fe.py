import time
from itertools import combinations

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 115.0  # Leave 5s margin for 120s limit (or 150s if that's the real limit)

# Known good construction: 51 edges on 16 vertices
# This is a known non-2-colorable 5-uniform hypergraph
# Source: Various constructions in the literature on Property B
# Let's use a specific known construction.
# One such construction is based on the projective plane or similar structures.
# However, to be safe, I will hardcode a verified 51-edge construction if I can find one,
# or use a robust heuristic to find one.
#
# Actually, the best known lower bound for m(5) is 51.
# A specific construction with 51 edges exists.
# Let's try to construct one using a greedy approach with a very fast checker,
# or hardcode a known one.
#
# Hardcoding a specific 51-edge hypergraph is risky if I don't have it memorized correctly.
# Instead, I will implement a highly optimized backtracking solver and use a greedy deletion
# strategy starting from a dense hypergraph (like K_11^5 or K_12^5) to reduce edge count.
#
# Key improvement:
# 1. Fast 2-colorability checker using bitsets and unit propagation.
# 2. Greedy deletion with random restarts and edge ordering.
# 3. Local search (edge swap) to potentially find smaller hypergraphs.

def is_2_colorable_fast(n, edges):
    """
    Fast check if hypergraph is 2-colorable.
    Returns True if 2-colorable, False otherwise.
    Uses backtracking with unit propagation and bitsets for speed.
    """
    if not edges:
        return True

    m = len(edges)
    # Precompute edge bitmasks
    # n <= 26, so we can use Python integers as bitsets.
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)

    # Vertex to edge indices
    v_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            v_edges[v].append(i)

    # State:
    # colors: list of -1, 0, 1
    # edge_count_0: count of 0s in each edge
    # edge_count_1: count of 1s in each edge
    # edge_unassigned_mask: bitmask of unassigned vertices in each edge

    colors = [-1] * n
    count_0 = [0] * m
    count_1 = [0] * m
    unassigned_mask = edge_masks[:]  # Initially all vertices unassigned

    def propagate(queue):
        """
        Unit propagation.
        Returns False if contradiction, True otherwise.
        Modifies colors, count_0, count_1, unassigned_mask.
        """
        while queue:
            v, c = queue.pop()
            if colors[v] != -1:
                if colors[v] != c:
                    return False
                else:
                    continue

            colors[v] = c
            # Update state
            for e_idx in v_edges[v]:
                if c == 0:
                    count_0[e_idx] += 1
                else:
                    count_1[e_idx] += 1

                # Remove v from unassigned mask
                unassigned_mask[e_idx] &= ~(1 << v)

                # Check for contradiction
                if count_0[e_idx] == 5 or count_1[e_idx] == 5:
                    return False

                # Unit propagation
                # If 4 are 0, the last unassigned must be 1
                # If 4 are 1, the last unassigned must be 0
                if count_0[e_idx] == 4:
                    # Find the last unassigned vertex
                    # unassigned_mask[e_idx] should have exactly one bit set
                    last_v = (unassigned_mask[e_idx] & -unassigned_mask[e_idx]).bit_length() - 1
                    if colors[last_v] == -1:
                        queue.append((last_v, 1))
                elif count_1[e_idx] == 4:
                    last_v = (unassigned_mask[e_idx] & -unassigned_mask[e_idx]).bit_length() - 1
                    if colors[last_v] == -1:
                        queue.append((last_v, 0))

        return True

    def backtrack():
        # Find unassigned vertex
        unassigned = [v for v in range(n) if colors[v] == -1]
        if not unassigned:
            return True

        # Heuristic: pick vertex with most constraints
        # Impact = number of edges where this vertex is the "last" or "near-last" unassigned
        # Simpler: pick vertex with highest degree in the "residual" hypergraph?
        # Or just pick the one that appears in the most "tight" edges.
        # Let's pick the vertex with the most colored neighbors in its edges.
        best_v = -1
        max_score = -1
        for v in unassigned:
            score = 0
            for e_idx in v_edges[v]:
                # Score based on how close the edge is to being determined
                # If 4 are colored, it's very tight.
                colored = 5 - (unassigned_mask[e_idx].bit_count())
                score += colored * colored  # Heuristic weight
            if score > max_score:
                max_score = score
                best_v = v

        if best_v == -1:
            return True

        # Try color 0
        # Save state
        old_colors = colors[:]
        old_count_0 = count_0[:]
        old_count_1 = count_1[:]
        old_unassigned = unassigned_mask[:]

        q = [(best_v, 0)]
        if propagate(q):
            if backtrack():
                return True

        # Restore state
        colors[:] = old_colors
        count_0[:] = old_count_0
        count_1[:] = old_count_1
        unassigned_mask[:] = old_unassigned

        # Try color 1
        old_colors = colors[:]
        old_count_0 = count_0[:]
        old_count_1 = count_1[:]
        old_unassigned = unassigned_mask[:]

        q = [(best_v, 1)]
        if propagate(q):
            if backtrack():
                return True

        return False

    return backtrack()


def construct():
    start_time = time.time()

    # Start with a dense hypergraph.
    # K_11^5 has C(11,5) = 462 edges. Too many.
    # K_10^5 has C(10,5) = 252 edges.
    # K_9^5 has C(9,5) = 126 edges.
    # Let's start with K_10^5 and try to reduce.
    # Or K_11^5 but only a random subset?
    # Actually, starting with a smaller n might be better if we can keep it non-2-colorable.
    # But n=9, K_9^5 is 126 edges. We need to get to ~51.
    # Let's try n=11, start with a random subset of K_11^5 of size ~100, then reduce.
    # Or just start with K_10^5.

    n = 10
    all_edges = [list(e) for e in combinations(range(n), 5)]

    best_edges = all_edges[:]
    best_score = len(best_edges)

    # Greedy deletion
    # We will try multiple random orderings.

    for restart in range(50):
        if time.time() - start_time > TIME_LIMIT - 10:
            break

        current_edges = all_edges[:]
        import random
        random.shuffle(current_edges)

        # Greedy deletion
        i = 0
        while i < len(current_edges):
            if time.time() - start_time > TIME_LIMIT - 5:
                break

            edge_to_test = current_edges[i]
            # Remove it
            remaining = current_edges[:i] + current_edges[i+1:]

            # Check if remaining is 2-colorable
            # If it IS 2-colorable, we MUST keep the edge.
            # If it IS NOT 2-colorable, we can remove it.

            # Optimization: Only check if the edge is "critical"
            # We just check every time.

            is_col = is_2_colorable_fast(n, remaining)

            if not is_col:
                # It is still non-2-colorable without this edge.
                # So we remove it permanently.
                current_edges = remaining
                # i stays the same, because the next edge is now at index i
            else:
                # It became 2-colorable. We must keep it.
                i += 1

        # Update best
        if len(current_edges) < best_score:
            best_score = len(current_edges)
            best_edges = current_edges[:]

            # If we hit the known lower bound or close to it, we can stop or try harder
            if best_score <= 51:
                # We found a very good solution.
                pass

    return {'n': n, 'edges': best_edges}

if __name__ == "__main__":
    # Self-test
    res = construct()
    print(f"Found hypergraph with {len(res['edges'])} edges on {res['n']} vertices.")
    # Verify
    print("Verifying...")
    valid = not is_2_colorable_fast(res['n'], res['edges'])
    print(f"Is non-2-colorable: {valid}")
