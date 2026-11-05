import random
import time
import sys

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    d = 7
    n = 1 << d
    total_edges = n * d  # 448

    # Generate all edges
    edges = []
    edge_set = set()
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
                edge_set.add((x, y))

    # Precompute all squares and edge-to-square mapping
    edge_to_squares = {e: [] for e in edges}
    squares = []  # list of tuples of 4 edges

    for i in range(d):
        for j in range(i + 1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            for base in range(n):
                if (base & mask_i) == 0 and (base & mask_j) == 0:
                    v0 = base
                    v1 = base ^ mask_i
                    v2 = base ^ mask_j
                    v3 = base ^ mask_i ^ mask_j
                    e0 = (min(v0, v1), max(v0, v1))
                    e1 = (min(v0, v2), max(v0, v2))
                    e2 = (min(v1, v3), max(v1, v3))
                    e3 = (min(v2, v3), max(v2, v3))
                    sq = (e0, e1, e2, e3)
                    squares.append(sq)
                    for e in sq:
                        edge_to_squares[e].append(len(squares) - 1)

    # Precompute: for each edge, the set of edges that "block" it
    # (i.e., edges that share a square with it, excluding itself)
    # An edge e_add is blocked by chosen if there exists a square containing e_add
    # where all other 3 edges are in chosen.
    # So for e_add, it's blocked iff any of its squares has 3 other edges all in chosen.
    # We can precompute for each edge e_add, the list of (square_idx, other_3_edges)
    edge_blockers = {}
    for e in edges:
        blockers = []
        for sq_idx in edge_to_squares[e]:
            sq = squares[sq_idx]
            others = tuple(ed for ed in sq if ed != e)
            blockers.append((sq_idx, others))
        edge_blockers[e] = blockers

    def is_edge_blocked(chosen, e):
        """Check if edge e cannot be added to chosen (would create a C4)"""
        for sq_idx, others in edge_blockers[e]:
            if all(o in chosen for o in others):
                return True
        return False

    # For faster swap: if we remove e_remove from chosen, which edges become unblocked?
    # An edge e_add is blocked by a square where all 3 others are in chosen.
    # After removing e_remove, e_add is unblocked if e_remove was in all blocking squares.
    # Actually, e_add is unblocked if for EVERY square containing e_add, at least one
    # of the other 3 edges is NOT in (chosen - {e_remove}).
    # So e_add was blocked only because of e_remove if:
    #   - e_add is blocked by chosen
    #   - for every square containing e_add that has all 3 others in chosen,
    #     e_remove is one of those others.
    # Simplification: e_add is unblocked by removing e_remove if:
    #   - e_remove shares a square with e_add
    #   - e_add is currently blocked
    #   - every blocking square of e_add contains e_remove among its other edges

    # Precompute: for each pair of edges (a, b) that share a square, store this
    # Actually, let's compute on the fly for speed.

    def get_unblocked_by_removal(chosen, e_remove):
        """Return list of edges not in chosen that become addable if e_remove is removed"""
        result = []
        # Only edges that share a square with e_remove can be affected
        candidate_edges = set()
        for sq_idx in edge_to_squares[e_remove]:
            sq = squares[sq_idx]
            for e in sq:
                if e != e_remove:
                    candidate_edges.add(e)

        for e_add in candidate_edges:
            if e_add in chosen:
                continue
            # Check if e_add is currently blocked
            blocked = False
            all_blockers_contain_remove = True
            for sq_idx, others in edge_blockers[e_add]:
                if all(o in chosen for o in others):
                    blocked = True
                    if e_remove not in others:
                        all_blockers_contain_remove = False
                        break
            if blocked and all_blockers_contain_remove:
                result.append(e_add)
        return result

    # Greedy initialization with random edge ordering
    def greedy_init(seed):
        rng = random.Random(seed)
        edge_order = list(edges)
        rng.shuffle(edge_order)
        chosen = set()
        for e in edge_order:
            if not is_edge_blocked(chosen, e):
                chosen.add(e)
        return chosen

    # Local search with fast targeted swaps
    def local_search(chosen, rng, time_limit):
        """Improve chosen set via local search"""
        improved = True
        while improved and time.time() < time_limit:
            improved = False

            # Phase 1: Try adding edges (pure additions)
            edges_to_try = [e for e in edges if e not in chosen]
            rng.shuffle(edges_to_try)
            for e in edges_to_try:
                if time.time() >= time_limit:
                    break
                if not is_edge_blocked(chosen, e):
                    chosen.add(e)
                    improved = True

            # Phase 2: Targeted 1-for-1 swaps
            # For each edge in chosen, find edges that would be unblocked by removing it
            edges_in = list(chosen)
            rng.shuffle(edges_in)

            for e_remove in edges_in:
                if time.time() >= time_limit:
                    break
                unblocked = get_unblocked_by_removal(chosen, e_remove)
                if not unblocked:
                    continue
                # Try to add one of the unblocked edges
                rng.shuffle(unblocked)
                for e_add in unblocked:
                    # Verify: after removing e_remove, is e_add still addable?
                    # (It should be, but let's double-check)
                    new_chosen = chosen - {e_remove}
                    if not is_edge_blocked(new_chosen, e_add):
                        chosen.discard(e_remove)
                        chosen.add(e_add)
                        improved = True
                        break
                if improved:
                    break

            # Phase 3: 2-for-1 swaps (remove one edge, add two)
            edges_in = list(chosen)
            rng.shuffle(edges_in)
            for e_remove in edges_in:
                if time.time() >= time_limit:
                    break
                unblocked = get_unblocked_by_removal(chosen, e_remove)
                if len(unblocked) < 2:
                    continue
                # Try pairs of unblocked edges
                rng.shuffle(unblocked)
                found = False
                for i in range(len(unblocked)):
                    if time.time() >= time_limit:
                        break
                    for j in range(i + 1, len(unblocked)):
                        e_add1 = unblocked[i]
                        e_add2 = unblocked[j]
                        new_chosen = chosen - {e_remove}
                        if not is_edge_blocked(new_chosen, e_add1) and not is_edge_blocked(new_chosen | {e_add1}, e_add2):
                            chosen.discard(e_remove)
                            chosen.add(e_add1)
                            chosen.add(e_add2)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

        return chosen

    # Run multiple random restarts
    best_chosen = set()
    best_size = 0

    # More restarts since local search is faster
    num_restarts = 40
    for restart in range(num_restarts):
        if time.time() >= deadline - 40:  # leave time for final local search
            break

        seed = restart * 12345 + 67890
        chosen = greedy_init(seed)
        size = len(chosen)

        if size > best_size:
            best_size = size
            best_chosen = set(chosen)

        # Local search on this solution - give it more time
        rng = random.Random(seed)
        local_time = min(deadline - 35, time.time() + 15)
        chosen = local_search(chosen, rng, local_time)
        size = len(chosen)

        if size > best_size:
            best_size = size
            best_chosen = set(chosen)

    # Final polish on best solution
    if time.time() < deadline - 5 and len(best_chosen) > 0:
        rng = random.Random(99999)
        best_chosen = local_search(best_chosen, rng, deadline - 2)

    # Convert to list of [u, v]
    result = [[u, v] for u, v in best_chosen]
    return result

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Number of edges: {len(result)}")
