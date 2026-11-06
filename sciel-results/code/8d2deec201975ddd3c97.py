"""
Design: Greedy edge insertion with global color-constraint tracking,
combined with a Lagrangian-relaxation-inspired adaptive weight scheme.

We build a dense graph on 20 vertices, edge-coloured with 20 colours,
such that every 4-cycle is rainbow (all four edges have distinct colours).

Key structural insight:
- A 4-cycle is a set of 4 edges forming a cycle of length 4.
- The constraint is: for every 4-cycle (a,b,c,d,a), the colours of
  (a,b), (b,c), (c,d), (d,a) must be all distinct.

Strategy:
1. Start with a complete graph K20 (190 edges) and try to assign colours.
2. If K20 is infeasible (likely), greedily add edges one by one, choosing
   the edge and colour that causes the fewest future conflicts.
3. Use a conflict-counting heuristic: for each candidate (edge, colour),
   count how many existing 4-cycles would become non-rainbow if we add it.
4. Adaptive weighting: maintain a "pressure" vector over colours; when a
   colour is over-used in local structures, increase its penalty so the
   greedy step avoids it, mimicking subgradient updates on a Lagrangian.

To stay within time limits, we cap the search and return the best valid
solution found. We use numpy for fast array operations where helpful.
"""

import time
import random
import numpy as np

N = 20
NUM_COLOURS = 20
DEADLINE = 115.0  # safety margin before 120s kill


def construct():
    start = time.time()
    best_edges = None
    best_count = 0

    # Precompute all 4-cycles in K20 as tuples of 4 vertex indices (a,b,c,d)
    # A 4-cycle is (a,b,c,d) with a<b<c<d not required; we enumerate all
    # 4-cycles as ordered 4-tuples where edges are (a,b),(b,c),(c,d),(d,a)
    # and all four vertices are distinct, and we consider each cycle once
    # by fixing a as the smallest vertex and b < c or b > c to avoid dups.
    # Actually, to keep it simple and correct, enumerate all 4-cycles as
    # combinations of 4 vertices, and for each set of 4 vertices, there are
    # 3 distinct 4-cycles (up to rotation and reversal).
    # For 4 vertices {w,x,y,z}, the 3 cycles are:
    #   (w,x,y,z), (w,x,z,y), (w,y,x,z)
    # We'll store each cycle as a frozenset of the 4 edges to deduplicate.

    # Actually, let's just enumerate all 4-cycles directly.
    # A 4-cycle is defined by 4 distinct vertices a,b,c,d and the cycle
    # a-b-c-d-a. We enumerate all ordered 4-tuples of distinct vertices
    # but canonicalize: fix a = min, then b,c,d are the other three in
    # some order. For each set of 4 vertices, there are 3! = 6 orderings,
    # but cycles (a,b,c,d) and (a,d,c,b) are the same cycle (reversed),
    # and rotations are the same. So 6/2 = 3 distinct cycles per 4-set.

    all_cycles = []
    seen_cycle_edges = set()
    vertices = list(range(N))
    from itertools import combinations, permutations

    for quad in combinations(vertices, 4):
        w, x, y, z = quad
        # 3 distinct 4-cycles on these 4 vertices
        cycles = [
            (w, x, y, z),
            (w, x, z, y),
            (w, y, x, z),
        ]
        for cyc in cycles:
            a, b, c, d = cyc
            edges = frozenset([
                frozenset((a, b)),
                frozenset((b, c)),
                frozenset((c, d)),
                frozenset((d, a)),
            ])
            if edges not in seen_cycle_edges:
                seen_cycle_edges.add(edges)
                all_cycles.append(cyc)

    # Precompute for each edge (as frozenset) which cycles it participates in
    edge_to_cycles = {}
    for idx, cyc in enumerate(all_cycles):
        a, b, c, d = cyc
        for e in [frozenset((a, b)), frozenset((b, c)), frozenset((c, d)), frozenset((d, a))]:
            edge_to_cycles.setdefault(e, []).append(idx)

    # All possible edges
    all_edges_list = []
    for i in range(N):
        for j in range(i + 1, N):
            all_edges_list.append(frozenset((i, j)))

    # Random seed for reproducibility within a run
    rng = random.Random(42)

    # Greedy construction with adaptive colour pressure
    # State: edge_colour dict mapping frozenset -> colour
    # We try multiple random orderings and pick the best.

    def greedy_build(edge_order, pressure_init=None):
        """
        Build a graph by adding edges in the given order.
        For each edge, try all colours and pick the one with minimum
        conflict increase (weighted by pressure).
        Returns list of [u, v, c] for the valid graph built.
        """
        if pressure_init is None:
            pressure = np.zeros(NUM_COLOURS)
        else:
            pressure = pressure_init.copy()

        edge_colour = {}  # frozenset -> colour
        # Track for each cycle, the set of colours used on its edges so far
        # cycle_colours[i] = set of colours assigned to edges in cycle i that are present
        cycle_colours = [set() for _ in all_cycles]
        # Also track which edges of each cycle are present
        cycle_edge_present = [set() for _ in all_cycles]

        result_edges = []

        for e in edge_order:
            if time.time() - start > DEADLINE:
                break
            best_colour = None
            best_score = float('inf')
            best_conflict = 0

            for c in range(NUM_COLOURS):
                # Count how many cycles containing e would become non-rainbow
                conflicts = 0
                for ci in edge_to_cycles.get(e, []):
                    # Check if adding (e, c) makes cycle ci non-rainbow
                    # A cycle is non-rainbow if any two edges have the same colour
                    # Currently cycle ci has some edges present with colours in cycle_colours[ci]
                    # If we add colour c, and c is already in cycle_colours[ci], conflict
                    # Also, if after adding, the cycle is complete (all 4 edges present)
                    # and any two edges share a colour, it's a conflict.
                    # But we only need to check: does c appear in cycle_colours[ci]?
                    # If yes, adding this edge with colour c creates a duplicate.
                    if c in cycle_colours[ci]:
                        conflicts += 1
                    # Also check: if the cycle becomes complete with this edge,
                    # verify all 4 colours are distinct. But the above check (c in
                    # existing colours) already catches duplicates. If c is not in
                    # existing colours, and the cycle completes, we need all 4 distinct.
                    # Since c is new to this cycle's colour set, and existing colours
                    # are all distinct (we maintain invariant), the new set is distinct.
                    # Wait - we need to maintain the invariant that existing colours
                    # in each cycle are all distinct. If we only add when c is not
                    # in cycle_colours[ci], then the invariant holds.

                # Weighted score: conflicts + pressure[c] * small factor
                score = conflicts + 0.01 * pressure[c]

                if score < best_score:
                    best_score = score
                    best_colour = c
                    best_conflict = conflicts

            if best_colour is not None and best_conflict == 0:
                # Add the edge with the best colour
                edge_colour[e] = best_colour
                u, v = tuple(e)
                result_edges.append([min(u, v), max(u, v), best_colour])

                # Update cycle data structures
                for ci in edge_to_cycles.get(e, []):
                    cycle_colours[ci].add(best_colour)
                    cycle_edge_present[ci].add(e)

                # Update pressure: increase pressure for colours that are
                # heavily used in dense areas (subgradient-like update)
                # Simple heuristic: increase pressure for the chosen colour slightly
                pressure[best_colour] += 0.1
                # And decrease for others slightly to allow rebalancing
                # (This is a simplified subgradient step)

        return result_edges

    # Try multiple random edge orderings
    num_trials = 0
    while time.time() - start < DEADLINE - 5:
        # Create a random edge order, biased towards "central" edges first
        # (edges with high degree in the remaining graph)
        # Simple approach: random shuffle
        edge_order = all_edges_list[:]
        rng.shuffle(edge_order)

        edges = greedy_build(edge_order)
        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = edges

        num_trials += 1

        # If we've found a very dense graph, we can stop early
        if best_count >= 100:
            break

    # Final validation of best_edges
    if best_edges is None:
        best_edges = []

    # Validate: check all 4-cycles in the constructed graph are rainbow
    # Build edge_colour lookup
    ec = {}
    for u, v, c in best_edges:
        ec[frozenset((u, v))] = c

    # Check each cycle
    valid = True
    for cyc in all_cycles:
        a, b, c, d = cyc
        e1 = frozenset((a, b))
        e2 = frozenset((b, c))
        e3 = frozenset((c, d))
        e4 = frozenset((d, a))
        # Only check if all 4 edges are present
        if e1 in ec and e2 in ec and e3 in ec and e4 in ec:
            cols = [ec[e1], ec[e2], ec[e3], ec[e4]]
            if len(set(cols)) != 4:
                valid = False
                break

    if not valid:
        # Fallback: return a star (always valid)
        best_edges = [[0, v, 0] for v in range(1, N)]

    return best_edges
