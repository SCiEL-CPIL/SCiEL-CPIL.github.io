"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    _t0 = time.time()
    _DEADLINE = _t0 + 130.0  # safety margin before 150 s kill

    n = 16
    # Build edge list in lexicographic order
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    m = len(edges)  # 120

    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx

    # Seed colouring: (i + j) mod 17
    colours = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]

    # Precompute all K4s and their edge indices
    all_k4s = list(combinations(range(n), 4))
    k4_count = len(all_k4s)  # 1820

    # For each K4, store the 6 edge indices
    k4_edges = []
    for quad in all_k4s:
        e_idx = []
        for a, b in combinations(quad, 2):
            e_idx.append(edge_idx[(a, b)])
        k4_edges.append(e_idx)

    # For each colour, store the list of edge indices
    colour_edges = {}
    for c in range(17):
        colour_edges[c] = []
    for idx, c in enumerate(colours):
        colour_edges[c].append(idx)

    # For each edge, store which K4s contain it
    edge_k4s = [[] for _ in range(m)]
    for k4i, e_idx in enumerate(k4_edges):
        for ei in e_idx:
            edge_k4s[ei].append(k4i)

    def compute_tight():
        """Return tight[k4i] = set of colours appearing exactly once in K4 k4i."""
        tight = [set() for _ in range(k4_count)]
        for k4i in range(k4_count):
            cnt = {}
            for ei in k4_edges[k4i]:
                c = colours[ei]
                cnt[c] = cnt.get(c, 0) + 1
            for c, v in cnt.items():
                if v == 1:
                    tight[k4i].add(c)
        return tight

    def compute_essential():
        """Return essential[k4i] = set of colours that appear in K4 k4i."""
        # Actually we need: for a merge a->b, K4 is violated if tight on a and b not in K4.
        # So we need k4_colours[k4i] = set of colours present.
        k4_colours = [set() for _ in range(k4_count)]
        for k4i in range(k4_count):
            for ei in k4_edges[k4i]:
                k4_colours[k4i].add(colours[ei])
        return k4_colours

    tight = compute_tight()
    k4_colours = compute_essential()

    def try_merge(ca, cb):
        """Try to recolour all edges of colour ca to colour cb.
        Returns True if successful (and updates state), False otherwise."""
        nonlocal colours, colour_edges, tight, k4_colours
        if ca == cb:
            return False
        if ca not in colour_edges or not colour_edges[ca]:
            return False

        # Check: for each edge e of colour ca, for each K4 containing e:
        #   if ca is in tight[k4i] (i.e., ca appears exactly once in that K4)
        #   and cb is NOT in k4_colours[k4i], then violation.
        for ei in colour_edges[ca]:
            for k4i in edge_k4s[ei]:
                if ca in tight[k4i] and cb not in k4_colours[k4i]:
                    return False

        # Safe to merge. Update colours.
        for ei in colour_edges[ca]:
            colours[ei] = cb
        colour_edges[cb].extend(colour_edges[ca])
        colour_edges[ca] = []

        # Update tight and k4_colours for affected K4s
        affected = set()
        for ei in colour_edges[cb]:
            if ei >= len(colour_edges[cb]) - len(colour_edges[ca]):
                # This is a newly recoloured edge
                pass
        # Simpler: recompute for all K4s that contain any edge of colour ca or cb
        # Actually, only K4s that had colour ca need updating.
        # Let's just recompute tight and k4_colours for all K4s containing recoloured edges.
        recoloured_edges = set()
        for ei in list(colour_edges[cb]):
            # We need to track which edges were recoloured. Let's just recompute globally
            # for correctness, but that's O(k4_count * 6). Let's be smarter.
            pass

        # For simplicity and correctness, recompute tight and k4_colours for affected K4s.
        # A K4 is affected if it contains at least one edge that was recoloured (was colour ca).
        # We can identify recoloured edges as those in the old colour_edges[ca].
        # But we already cleared colour_edges[ca]. Let's save them first.
        # Actually, let's just recompute everything for affected K4s.
        # The affected K4s are those in edge_k4s[e] for e in old colour_edges[ca].
        # Since we cleared it, let's use a different approach: recompute all.
        # k4_count = 1820, each has 6 edges. That's 10920 operations. Fast enough.
        tight = compute_tight()
        k4_colours = compute_essential()
        return True

    # Local search: try to merge colours to reduce palette size.
    # Start from the highest colour index and try to merge it into others.
    current_max_colour = 16

    while current_max_colour >= 0 and time.time() < _DEADLINE:
        # Try to merge colour current_max_colour into each other colour
        merged = False
        for cb in range(current_max_colour):
            if time.time() >= _DEADLINE:
                break
            if try_merge(current_max_colour, cb):
                merged = True
                break
        if not merged:
            current_max_colour -= 1

    # Return colours in lexicographic edge order
    return colours
# EVOLVE-BLOCK-END
