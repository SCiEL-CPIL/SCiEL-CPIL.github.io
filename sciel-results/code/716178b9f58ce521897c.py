"""Search-based construction for edge-colourings of K16 with K4 seeing >=5 colours."""
import random
import time
from itertools import combinations

DEADLINE = 130.0  # safety margin before 150s kill

def _baseline_colours():
    """(i+j) mod 17 colouring, uses 16 colours."""
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

def _count_colours(cols):
    return len(set(cols))

def _is_valid(cols):
    """Check every K4 spans >= 5 distinct colours."""
    for quad in combinations(range(16), 4):
        a, b, c, d = quad
        idx = lambda x, y: x * 15 - x * (x + 1) // 2 + y - x - 1
        colours = set()
        for u, v in combinations(quad, 2):
            colours.add(cols[idx(u, v)])
        if len(colours) < 5:
            return False
    return True

def _edge_index(i, j):
    """Index of edge (i,j) in lexicographic flat array."""
    return i * 15 - i * (i + 1) // 2 + j - i - 1

def construct():
    start = time.time()
    random.seed(42)

    # Start from baseline: 16 colours
    best_cols = _baseline_colours()
    best_count = _count_colours(best_cols)
    assert _is_valid(best_cols), "baseline must be valid"

    # Try to reduce colour count by remapping colours to fewer values.
    # Strategy: take baseline colours, try to merge colour classes.
    # For a target number of colours K, we assign each of the 16 baseline
    # colour-classes to one of K output colours, then check validity.
    # We do a random search over such mappings, plus local search on the
    # final edge colours.

    # First, try simple remappings from 16 down.
    # Baseline colour classes: colours 0..15 (since (i+j)%17 for i<j<16 gives 0..15? let's check)
    # Actually (i+j)%17: i in 0..15, j in i+1..15, i+j ranges 1..29, mod 17 gives 0..16? 
    # i=0,j=17? no j<16. max i+j = 14+15=29, 29%17=12. min=1. So colours 1..16? Let's just compute.
    base_set = set(best_cols)
    # Map baseline colours to 0..len-1
    colour_list = sorted(base_set)
    base_to_idx = {c: k for k, c in enumerate(colour_list)}
    num_base = len(colour_list)
    base_indices = [base_to_idx[c] for c in best_cols]

    # Try target K from 13 upward; find smallest valid K.
    # For each K, we need a function f: {0..num_base-1} -> {0..K-1}
    # such that the resulting edge colouring is valid.
    # This is a constraint satisfaction problem; we use random restarts + local search.

    def try_find_mapping(K, time_budget):
        """Try to find a mapping from num_base colour-classes to K colours that is valid."""
        deadline = time.time() + time_budget
        # Represent mapping as list of length num_base with values in 0..K-1
        # Start with a greedy assignment then do local search.
        for attempt in range(200000):
            if time.time() > deadline:
                return None
            # Random initial mapping
            mapping = [random.randrange(K) for _ in range(num_base)]
            # Build edge colours
            cols = [mapping[bi] for bi in base_indices]
            # Check validity quickly
            if _is_valid(cols):
                return mapping
        return None

    # Try K = 13 first (theoretical lower bound)
    for K in range(13, 17):
        if time.time() > DEADLINE - 30:
            break
        budget = min(15.0, DEADLINE - 30 - (time.time() - start))
        if budget <= 0:
            break
        mapping = try_find_mapping(K, budget)
        if mapping is not None:
            cols = [mapping[bi] for bi in base_indices]
            assert _is_valid(cols)
            best_cols = cols
            best_count = K
            break

    # If we couldn't get below 16, try local search on the full edge colouring
    # to reduce colour count. This is harder; skip if no time.
    if best_count >= 16 and time.time() < DEADLINE - 10:
        # Local search: try to reduce number of distinct colours by
        # recoloring edges.
        # We'll do a simple hill-climbing: repeatedly pick an edge and a new
        # colour, accept if it reduces distinct colours and stays valid.
        # This is expensive; only do a few steps.
        cols = best_cols[:]
        used = set(cols)
        distinct = len(used)
        # Precompute edge list
        edge_list = [(i, j) for i in range(16) for j in range(i + 1, 16)]
        # For speed, we'll limit iterations
        max_iter = 5000
        for it in range(max_iter):
            if time.time() > DEADLINE - 5:
                break
            # Pick random edge
            ei = random.randrange(120)
            old_c = cols[ei]
            # Try a colour from the existing palette or a new one? 
            # To reduce count, we should try colours already in use.
            palette = list(used)
            new_c = random.choice(palette)
            if new_c == old_c:
                continue
            cols[ei] = new_c
            # Check validity
            if _is_valid(cols):
                new_used = set(cols)
                if len(new_used) < distinct:
                    distinct = len(new_used)
                    used = new_used
                # Keep the change even if distinct didn't decrease, to explore
            else:
                cols[ei] = old_c
        if distinct < best_count:
            best_count = distinct
            best_cols = cols

    # Final safety: ensure we return a valid colouring
    if not _is_valid(best_cols):
        best_cols = _baseline_colours()
        best_count = _count_colours(best_cols)

    return best_cols
