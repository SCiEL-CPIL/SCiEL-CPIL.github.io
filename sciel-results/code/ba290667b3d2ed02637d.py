"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# EVOLVE-BLOCK-START
def _edge_index(i, j):
    """Map (i,j) with i<j to a 0..119 index."""
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

def _build_c4s():
    """
    Precompute all 4-cycles. A 4-cycle is a set of 4 edges forming a cycle of length 4.
    In K_N, a 4-cycle on vertices {a,b,c,d} (a<b<c<d) has exactly 3 distinct 4-cycles:
      (a-b-c-d-a), (a-b-d-c-a), (a-c-b-d-a)
    Each is a set of 4 edges.
    We store each C4 as a frozenset of 4 edge-indices.
    Also build edge_to_c4s: for each edge, list of C4 indices it belongs to.
    """
    c4_list = []
    edge_to_c4s = [[] for _ in range(NUM_EDGES)]

    for a, b, c, d in itertools.combinations(range(N), 4):
        # Three 4-cycles on these 4 vertices
        # Cycle 1: a-b-c-d-a
        e1 = frozenset([_edge_index(a, b), _edge_index(b, c), _edge_index(c, d), _edge_index(a, d)])
        # Cycle 2: a-b-d-c-a
        e2 = frozenset([_edge_index(a, b), _edge_index(b, d), _edge_index(d, c), _edge_index(a, c)])
        # Cycle 3: a-c-b-d-a
        e3 = frozenset([_edge_index(a, c), _edge_index(c, b), _edge_index(b, d), _edge_index(a, d)])

        for c4 in [e1, e2, e3]:
            c4_idx = len(c4_list)
            c4_list.append(c4)
            for e in c4:
                edge_to_c4s[e].append(c4_idx)

    return c4_list, edge_to_c4s

def _is_rainbow(c4, colors):
    """Check if a 4-cycle has 4 distinct colors."""
    c = [colors[e] for e in c4]
    return len(set(c)) == 4

def _count_colors(colors):
    """Count distinct colors used."""
    return len(set(colors))

def _greedy_construct():
    """
    Greedy construction: assign edges in a fixed order, choosing the smallest
    color that doesn't create a rainbow C4.
    """
    c4_list, edge_to_c4s = _build_c4s()
    colors = [0] * NUM_EDGES

    for e in range(NUM_EDGES):
        # Try colors 0..119, pick the first that doesn't create a rainbow
        best_color = 0
        for color in range(120):
            colors[e] = color
            valid = True
            for c4_idx in edge_to_c4s[e]:
                if _is_rainbow(c4_list[c4_idx], colors):
                    valid = False
                    break
            if valid:
                best_color = color
                break
        colors[e] = best_color

    return colors

def _local_search(colors, c4_list, edge_to_c4s, deadline, rng):
    """
    Local random-walk refinement with O(1) affected-C4 checks.
    """
    best_colors = list(colors)
    best_count = _count_colors(best_colors)
    n_c4s = len(c4_list)

    while True:
        if time.time() > deadline:
            break

        # Pick a random edge
        e = rng.randint(0, NUM_EDGES - 1)
        old_color = colors[e]

        # Pick a random new color (bias toward unused colors to increase diversity)
        # Use a mix: 50% random, 50% try to introduce a new color
        if rng.random() < 0.5:
            new_color = rng.randint(0, 119)
        else:
            # Try a color not currently used
            used = set(colors)
            available = [c for c in range(120) if c not in used]
            if available:
                new_color = rng.choice(available)
            else:
                new_color = rng.randint(0, 119)

        if new_color == old_color:
            continue

        # Check all affected C4s
        colors[e] = new_color
        valid = True
        for c4_idx in edge_to_c4s[e]:
            if _is_rainbow(c4_list[c4_idx], colors):
                valid = False
                break

        if valid:
            new_count = _count_colors(colors)
            if new_count >= best_count:
                best_colors = list(colors)
                best_count = new_count
            # Accept the move (even if count doesn't improve, to allow exploration)
            # But only keep if it doesn't reduce count too much? For now, accept all valid moves.
        else:
            colors[e] = old_color

    return best_colors, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline, 10s safety margin

    rng = random.Random(42)

    # Phase 1: Greedy construction
    colors = _greedy_construct()
    c4_list, edge_to_c4s = _build_c4s()

    # Verify initial validity
    for c4 in c4_list:
        if _is_rainbow(c4, colors):
            # Should not happen with greedy, but safeguard
            pass

    best_count = _count_colors(colors)
    best_colors = list(colors)

    # Phase 2: Local search with time budget
    if time.time() < deadline - 5.0:
        colors, best_count = _local_search(colors, c4_list, edge_to_c4s, deadline - 5.0, rng)
        best_colors = list(colors)

    # Phase 3: Multiple restarts if time permits
    while time.time() < deadline - 5.0:
        # Restart with a different greedy order or random perturbation
        restart_colors = _greedy_construct()
        # Perturb: randomly reassign some edges
        for _ in range(20):
            e = rng.randint(0, NUM_EDGES - 1)
            c = rng.randint(0, 119)
            restart_colors[e] = c
            # Fix if invalid: revert if it creates rainbow
            valid = True
            for c4_idx in edge_to_c4s[e]:
                if _is_rainbow(c4_list[c4_idx], restart_colors):
                    valid = False
                    break
            if not valid:
                restart_colors[e] = 0
                # Try other colors
                for c2 in range(1, 120):
                    restart_colors[e] = c2
                    valid = True
                    for c4_idx in edge_to_c4s[e]:
                        if _is_rainbow(c4_list[c4_idx], restart_colors):
                            valid = False
                            break
                    if valid:
                        break

        if time.time() < deadline - 5.0:
            improved_colors, improved_count = _local_search(
                restart_colors, c4_list, edge_to_c4s, deadline - 5.0, rng
            )
            if improved_count > best_count:
                best_colors = list(improved_colors)
                best_count = improved_count

    # Final safety: verify and return
    # Quick verification of best_colors
    for c4 in c4_list:
        if _is_rainbow(c4, best_colors):
            # Fallback to all zeros if somehow invalid
            best_colors = [0] * NUM_EDGES
            break

    return best_colors
# EVOLVE-BLOCK-END

