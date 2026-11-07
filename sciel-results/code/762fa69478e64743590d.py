import time
import numpy as np
from itertools import product

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin for 60s limit

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)

    # Identify edge types
    bottom_edges = []  # edges in bottom cycle
    top_edges = []     # edges in top cycle
    match_edges = []   # matching edges

    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            bottom_edges.append(i)
        elif u >= m and v >= m:
            top_edges.append(i)
        else:
            match_edges.append(i)

    def compute_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        try:
            eigvals = np.linalg.eigvalsh(A)
            return np.max(np.abs(eigvals))
        except:
            return np.inf

    def is_valid_lift(signs):
        """Check if the lift is connected and 3-regular."""
        try:
            from autoresearch_suite.science_tcs_expansion import check_lift
            radius, details = check_lift(graph, signs)
            return True, radius
        except Exception:
            return False, np.inf

    # Gauge invariance: fix a spanning tree to +1
    # The prism graph has 2m vertices and 3m edges.
    # A spanning tree has 2m-1 edges, so there are 3m - (2m-1) = m+1 chords.
    #
    # Construct a spanning tree:
    # - Bottom cycle: edges (0,1), (1,2), ..., (m-2, m-1) -> m-1 edges
    # - Top cycle: edges (m, m+1), (m+1, m+2), ..., (2m-2, 2m-1) -> m-1 edges
    # - Matching: all m edges (i, i+m) -> m edges
    # Total: (m-1) + (m-1) + m = 3m - 2... that's too many.
    #
    # Wait, let me recount. The prism graph P_m has:
    # - 2m vertices
    # - 3m edges (m bottom + m top + m matching)
    # A spanning tree on 2m vertices has 2m-1 edges.
    # Number of chords = 3m - (2m-1) = m+1.
    #
    # Let's construct a specific spanning tree:
    # - Bottom cycle: keep m-1 edges (break the cycle by removing one edge)
    # - Top cycle: keep m-1 edges (break the cycle by removing one edge)
    # - Matching: keep all m edges
    # Total: (m-1) + (m-1) + m = 3m - 2. That's not 2m-1.
    #
    # I need to be more careful. Let me think about this differently.
    #
    # Actually, for a prism graph, a natural spanning tree is:
    # - Take all m matching edges
    # - Take m-1 edges from the bottom cycle (e.g., all except the last)
    # - Take 0 edges from the top cycle
    # Total: m + (m-1) + 0 = 2m-1. 
    #
    # Or alternatively:
    # - Take all m matching edges
    # - Take m-1 edges from the bottom cycle
    # - Take 0 from top
    # That gives 2m-1 edges and connects all 2m vertices (the matching connects top to bottom, and the bottom cycle edges connect all bottom vertices, so all top vertices are connected via their matching edge to the bottom).
    #
    # So the chords (non-tree edges) are:
    # - The remaining 1 edge from the bottom cycle
    # - All m edges from the top cycle
    # Total: m+1 chords.
    #
    # Let's identify which edges are in the spanning tree:
    # We need to know the actual edge ordering in the graph.

    # Let's identify the specific edges
    # Bottom cycle edges: these connect vertices in 0..m-1
    # Top cycle edges: these connect vertices in m..2m-1
    # Matching edges: one in 0..m-1, one in m..2m-1

    # For the spanning tree, I'll use:
    # - All matching edges (m edges)
    # - First m-1 bottom cycle edges
    # - No top cycle edges
    # Chords: last bottom cycle edge + all m top cycle edges = m+1 chords

    # But I need to handle the edge ordering carefully.
    # Let's just use a different approach: since we're fixing the gauge,
    # we can fix ANY spanning tree. Let's use a simpler characterization.

    # Actually, the simplest approach: 
    # Fix the signs of a spanning tree to +1. The remaining m+1 edges are free.
    # We need to identify which edges form a spanning tree.

    # Let me build the spanning tree explicitly:
    # Tree edges: all matching edges + first m-1 bottom edges
    # Non-tree edges: last bottom edge + all top edges

    # Find the last bottom edge and all top edges
    # bottom_edges has m entries (the m edges of the m-cycle)
    # top_edges has m entries

    # The spanning tree consists of:
    # - All m matching edges
    # - First m-1 bottom cycle edges
    # Total: 2m-1 edges (correct for a tree on 2m vertices)

    # The m+1 non-tree (chord) edges are:
    # - Last bottom cycle edge: bottom_edges[m-1]
    # - All m top cycle edges: top_edges[0], ..., top_edges[m-1]

    chord_indices = [bottom_edges[m-1]] + top_edges[:]
    tree_indices = set(range(E)) - set(chord_indices)

    # Now enumerate all 2^(m+1) sign assignments for the chords
    # All tree edges are fixed to +1
    base_signs = np.ones(E, dtype=np.int8)

    best_signs = None
    best_radius = np.inf

    num_chords = len(chord_indices)
    num_assignments = 2 ** num_chords

    for assignment in product([1, -1], repeat=num_chords):
        if time.time() > deadline:
            break

        signs = base_signs.copy()
        for j, chord_idx in enumerate(chord_indices):
            signs[chord_idx] = assignment[j]

        valid, radius = is_valid_lift(signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()

    # Fallback if no valid lift found (shouldn't happen for m=4,5,7,9)
    if best_signs is None:
        best_signs = base_signs.copy()

    return best_signs
