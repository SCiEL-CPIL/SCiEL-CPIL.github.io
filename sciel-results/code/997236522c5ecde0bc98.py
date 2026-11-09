# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square [0,1]^2 to maximize
    the sum of their side lengths.

    Strategy: enumerate separation topologies (binary axis choices per pair),
    for each topology build a difference-constraint LP that maximizes sum(s_i),
    and return the best feasible packing found.
    """
    n = 10
    tol = 1e-7
    deadline = time.time() + 580  # safety margin

    best_centers = None
    best_sides = None
    best_score = -np.inf

    # --- Baseline: known analytical tiling (sum = 3.0) ---
    base_centers, base_sides = _baseline_tiling()
    if _is_feasible(base_centers, base_sides):
        best_centers = base_centers.copy()
        best_sides = base_sides.copy()
        best_score = float(np.sum(base_sides))

    # --- Topology enumeration with LP ---
    if time.time() < deadline:
        centers, sides, score = _topology_lp_search(n, deadline)
        if score > best_score:
            best_centers = centers
            best_sides = sides
            best_score = score

    # --- Final fallback ---
    if best_centers is None:
        side = 0.2
        cx, cy = np.meshgrid(np.linspace(side/2, 1-side/2, 5),
                             np.linspace(side/2, 1-side/2, 2))
        best_centers = np.column_stack((cx.flatten(), cy.flatten()))
        best_sides = np.full(n, side)
        best_score = float(np.sum(best_sides))

    return best_centers, best_sides


def _baseline_tiling():
    """2 large 0.5 squares + 8 small 0.25 squares, sum = 3.0"""
    centers = np.array([
        [0.25, 0.25],
        [0.25, 0.75],
        [0.625, 0.125],
        [0.875, 0.125],
        [0.625, 0.375],
        [0.875, 0.375],
        [0.625, 0.625],
        [0.875, 0.625],
        [0.625, 0.875],
        [0.875, 0.875],
    ])
    sides = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
    return centers, sides


def _is_feasible(centers, sides):
    """Check all constraints with tolerance."""
    n = len(centers)
    tol = 1e-7
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(s):
            return False
        if s <= 0:
            return False
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i][0] - centers[j][0]) < t and abs(centers[i][1] - centers[j][1]) < t:
                return False
    return True


def _topology_lp_search(n, deadline):
    """
    Enumerate separation topologies. For each topology (a set of binary
    horizontal/vertical decisions for each pair), solve an LP:
      maximize sum(s_i)
      subject to:
        s_i >= 0, s_i <= 1
        For each pair (i,j) with axis 'x': cx_i + s_i/2 <= cx_j - s_j/2  OR  cx_j + s_j/2 <= cx_i - s_i/2
        For each pair (i,j) with axis 'y': cy_i + s_i/2 <= cy_j - s_j/2  OR  cy_j + s_j/2 <= cy_i - s_i/2
        Boundary: cx_i - s_i/2 >= 0, cx_i + s_i/2 <= 1, etc.

    Variables: cx_i, cy_i, s_i for i=0..n-1 => 3n variables.

    We use a greedy randomized search over topologies: start from the baseline
    topology, randomly flip a few axis decisions, check DAG acyclicity, solve LP.
    """
    best_centers = None
    best_sides = None
    best_score = -np.inf

    # Generate initial topologies
    topologies = _generate_topologies(n, deadline)

    for topo in topologies:
        if time.time() > deadline - 10:
            break
        centers, sides, score = _solve_topology_lp(n, topo)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            best_sides = sides.copy()

    return best_centers, best_sides, best_score


def _generate_topologies(n, deadline):
    """
    Generate diverse separation topologies.
    Topology is a dict: (i,j) -> 'x' or 'y' for i<j.
    We use:
    1. Baseline topology (from the 2+8 tiling).
    2. Randomized perturbations of the baseline.
    3. Symmetry-based variants.
    """
    topologies = []

    # Baseline topology from the 2+8 tiling
    base_topo = _baseline_topology(n)
    topologies.append(base_topo)

    # Add some structured variants
    # Variant: all horizontal (stack vertically)
    all_h = {}
    for i in range(n):
        for j in range(i+1, n):
            all_h[(i,j)] = 'y'  # vertical separation (stacked)
    topologies.append(all_h)

    # Variant: all vertical (side by side)
    all_v = {}
    for i in range(n):
        for j in range(i+1, n):
            all_v[(i,j)] = 'x'
    topologies.append(all_v)

    # Randomized topologies: start from baseline, flip random pairs
    np.random.seed(42)
    num_random = 50
    for _ in range(num_random):
        if time.time() > deadline - 30:
            break
        topo = dict(base_topo)
        # Flip 2-5 random edges
        pairs = list(topo.keys())
        num_flips = np.random.randint(2, 6)
        for _ in range(num_flips):
            p = np.random.choice(len(pairs))
            pair = pairs[p]
            topo[pair] = 'x' if topo[pair] == 'y' else 'y'
        # Check acyclicity before adding
        if _is_acyclic(n, topo):
            topologies.append(topo)

    # More targeted: try to improve by making large squares separate from each other
    # and small squares pack tighter
    for trial in range(30):
        if time.time() > deadline - 30:
            break
        topo = dict(base_topo)
        # Force squares 0 and 1 to be vertically separated (they are large)
        topo[(0,1)] = 'y'
        # Randomly assign others
        for i in range(2, n):
            for j in range(i+1, n):
                if np.random.random() < 0.5:
                    topo[(i,j)] = 'x'
                else:
                    topo[(i,j)] = 'y'
        if _is_acyclic(n, topo):
            topologies.append(topo)

    return topologies


def _baseline_topology(n):
    """
    Derive the separation topology from the baseline 2+8 tiling.
    Squares 0,1 are in left column (vertically separated).
    Squares 2-9 are in right area (2 cols x 4 rows).
    Left vs right: horizontal separation.
    Within left: vertical.
    Within right: same row -> horizontal, same col -> vertical.
    """
    topo = {}
    # Square positions (col, row) in a conceptual grid
    # 0: (0,0), 1: (0,1)  -- left column
    # 2: (1,0), 3: (2,0)  -- right, row 0
    # 4: (1,1), 5: (2,1)  -- right, row 1
    # 6: (1,2), 7: (2,2)  -- right, row 2
    # 8: (1,3), 9: (2,3)  -- right, row 3

    # For pairs within same column (same x-position concept): vertical sep
    # For pairs within same row (same y-position concept): horizontal sep
    # For cross-column/row: decide based on geometry

    # Simple rule: if they share a "column" index, use 'y'; if share "row", use 'x'
    # Col assignments: 0->0, 1->0, 2->1, 3->2, 4->1, 5->2, 6->1, 7->2, 8->1, 9->2
    # Row assignments: 0->0, 1->1, 2->0, 3->0, 4->1, 5->1, 6->2, 7->2, 8->3, 9->3
    col = [0, 0, 1, 2, 1, 2, 1, 2, 1, 2]
    row = [0, 1, 0, 0, 1, 1, 2, 2, 3, 3]

    for i in range(n):
        for j in range(i+1, n):
            if col[i] == col[j]:
                topo[(i,j)] = 'y'  # same column -> vertical separation
            elif row[i] == row[j]:
                topo[(i,j)] = 'x'  # same row -> horizontal separation
            else:
                # Different col and row: use the one that gives more room
                # If col differs more, use horizontal; else vertical
                if abs(col[i] - col[j]) >= abs(row[i] - row[j]):
                    topo[(i,j)] = 'x'
                else:
                    topo[(i,j)] = 'y'
    return topo


def _is_acyclic(n, topo):
    """
    Check if the separation constraints form a DAG.
    For each pair (i,j) with axis 'x': either cx_i + s_i/2 <= cx_j - s_j/2 (i left of j)
    or cx_j + s_j/2 <= cx_i - s_i/2 (j left of i).
    The LP will determine which direction; for acyclicity we need to ensure
    no contradictory ordering is forced. We check by trying to assign a topological
    order consistent with the constraints.

    Actually, for difference constraints, we just need the constraint graph to be
    acyclic. Each constraint is of the form:
      x_j - x_i >= c  (for some variables x being cx or cy)

    We build a directed graph: for each pair, add an edge in one direction.
    Since the LP decides the direction, we need to check that BOTH possible
    assignments don't create a cycle. Actually, the standard approach:
    For each pair, we have a disjunction. We fix one direction (say i before j)
    and check if the resulting graph is acyclic. If not, try the other direction.

    For simplicity in topology generation, we'll use a heuristic: assign directions
    based on a total order and check consistency.
    """
    # Build directed graph based on a canonical ordering
    # For each pair (i,j), if topo says 'x', add edge i->j or j->i in x-dimension
    # For acyclicity, we need: in x-dim, the ordering must be consistent (no cycles)
    # In y-dim, the ordering must be consistent.

    # Simple check: assign each square a rank. For 'x' constraints, the rank order
    # must be consistent; for 'y' constraints, a different rank order must be consistent.
    # This is complex; instead, we just try to solve the LP and check feasibility.
    # For the topology filter, we'll use a simpler heuristic:
    # Check that the 'x' constraints don't force a cycle (i before j before k before i)

    # Build x-ordering constraints
    x_edges = {}  # (i,j) -> True means i must be left of j in x
    y_edges = {}

    for (i,j), axis in topo.items():
        if axis == 'x':
            # We'll tentatively say i is left of j (i < j in x)
            x_edges[(i,j)] = True
        else:
            y_edges[(i,j)] = True

    # Check for cycles in x_edges using DFS
    if _has_cycle(n, x_edges):
        # Try reversing all x edges
        x_edges_rev = {(j,i): True for (i,j) in x_edges}
        if _has_cycle(n, x_edges_rev):
            return False
    if _has_cycle(n, y_edges):
        y_edges_rev = {(j,i): True for (i,j) in y_edges}
        if _has_cycle(n, y_edges_rev):
            return False

    return True


def _has_cycle(n, edges):
    """Check if directed graph has a cycle using Kahn's algorithm."""
    in_degree = [0] * n
    adj = [[] for _ in range(n)]
    for (i,j) in edges:
        adj[i].append(j)
        in_degree[j] += 1

    queue = [i for i in range(n) if in_degree[i] == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return visited < n


def _solve_topology_lp(n, topo):
    """
    Solve the LP for a given separation topology.

    Variables: [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ..., cx_{n-1}, cy_{n-1}, s_{n-1}]
    Total: 3n variables.

    Objective: maximize sum(s_i) => minimize -sum(s_i)

    Constraints:
    1. s_i >= 0 (for all i)
    2. s_i <= 1 (for all i)
    3. Boundary:
       cx_i - s_i/2 >= 0  =>  cx_i - 0.5*s_i >= 0
       cx_i + s_i/2 <= 1  =>  cx_i + 0.5*s_i <= 1
       cy_i - s_i/2 >= 0
       cy_i + s_i/2 <= 1
    4. Separation constraints (for each pair with fixed axis and direction):
       If axis='x' and i is left of j: cx_i + s_i/2 <= cx_j - s_j/2
         => cx_i + 0.5*s_i - cx_j + 0.5*s_j <= 0
       If axis='y' and i is below j: cy_i + s_i/2 <= cy_j - s_j/2
         => cy_i + 0.5*s_i - cy_j + 0.5*s_j <= 0

    For the direction, we need to determine which square is "before" the other.
    We'll use the topology's implied ordering. For simplicity, we'll try both
    directions for each pair and use the one that gives a feasible LP.
    Actually, to keep it tractable, we'll fix the direction based on index:
    For pair (i,j) with i<j, if axis='x', we say i is left of j.
    If axis='y', we say i is below j.
    This may be suboptimal but gives a valid packing if feasible.
    """
    num_vars = 3 * n

    # Objective: minimize -sum(s_i)
    c = np.zeros(num_vars)
    for i in range(n):
        c[3*i + 2] = -1.0  # s_i is at index 3*i + 2

    # Inequality constraints: A_ub @ x <= b_ub
    A_rows = []
    b_rows = []

    # s_i >= 0 => -s_i <= 0
    for i in range(n):
        row = np.zeros(num_vars)
        row[3*i + 2] = -1.0
        A_rows.append(row)
        b_rows.append(0.0)

    # s_i <= 1
    for i in range(n):
        row = np.zeros(num_vars)
        row[3*i + 2] = 1.0
        A_rows.append(row)
        b_rows.append(1.0)

    # Boundary: cx_i - 0.5*s_i >= 0 => -cx_i + 0.5*s_i <= 0
    for i in range(n):
        row = np.zeros(num_vars)
        row[3*i + 0] = -1.0
        row[3*i + 2] = 0.5
        A_rows.append(row)
        b_rows.append(0.0)

    # Boundary: cx_i + 0.5*s_i <= 1
    for i in range(n):
        row = np.zeros(num_vars)
        row[3*i + 0] = 1.0
        row[3*i + 2] = 0.5
        A_rows.append(row)
        b_rows.append(1.0)

    # Boundary: cy_i - 0.5*s_i >= 0 => -cy_i + 0.5*s_i <= 0
    for i in range(n):
        row = np.zeros(num_vars)
        row[3*i + 1] = -1.0
        row[3*i + 2] = 0.5
        A_rows.append(row)
        b_rows.append(0.0)

    # Boundary: cy_i + 0.5*s_i <= 1
    for i in range(n):
        row = np.zeros(num_vars)
        row[3*i + 1] = 1.0
        row[3*i + 2] = 0.5
        A_rows.append(row)
        b_rows.append(1.0)

    # Separation constraints
    # For pair (i,j) with i<j:
    #   axis 'x': i is left of j => cx_i + 0.5*s_i - cx_j + 0.5*s_j <= 0
    #   axis 'y': i is below j => cy_i + 0.5*s_i - cy_j + 0.5*s_j <= 0
    for (i, j), axis in topo.items():
        row = np.zeros(num_vars)
        if axis == 'x':
            row[3*i + 0] = 1.0   # cx_i
            row[3*i + 2] = 0.5   # + 0.5*s_i
            row[3*j + 0] = -1.0  # - cx_j
            row[3*j + 2] = 0.5   # + 0.5*s_j
        else:  # 'y'
            row[3*i + 1] = 1.0   # cy_i
            row[3*i + 2] = 0.5   # + 0.5*s_i
            row[3*j + 1] = -1.0  # - cy_j
            row[3*j + 2] = 0.5   # + 0.5*s_j
        A_rows.append(row)
        b_rows.append(0.0)

    if not A_rows:
        return None, None, -np.inf

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    # Variable bounds: cx, cy in [0, 1], s in [0, 1]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # cx_i
        bounds.append((0.0, 1.0))  # cy_i
        bounds.append((0.0, 1.0))  # s_i

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        return None, None, -np.inf

    if result.status != 0:
        return None, None, -np.inf

    x = result.x
    centers = np.array([[x[3*i + 0], x[3*i + 1]] for i in range(n)])
    sides = np.array([x[3*i + 2] for i in range(n)])

    # Verify feasibility with tolerance
    if not _is_feasible(centers, sides):
        # Try a small perturbation to fix tolerance issues
        sides = np.maximum(sides, 1e-6)
        # Re-center to ensure boundary compliance
        for i in range(n):
            s = sides[i]
            centers[i, 0] = np.clip(centers[i, 0], s/2, 1 - s/2)
            centers[i, 1] = np.clip(centers[i, 1], s/2, 1 - s/2)
        if not _is_feasible(centers, sides):
            return None, None, -np.inf

    score = float(np.sum(sides))
    return centers, sides, score


# EVOLVE-BLOCK-END
