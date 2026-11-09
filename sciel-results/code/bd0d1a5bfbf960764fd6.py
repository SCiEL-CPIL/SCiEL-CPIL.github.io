# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from itertools import product
from collections import deque

def _is_valid(centers, sides, tol=1e-7):
    """Check if configuration is valid."""
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            if dx < t and dy < t:
                return False
    return True

def _solve_lp(sep_x, sep_y, n=10):
    """
    Solve LP for a given separation graph.
    sep_x: list of (i,j) meaning square i is LEFT of square j (x_i + s_i/2 <= x_j - s_j/2)
    sep_y: list of (i,j) meaning square i is BELOW square j (y_i + s_i/2 <= y_j - s_j/2)
    
    Variables: [x_0, y_0, s_0, x_1, y_1, s_1, ..., x_9, y_9, s_9]
    Maximize sum(s_i)
    Subject to:
      x_i + s_i/2 <= 1
      y_i + s_i/2 <= 1
      x_i - s_i/2 >= 0
      y_i - s_i/2 >= 0
      s_i > 0 (we'll use s_i >= eps)
      For each (i,j) in sep_x: x_i + s_i/2 <= x_j - s_j/2
      For each (i,j) in sep_y: y_i + s_i/2 <= y_j - s_j/2
    """
    # Variable indices: var_k(i) = 3*i + k where k=0(x), 1(y), 2(s)
    def vx(i): return 3*i
    def vy(i): return 3*i + 1
    def vs(i): return 3*i + 2
    
    nvars = 3 * n
    eps = 1e-9
    
    # Objective: maximize sum(s_i) => minimize -sum(s_i)
    c = np.zeros(nvars)
    for i in range(n):
        c[vs(i)] = -1.0
    
    # Inequality constraints: A_ub @ x <= b_ub
    A_ub = []
    b_ub = []
    
    # Boundary constraints
    for i in range(n):
        # x_i + s_i/2 <= 1
        row = np.zeros(nvars)
        row[vx(i)] = 1.0
        row[vs(i)] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)
        
        # y_i + s_i/2 <= 1
        row = np.zeros(nvars)
        row[vy(i)] = 1.0
        row[vs(i)] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)
        
        # x_i - s_i/2 >= 0 => -x_i + s_i/2 <= 0
        row = np.zeros(nvars)
        row[vx(i)] = -1.0
        row[vs(i)] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        # y_i - s_i/2 >= 0 => -y_i + s_i/2 <= 0
        row = np.zeros(nvars)
        row[vy(i)] = -1.0
        row[vs(i)] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        # s_i >= eps => -s_i <= -eps
        row = np.zeros(nvars)
        row[vs(i)] = -1.0
        A_ub.append(row)
        b_ub.append(-eps)
    
    # Separation constraints
    for (i, j) in sep_x:
        # x_i + s_i/2 <= x_j - s_j/2
        # x_i - x_j + s_i/2 + s_j/2 <= 0
        row = np.zeros(nvars)
        row[vx(i)] = 1.0
        row[vx(j)] = -1.0
        row[vs(i)] = 0.5
        row[vs(j)] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
    
    for (i, j) in sep_y:
        # y_i + s_i/2 <= y_j - s_j/2
        # y_i - y_j + s_i/2 + s_j/2 <= 0
        row = np.zeros(nvars)
        row[vy(i)] = 1.0
        row[vy(j)] = -1.0
        row[vs(i)] = 0.5
        row[vs(j)] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
    
    if not A_ub:
        return None
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(None, None)] * nvars
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            x = res.x
            centers = np.array([[x[vx(i)], x[vy(i)]] for i in range(n)])
            sides = np.array([x[vs(i)] for i in range(n)])
            return centers, sides
        else:
            return None
    except Exception:
        return None

def _topological_sort_possible(adj, n):
    """Check if graph is a DAG using Kahn's algorithm. Returns True if acyclic."""
    in_degree = [0] * n
    for i in range(n):
        for j in adj[i]:
            in_degree[j] += 1
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    count = 0
    while queue:
        u = queue.popleft()
        count += 1
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    return count == n

def _generate_separation_graphs(n=10, max_edges_per_pair=1):
    """
    Generate candidate separation graphs.
    For each pair (i,j), we can have:
      - i left of j (x-separation)
      - j left of i (x-separation)  
      - i below j (y-separation)
      - j below i (y-separation)
      - or no constraint (they might be separated by other squares)
    
    This is 5^45 which is way too many. Instead, we use a smarter approach:
    Try specific structured layouts.
    """
    pass

def _try_layout(layout_desc, n=10):
    """
    Try a specific layout description.
    layout_desc: dict mapping square index to (row, col) in a grid-like structure
    """
    pass

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 570  # Safety margin
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if _is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()
    
    # Strategy: Try many different separation graphs via combinatorial search
    # For 10 squares, we try structured layouts
    
    # Layout type 1: 2x5 grid (2 rows, 5 cols)
    # All squares same size, arranged in 2x5
    # Separation: in each row, left-to-right; between rows, bottom-to-top
    sep_x = []
    sep_y = []
    for row in range(2):
        for col in range(4):
            i = row * 5 + col
            j = row * 5 + col + 1
            sep_x.append((i, j))
    for col in range(5):
        i = col
        j = 5 + col
        sep_y.append((i, j))
    
    result = _solve_lp(sep_x, sep_y, n)
    if result:
        update_best(result[0], result[1])
    
    if time.time() > deadline:
        if best_centers is not None:
            return best_centers, best_sides
        # Fallback
        c = np.array([[0.5, 0.5]] * n)
        s = np.array([0.1] * n)
        return c, s
    
    # Layout type 2: 3x3 grid + 1 center (but this doesn't work well)
    # Instead: 3 rows: row0 has 3, row1 has 4, row2 has 3
    # This is getting complex. Let's try a different approach.
    
    # Better approach: Use the fact that in optimal packing, squares form a "contact graph"
    # Try many random separation graphs and solve LP for each
    
    np.random.seed(42)
    
    # Generate random separation graphs
    # For each pair, randomly choose one of 4 separation directions or none
    # But we need to avoid cycles
    
    tried_graphs = set()
    attempts = 0
    max_attempts = 5000
    
    while attempts < max_attempts and time.time() < deadline - 30:
        attempts += 1
        
        # Generate random separation graph
        sep_x = []
        sep_y = []
        
        for i in range(n):
            for j in range(i+1, n):
                r = np.random.random()
                if r < 0.25:
                    sep_x.append((i, j))
                elif r < 0.50:
                    sep_x.append((j, i))
                elif r < 0.75:
                    sep_y.append((i, j))
                elif r < 1.00:
                    sep_y.append((j, i))
                # else: no constraint
        
        # Check for cycles
        # Build adjacency list
        adj_x = [[] for _ in range(n)]
        for (i, j) in sep_x:
            adj_x[i].append(j)
        
        adj_y = [[] for _ in range(n)]
        for (i, j) in sep_y:
            adj_y[i].append(j)
        
        if not _topological_sort_possible(adj_x, n) or not _topological_sort_possible(adj_y, n):
            continue
        
        # Solve LP
        result = _solve_lp(sep_x, sep_y, n)
        if result:
            update_best(result[0], result[1])
    
    if time.time() > deadline:
        if best_centers is not None:
            return best_centers, best_sides
    
    # Try structured layouts more systematically
    # Layout: k squares in top row, (10-k) in bottom row
    for k in range(1, 10):
        sep_x = []
        sep_y = []
        # Top row: squares 0 to k-1, left to right
        for i in range(k-1):
            sep_x.append((i, i+1))
        # Bottom row: squares k to 9, left to right
        for i in range(k, n-1):
            sep_x.append((i, i+1))
        # Bottom row is below top row
        for i in range(k):
            for j in range(k, n):
                sep_y.append((i, j))
        
        result = _solve_lp(sep_x, sep_y, n)
        if result:
            update_best(result[0], result[1])
    
    # Layout: 3 rows with varying counts
    row_counts = [
        [2, 3, 5], [3, 3, 4], [3, 4, 3], [4, 3, 3], [5, 3, 2],
        [2, 4, 4], [4, 4, 2], [3, 5, 2], [2, 5, 3], [4, 2, 4],
        [1, 4, 5], [5, 4, 1], [2, 2, 6], [6, 2, 2], [3, 2, 5],
    ]
    
    for counts in row_counts:
        if sum(counts) != n:
            continue
        
        sep_x = []
        sep_y = []
        
        # Assign squares to rows
        row_sq = []
        idx = 0
        for cnt in counts:
            row_sq.append(list(range(idx, idx+cnt)))
            idx += cnt
        
        # Within each row, left to right
        for row in row_sq:
            for i in range(len(row)-1):
                sep_x.append((row[i], row[i+1]))
        
        # Between rows, all squares in lower row are below all in upper row
        for r in range(len(row_sq)-1):
            for i in row_sq[r]:
                for j in row_sq[r+1]:
                    sep_y.append((i, j))
        
        result = _solve_lp(sep_x, sep_y, n)
        if result:
            update_best(result[0], result[1])
    
    # Layout: 4 rows
    row_counts_4 = [
        [2, 2, 3, 3], [2, 3, 2, 3], [3, 2, 2, 3], [3, 2, 3, 2],
        [2, 3, 3, 2], [3, 3, 2, 2], [1, 3, 3, 3], [3, 3, 3, 1],
        [2, 2, 2, 4], [4, 2, 2, 2], [1, 2, 3, 4], [4, 3, 2, 1],
        [2, 4, 2, 2], [2, 2, 4, 2], [3, 1, 3, 3], [3, 3, 1, 3],
    ]
    
    for counts in row_counts_4:
        if sum(counts) != n:
            continue
        
        sep_x = []
        sep_y = []
        
        row_sq = []
        idx = 0
        for cnt in counts:
            row_sq.append(list(range(idx, idx+cnt)))
            idx += cnt
        
        for row in row_sq:
            for i in range(len(row)-1):
                sep_x.append((row[i], row[i+1]))
        
        for r in range(len(row_sq)-1):
            for i in row_sq[r]:
                for j in row_sq[r+1]:
                    sep_y.append((i, j))
        
        result = _solve_lp(sep_x, sep_y, n)
        if result:
            update_best(result[0], result[1])
    
    # Layout: 5 rows
    row_counts_5 = [
        [2, 2, 2, 2, 2], [1, 2, 2, 2, 3], [3, 2, 2, 2, 1],
        [1, 1, 2, 3, 3], [3, 3, 2, 1, 1], [2, 1, 2, 3, 2],
        [2, 3, 1, 2, 2], [1, 3, 2, 2, 2], [2, 2, 3, 1, 2],
        [2, 2, 1, 3, 2], [3, 1, 2, 2, 2], [2, 2, 2, 1, 3],
    ]
    
    for counts in row_counts_5:
        if sum(counts) != n:
            continue
        
        sep_x = []
        sep_y = []
        
        row_sq = []
        idx = 0
        for cnt in counts:
            row_sq.append(list(range(idx, idx+cnt)))
            idx += cnt
        
        for row in row_sq:
            for i in range(len(row)-1):
                sep_x.append((row[i], row[i+1]))
        
        for r in range(len(row_sq)-1):
            for i in row_sq[r]:
                for j in row_sq[r+1]:
                    sep_y.append((i, j))
        
        result = _solve_lp(sep_x, sep_y, n)
        if result:
            update_best(result[0], result[1])
    
    # More random searches with bias toward structured patterns
    np.random.seed(123)
    
    for _ in range(3000):
        if time.time() > deadline - 20:
            break
        
        # Random separation graph with some structure
        sep_x = []
        sep_y = []
        
        # First, create a random permutation to define a "natural" order
        perm = np.random.permutation(n)
        
        for i in range(n):
            for j in range(i+1, n):
                r = np.random.random()
                if r < 0.30:
                    sep_x.append((perm[i], perm[j]))
                elif r < 0.60:
                    sep_x.append((perm[j], perm[i]))
                elif r < 0.90:
                    sep_y.append((perm[i], perm[j]))
                elif r < 1.00:
                    sep_y.append((perm[j], perm[i]))
        
        adj_x = [[] for _ in range(n)]
        for (i, j) in sep_x:
            adj_x[i].append(j)
        
        adj_y = [[] for _ in range(n)]
        for (i, j) in sep_y:
            adj_y[i].append(j)
        
        if not _topological_sort_possible(adj_x, n) or not _topological_sort_possible(adj_y, n):
            continue
        
        result = _solve_lp(sep_x, sep_y, n)
        if result:
            update_best(result[0], result[1])
    
    # Final: try to polish the best solution with a local search
    if best_centers is not None and time.time() < deadline - 5:
        # Try small perturbations
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            c_pert = best_centers + np.random.normal(0, 0.01, (n, 2))
            s_pert = np.maximum(best_sides + np.random.normal(0, 0.01, n), 0.01)
            # This won't directly work since we need to re-solve LP
            # Instead, just check if perturbation is valid
            if _is_valid(c_pert, s_pert):
                update_best(c_pert, s_pert)
    
    if best_centers is None:
        # Ultimate fallback
        c = np.array([[0.5, 0.5]] * n)
        s = np.array([0.1] * n)
        return c, s
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
