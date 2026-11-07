import time
import numpy as np
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _check(coloring):
    """Validate: adjacent differ and every two-color induced subgraph is a forest."""
    adj = _build_adj()
    n = len(coloring)
    for u in range(n):
        for v in adj[u]:
            if u < v and coloring[u] == coloring[v]:
                return False
    pair_edges = defaultdict(list)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                c1, c2 = coloring[u], coloring[v]
                pair_edges[(min(c1, c2), max(c1, c2))].append((u, v))

    for (a, b), edges in pair_edges.items():
        verts = set()
        for u, v in edges:
            verts.add(u)
            verts.add(v)
        vlist = list(verts)
        idx = {v: i for i, v in enumerate(vlist)}
        par = list(range(len(vlist)))
        rn = [0] * len(vlist)

        def f(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        def un(x, y):
            rx, ry = f(x), f(y)
            if rx == ry:
                return False
            if rn[rx] < rn[ry]:
                rx, ry = ry, rx
            par[ry] = rx
            if rn[rx] == rn[ry]:
                rn[rx] += 1
            return True

        for u, v in edges:
            if not un(idx[u], idx[v]):
                return False
    return True


def _try_milp(adj, nc, deadline):
    """Try to solve via MILP: minimize number of colors used."""
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except ImportError:
        return None
    
    n = P['n']
    edges = P['edges']
    m = len(edges)
    
    # Variables: x[v][c] = 1 if vertex v gets color c
    # Total variables: n * nc
    # Objective: minimize sum of color usage indicators
    # For each color c, add binary y[c] = 1 if any vertex uses color c
    # Minimize sum(y[c])
    
    # x is n*nc binary vars, y is nc binary vars
    # Total: n*nc + nc = nc*(n+1)
    
    total_vars = nc * (n + 1)
    
    # c = c * nc + v for x[v][c]
    # c = nc*(n+1) + c for y[c]  -- wait, let me re-index
    
    # Let's index: var 0..n*nc-1 are x[v*nc + c]
    # var n*nc..n*nc+nc-1 are y[c]
    
    y_offset = n * nc
    
    # Objective: minimize sum y[c]
    c_obj = np.zeros(total_vars)
    for c in range(nc):
        c_obj[y_offset + c] = 1.0
    
    # Constraints:
    # 1. Each vertex gets exactly one color: sum_c x[v][c] = 1 for all v
    # 2. Adjacent vertices differ: x[u][c] + x[v][c] <= 1 for all edges (u,v), all c
    # 3. y[c] >= x[v][c] for all v, c (y[c] = 1 if color c is used)
    # 4. Bicolored forest constraint: this is complex; we'll handle via post-check
    
    # Build constraint matrix
    # Constraint types:
    # Type A: equality (sum x[v][c] = 1): n constraints
    # Type B: inequality (x[u][c] + x[v][c] <= 1): m * nc constraints  
    # Type C: inequality (y[c] - x[v][c] >= 0): n * nc constraints
    
    n_A = n
    n_B = m * nc
    n_C = n * nc
    total_cons = n_A + n_B + n_C
    
    A_rows = []
    lb = []
    ub = []
    
    # Type A: sum_c x[v][c] = 1
    for v in range(n):
        row = np.zeros(total_vars)
        for c in range(nc):
            row[v * nc + c] = 1.0
        A_rows.append(row)
        lb.append(1.0)
        ub.append(1.0)
    
    # Type B: x[u][c] + x[v][c] <= 1
    for (u, v) in edges:
        for c in range(nc):
            row = np.zeros(total_vars)
            row[u * nc + c] = 1.0
            row[v * nc + c] = 1.0
            A_rows.append(row)
            lb.append(-np.inf)
            ub.append(1.0)
    
    # Type C: y[c] - x[v][c] >= 0  =>  -x[v][c] + y[c] >= 0
    for v in range(n):
        for c in range(nc):
            row = np.zeros(total_vars)
            row[v * nc + c] = -1.0
            row[y_offset + c] = 1.0
            A_rows.append(row)
            lb.append(0.0)
            ub.append(np.inf)
    
    A = np.array(A_rows)
    
    # Variable bounds: all binary
    bounds = Bounds(lb=np.zeros(total_vars), ub=np.ones(total_vars))
    
    # Integrality: all binary
    integrality = np.ones(total_vars)
    
    # Check time
    if time.time() > deadline:
        return None
    
    try:
        res = milp(c=c_obj, constraints=LinearConstraint(A, lb, ub),
                   integrality=integrality, bounds=bounds,
                   options={'time_limit': max(1, deadline - time.time())})
    except Exception:
        return None
    
    if res.status != 0 and res.status != 1:  # 0=optimal, 1=optimal (within tolerance)
        return None
    
    if res.x is None:
        return None
    
    x = res.x[:n * nc].reshape(n, nc)
    coloring = []
    for v in range(n):
        best_c = 0
        best_val = -1
        for c in range(nc):
            if x[v, c] > best_val:
                best_val = x[v, c]
                best_c = c
        coloring.append(int(round(best_c)))
    
    # Verify
    if _check(coloring):
        return coloring
    
    return None


def _backtrack(adj, nc, deadline, init_coloring=None):
    n = P['n']
    if init_coloring is None:
        coloring = [-1] * n
    else:
        coloring = init_coloring[:]
    
    # Precompute neighbor colors for quick check
    def get_available_colors(v):
        used = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                used.add(coloring[nb])
        return [c for c in range(nc) if c not in used]
    
    # Bicolored cycle check: after coloring v with c, check for each neighbor nb
    # with color cn != c, whether there's a path from v to nb in {c,cn} subgraph
    # not using edge (v,nb)
    def has_bicolored_cycle(v, c, adj, coloring, nc):
        for nb in adj[v]:
            cn = coloring[nb]
            if cn == -1 or cn == c:
                continue
            # BFS from v in subgraph induced by colors {c, cn}, avoiding direct edge to nb
            from collections import deque
            visited = {v}
            queue = deque([v])
            found = False
            while queue:
                cur = queue.popleft()
                for w in adj[cur]:
                    if cur == v and w == nb:
                        continue
                    if w == nb:
                        found = True
                        break
                    cw = coloring[w]
                    if cw in (c, cn) and w not in visited:
                        visited.add(w)
                        queue.append(w)
                if found:
                    break
            if found:
                return True
        return False
    
    def dfs():
        if time.time() > deadline:
            return None
        
        # Find uncolored vertex with fewest available colors (MRV heuristic)
        min_count = nc + 1
        best_v = -1
        best_colors = []
        
        for v in range(n):
            if coloring[v] == -1:
                avail = get_available_colors(v)
                if len(avail) == 0:
                    return None
                if len(avail) < min_count:
                    min_count = len(avail)
                    best_v = v
                    best_colors = avail
                    if min_count == 1:
                        break
        
        if best_v == -1:
            return coloring[:]
            
        v = best_v
        for c in best_colors:
            coloring[v] = c
            if not has_bicolored_cycle(v, c, adj, coloring, nc):
                res = dfs()
                if res is not None:
                    return res
            coloring[v] = -1
        return None

    return dfs()


def _grid_initial_coloring(nc):
    """Create an initial coloring based on the 5x10 grid structure."""
    # The graph is a 5x10 toroidal grid
    # Vertex i is at row i//10, col i%10
    # For a toroidal grid, we can use (row + col) % nc as a starting point
    n = P['n']
    coloring = [0] * n
    for v in range(n):
        row = v // 10
        col = v % 10
        coloring[v] = (row + col) % nc
    return coloring


def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin before 90s kill
    
    adj = _build_adj()
    
    # Strategy 1: Try MILP for small nc
    for nc in [3, 4]:
        if time.time() > deadline - 30:
            break
        result = _try_milp(adj, nc, deadline - 5)
        if result is not None:
            return [int(x) for x in result]
    
    # Strategy 2: Backtracking with grid-based initial coloring
    for nc in [3, 4, 5]:
        if time.time() > deadline - 10:
            break
        
        # Try with grid-based initialization
        init = _grid_initial_coloring(nc)
        result = _backtrack(adj, nc, deadline - 5, init)
        if result is not None:
            if _check(result):
                return [int(x) for x in result]
        
        # Try without initialization
        result = _backtrack(adj, nc, deadline - 5)
        if result is not None:
            if _check(result):
                return [int(x) for x in result]
    
    # Strategy 3: Greedy with local search
    for nc in [4, 5]:
        if time.time() > deadline - 5:
            break
        
        # Greedy coloring
        coloring = [-1] * P['n']
        ordering = sorted(range(P['n']), key=lambda x: -len(adj[x]))
        for v in ordering:
            used = set()
            for nb in adj[v]:
                if coloring[nb] != -1:
                    used.add(coloring[nb])
            for c in range(nc):
                if c not in used:
                    coloring[v] = c
                    break
        
        if -1 not in coloring and _check(coloring):
            return [int(x) for x in coloring]
    
    # Fallback: all distinct colors
    return [int(i) for i in range(P['n'])]
