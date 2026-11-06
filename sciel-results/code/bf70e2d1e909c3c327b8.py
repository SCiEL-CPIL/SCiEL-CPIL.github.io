import time
import random
import sys
import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

N = 9
MAX_V = 1 << N

def construct():
    deadline = time.time() + 115.0  # Safety margin
    
    # Precompute neighbors
    neighbors = [[] for _ in range(MAX_V)]
    for v in range(MAX_V):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
            
    # Map edge (u, v) to index
    # We use directed edges for ILP
    edge_indices = {}
    edge_list = []
    for u in range(MAX_V):
        for v in neighbors[u]:
            if u < v: # Only add once? No, directed.
                pass
            idx = len(edge_list)
            edge_indices[(u, v)] = idx
            edge_list.append((u, v))
            
    num_edges = len(edge_list)
    
    best_path = [0]
    best_len = 1
    
    def verify_and_extract_path(x_values, y_values):
        """
        x_values: array of shape (num_edges,), values for x_uv
        y_values: array of shape (MAX_V,), values for y_v
        Returns a list of vertices in order.
        """
        # Identify vertices in path
        in_path = [i for i in range(MAX_V) if y_values[i] > 0.5]
        if not in_path:
            return []
            
        # Build adjacency from x values
        adj = {v: [] for v in in_path}
        for (u, v), idx in edge_indices.items():
            if x_values[idx] > 0.5:
                adj[u].append(v)
                
        # Find start (in-degree 0)
        start = None
        for v in in_path:
            # Check in-degree
            in_deg = 0
            for u in neighbors[v]:
                if u in adj and v in adj[u]: # u -> v
                    in_deg += 1
            if in_deg == 0:
                start = v
                break
                
        if start is None:
            # Might be a cycle or no start found (error)
            # Try to find any node with out-degree 1 and in-degree 0?
            # If all have in-degree 1, it's a cycle. Snake-in-the-box is a path.
            # If ILP returns a cycle, it's invalid for our purpose (induced path).
            # We'll try to break it or return empty.
            return []
            
        # Trace path
        path = [start]
        current = start
        visited = {start}
        while True:
            nexts = adj.get(current, [])
            if not nexts:
                break
            # There should be exactly one next
            if len(nexts) > 1:
                # Error
                break
            nxt = nexts[0]
            if nxt in visited:
                break
            path.append(nxt)
            visited.add(nxt)
            current = nxt
            
        return path

    # 1. Try ILP
    if HAS_SCIPY:
        try:
            # Variables: x_edges (num_edges), y_vertices (MAX_V), s_vertices (MAX_V), z_vertices (MAX_V)
            # Total vars: num_edges + 3 * MAX_V
            
            c = np.zeros(num_edges + 3 * MAX_V)
            # Maximize sum(y) -> Minimize -sum(y)
            c[MAX_V : MAX_V + MAX_V] = -1.0
            
            # Constraints
            # 1. Flow conservation: sum_in(x) + sum_out(x) = 2*y - s
            # sum_in(x) + sum_out(x) + s - 2*y = 0
            # For each v:
            #   sum_{u} x_{uv} + sum_{u} x_{vu} + s_v - 2 y_v = 0
            
            A_rows = []
            b_lower = []
            b_upper = []
            
            # Constraint 1: Flow
            for v in range(MAX_V):
                row = np.zeros(num_edges + 3 * MAX_V)
                # x terms
                for u in neighbors[v]:
                    if (u, v) in edge_indices:
                        row[edge_indices[(u, v)]] = 1.0
                    if (v, u) in edge_indices:
                        row[edge_indices[(v, u)]] = 1.0
                # s term
                row[num_edges + 2 * MAX_V + v] = 1.0
                # y term
                row[MAX_V + v] = -2.0
                
                A_rows.append(row)
                b_lower.append(0.0)
                b_upper.append(0.0)
                
            # Constraint 2: Induced path
            # y_u + y_v - 1 <= x_uv + x_vu  =>  -x_uv - x_vu + y_u + y_v <= 1
            for u in range(MAX_V):
                for v in neighbors[u]:
                    if u < v: # Symmetric, just check once? No, x is directed.
                        # We need this for every pair (u,v) in E.
                        # The constraint is: if u and v are both in path, they must be connected.
                        # y_u + y_v - 1 <= x_{uv} + x_{vu}
                        # -x_{uv} - x_{vu} + y_u + y_v <= 1
                        row = np.zeros(num_edges + 3 * MAX_V)
                        if (u, v) in edge_indices:
                            row[edge_indices[(u, v)]] = -1.0
                        if (v, u) in edge_indices:
                            row[edge_indices[(v, u)]] = -1.0
                        row[MAX_V + u] = 1.0
                        row[MAX_V + v] = 1.0
                        
                        A_rows.append(row)
                        b_lower.append(-np.inf)
                        b_upper.append(1.0)
                        
            # Constraint 3: s_v <= y_v
            # s_v - y_v <= 0
            for v in range(MAX_V):
                row = np.zeros(num_edges + 3 * MAX_V)
                row[num_edges + 2 * MAX_V + v] = 1.0
                row[MAX_V + v] = -1.0
                A_rows.append(row)
                b_lower.append(-np.inf)
                b_upper.append(0.0)
                
            # Constraint 4: Sum s_v = 2 (Exactly 2 endpoints)
            # This ensures a single path if no subtours.
            row = np.zeros(num_edges + 3 * MAX_V)
            for v in range(MAX_V):
                row[num_edges + 2 * MAX_V + v] = 1.0
            A_rows.append(row)
            b_lower.append(2.0)
            b_upper.append(2.0)
            
            # Constraint 5: MTZ Subtour Elimination
            # z_v - z_u + N * x_uv <= N - 1
            # z_v - z_u - (N-1) <= -N * x_uv  ??
            # Standard: z_v - z_u + N x_uv <= N - 1
            # z_v - z_u <= N - 1 - N x_uv
            # If x_uv = 1, z_v - z_u <= N - 1 - N = -1 => z_v < z_u
            # If x_uv = 0, z_v - z_u <= N - 1
            
            for (u, v), idx in edge_indices.items():
                row = np.zeros(num_edges + 3 * MAX_V)
                row[idx] = N
                row[num_edges + 3 * MAX_V + v] = 1.0
                row[num_edges + 3 * MAX_V + u] = -1.0
                
                A_rows.append(row)
                b_lower.append(-np.inf)
                b_upper.append(N - 1)
                
            # Constraint 6: z_v bounds
            # 0 <= z_v <= N-1 if y_v=1?
            # z_v <= (N-1) y_v
            # z_v >= 0
            
            for v in range(MAX_V):
                # z_v - (N-1)y_v <= 0
                row = np.zeros(num_edges + 3 * MAX_V)
                row[num_edges + 3 * MAX_V + v] = 1.0
                row[MAX_V + v] = -(N - 1)
                A_rows.append(row)
                b_lower.append(-np.inf)
                b_upper.append(0.0)
                
                # z_v >= 0
                row = np.zeros(num_edges + 3 * MAX_V)
                row[num_edges + 3 * MAX_V + v] = -1.0
                A_rows.append(row)
                b_lower.append(-np.inf)
                b_upper.append(0.0) # -z_v <= 0 => z_v >= 0

            A = np.array(A_rows)
            b_lower = np.array(b_lower)
            b_upper = np.array(b_upper)
            
            integrality = np.ones(num_edges + 3 * MAX_V)
            integrality[num_edges + 3 * MAX_V:] = 0 # z can be continuous? No, MTZ usually works with integers or continuous if bounds are tight. Let's keep integer for safety.
            # Actually, z can be continuous in MTZ if the graph is complete? No, it's usually integer.
            # Let's set z to integer.
            integrality[num_edges + 3 * MAX_V:] = 1.0
            
            bounds = Bounds(lb=0, ub=1)
            # z bounds are handled in constraints, but variable bounds 0..1 is wrong for z.
            # z should be 0..N-1.
            # milp variable bounds are per variable.
            # x, y, s are 0..1. z is 0..N-1.
            
            lb = np.zeros(num_edges + 3 * MAX_V)
            ub = np.ones(num_edges + 3 * MAX_V)
            ub[num_edges + 3 * MAX_V:] = N - 1
            
            constraints = LinearConstraint(A, b_lower, b_upper)
            
            # Run ILP
            res = milp(c, constraints=constraints, integrality=integrality, bounds=Bounds(lb, ub))
            
            if res.status == 0: # Optimal
                x_vals = res.x[:num_edges]
                y_vals = res.x[num_edges : num_edges + MAX_V]
                path = verify_and_extract_path(x_vals, y_vals)
                if len(path) > best_len:
                    best_path = path
                    best_len = len(path)
                    
        except Exception as e:
            pass # Fall through to greedy

    # 2. Fast Greedy Fallback
    if time.time() < deadline - 5:
        def fast_greedy(start):
            path = [start]
            used = {start}
            # To speed up validity check:
            # A vertex v is valid to add to end e if:
            # 1. v not in used
            # 2. For all u in neighbors[v], u == e OR u not in used
            
            while time.time() < deadline - 2:
                best_candidate = None
                best_deg = float('inf')
                best_end = None
                
                for end in [0, -1]:
                    cur = path[0] if end == 0 else path[-1]
                    
                    for nb in neighbors[cur]:
                        if nb in used:
                            continue
                        
                        # Check induced constraint
                        valid = True
                        for u in neighbors[nb]:
                            if u == cur:
                                continue
                            if u in used:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # Calculate degree (number of valid next steps)
                        deg = 0
                        for nn in neighbors[nb]:
                            if nn in used or nn == cur:
                                continue
                            # Check if nn would be valid
                            # nn is valid if all its neighbors (except nb) are not in used
                            ok = True
                            for uu in neighbors[nn]:
                                if uu == nb:
                                    continue
                                if uu in used:
                                    ok = False
                                    break
                            if ok:
                                deg += 1
                                
                        if deg < best_deg:
                            best_deg = deg
                            best_candidate = nb
                            best_end = end
                            
                if best_candidate is None:
                    break
                    
                if best_end == 0:
                    path.insert(0, best_candidate)
                else:
                    path.append(best_candidate)
                used.add(best_candidate)
                
            return path

        # Run multiple restarts
        num_restarts = 0
        while time.time() < deadline - 2:
            start = random.randint(0, MAX_V - 1)
            path = fast_greedy(start)
            if len(path) > best_len:
                best_len = len(path)
                best_path = path
            num_restarts += 1
            if num_restarts > 2000: # Hard cap to prevent infinite loop if time check fails
                break

    return best_path
