# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    deadline = time.time() + 620  # leave 10s margin before 630s kill

    def is_feasible(c, s):
        """Check if configuration (centers, sides) is feasible."""
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(s)):
            return False
        if np.any(s <= 0):
            return False
        for i in range(n):
            cx, cy = c[i]
            si = s[i]
            if cx - si / 2 < -tol or cx + si / 2 > 1 + tol:
                return False
            if cy - si / 2 < -tol or cy + si / 2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2 - tol
                if abs(c[i][0] - c[j][0]) < t and abs(c[i][1] - c[j][1]) < t:
                    return False
        return True

    def objective(c, s):
        return np.sum(s)

    def solve_topology(topology):
        """
        Solve LP for a given topology.
        topology: dict mapping (i,j) -> 'h' or 'v' for i<j
        Variables: x[i], y[i], s[i] for i in 0..n-1
        Total variables: 3*n
        
        Constraints:
        1. Boundary: s[i]/2 <= x[i] <= 1 - s[i]/2
                     s[i]/2 <= y[i] <= 1 - s[i]/2
        2. Non-overlap: for each pair (i,j) with topology 'h':
                         x[j] - x[i] >= (s[i] + s[j])/2
                     or 'v':
                         y[j] - y[i] >= (s[i] + s[j])/2
        3. s[i] >= eps (small positive)
        
        Objective: maximize sum(s[i])
        """
        eps = 1e-9
        num_vars = 3 * n
        # Variable ordering: x[0..n-1], y[0..n-1], s[0..n-1]
        
        # Objective: minimize -sum(s[i])
        c_obj = np.zeros(num_vars)
        for i in range(n):
            c_obj[2*n + i] = -1.0
        
        # Inequality constraints: A_ub @ vars <= b_ub
        A_ub_rows = []
        b_ub_rows = []
        
        # Boundary constraints for x: x[i] - s[i]/2 >= 0  =>  -x[i] + s[i]/2 <= 0
        for i in range(n):
            row = np.zeros(num_vars)
            row[i] = -1.0
            row[2*n + i] = 0.5
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
        
        # x[i] + s[i]/2 <= 1  =>  x[i] + s[i]/2 <= 1
        for i in range(n):
            row = np.zeros(num_vars)
            row[i] = 1.0
            row[2*n + i] = 0.5
            A_ub_rows.append(row)
            b_ub_rows.append(1.0)
        
        # Boundary constraints for y: y[i] - s[i]/2 >= 0  =>  -y[i] + s[i]/2 <= 0
        for i in range(n):
            row = np.zeros(num_vars)
            row[n + i] = -1.0
            row[2*n + i] = 0.5
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
        
        # y[i] + s[i]/2 <= 1  =>  y[i] + s[i]/2 <= 1
        for i in range(n):
            row = np.zeros(num_vars)
            row[n + i] = 1.0
            row[2*n + i] = 0.5
            A_ub_rows.append(row)
            b_ub_rows.append(1.0)
        
        # Non-overlap constraints
        for (i, j), direction in topology.items():
            if direction == 'h':
                # x[j] - x[i] >= (s[i] + s[j])/2
                # x[j] - x[i] - (s[i] + s[j])/2 >= 0
                # -x[j] + x[i] + s[i]/2 + s[j]/2 <= 0
                row = np.zeros(num_vars)
                row[i] = 1.0       # x[i]
                row[j] = -1.0      # x[j]
                row[2*n + i] = 0.5 # s[i]/2
                row[2*n + j] = 0.5 # s[j]/2
                A_ub_rows.append(row)
                b_ub_rows.append(0.0)
            else:  # 'v'
                # y[j] - y[i] >= (s[i] + s[j])/2
                # -y[j] + y[i] + s[i]/2 + s[j]/2 <= 0
                row = np.zeros(num_vars)
                row[n + i] = 1.0       # y[i]
                row[n + j] = -1.0      # y[j]
                row[2*n + i] = 0.5     # s[i]/2
                row[2*n + j] = 0.5     # s[j]/2
                A_ub_rows.append(row)
                b_ub_rows.append(0.0)
        
        if A_ub_rows:
            A_ub = np.array(A_ub_rows)
            b_ub = np.array(b_ub_rows)
        else:
            A_ub = None
            b_ub = None
        
        # Bounds: x[i] in [0,1], y[i] in [0,1], s[i] in [eps, 1]
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))  # x[i]
        for i in range(n):
            bounds.append((0.0, 1.0))  # y[i]
        for i in range(n):
            bounds.append((eps, 1.0))  # s[i]
        
        try:
            result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if result.status == 0:  # Optimal
                x_opt = result.x
                c = np.zeros((n, 2))
                s = np.zeros(n)
                for i in range(n):
                    c[i, 0] = x_opt[i]
                    c[i, 1] = x_opt[n + i]
                    s[i] = x_opt[2*n + i]
                return c, s, result.fun * -1  # negative of objective value
            else:
                return None, None, -1
        except Exception:
            return None, None, -1

    def generate_topologies():
        """
        Generate a diverse set of valid topologies.
        A topology assigns 'h' or 'v' to each pair (i,j), i<j.
        We need to ensure the constraint graph is acyclic (or at least feasible).
        
        Strategy: 
        1. Start with simple structured topologies (grid-like, guillotine-like)
        2. Add random topologies and check feasibility via LP
        """
        topologies = []
        
        # Topology 1: All horizontal (stacked vertically)
        topo1 = {}
        for i in range(n):
            for j in range(i+1, n):
                topo1[(i,j)] = 'v'  # vertical separation
        topologies.append(topo1)
        
        # Topology 2: All vertical (side by side)
        topo2 = {}
        for i in range(n):
            for j in range(i+1, n):
                topo2[(i,j)] = 'h'
        topologies.append(topo2)
        
        # Topology 3: Grid-like (3x4 grid, with some flexibility)
        # Arrange in 3 rows, 4 columns
        # Pairs in same row: horizontal
        # Pairs in same column: vertical
        # Pairs in different rows and columns: choose based on position
        topo3 = {}
        cols = 4
        rows = 3
        for i in range(n):
            ri = i // cols
            ci = i % cols
            for j in range(i+1, n):
                rj = j // cols
                cj = j % cols
                if ri == rj:
                    topo3[(i,j)] = 'h'  # same row, horizontal
                elif ci == cj:
                    topo3[(i,j)] = 'v'  # same column, vertical
                else:
                    # Different row and column: use diagonal heuristic
                    # If i is "above-left" of j, use 'h'; else 'v'
                    if ri < rj and ci <= cj:
                        topo3[(i,j)] = 'h'
                    else:
                        topo3[(i,j)] = 'v'
        topologies.append(topo3)
        
        # Topology 4: 2x5 grid
        topo4 = {}
        cols = 5
        rows = 2
        for i in range(n):
            ri = i // cols
            ci = i % cols
            for j in range(i+1, n):
                rj = j // cols
                cj = j % cols
                if ri == rj:
                    topo4[(i,j)] = 'h'
                elif ci == cj:
                    topo4[(i,j)] = 'v'
                else:
                    if ri < rj and ci <= cj:
                        topo4[(i,j)] = 'h'
                    else:
                        topo4[(i,j)] = 'v'
        topologies.append(topo4)
        
        # Topology 5: 5x2 grid
        topo5 = {}
        cols = 2
        rows = 5
        for i in range(n):
            ri = i // cols
            ci = i % cols
            for j in range(i+1, n):
                rj = j // cols
                cj = j % cols
                if ri == rj:
                    topo5[(i,j)] = 'h'
                elif ci == cj:
                    topo5[(i,j)] = 'v'
                else:
                    if ri < rj and ci <= cj:
                        topo5[(i,j)] = 'h'
                    else:
                        topo5[(i,j)] = 'v'
        topologies.append(topo5)
        
        # Topology 6: 2 big + 8 small (guillotine-like)
        # Squares 0,1 are big (left half), 2-9 are small (right half)
        # Within big: vertical
        # Within small: grid
        # Big vs small: horizontal
        topo6 = {}
        big = [0, 1]
        small = list(range(2, 10))
        for i in big:
            for j in big:
                if i < j:
                    topo6[(i,j)] = 'v'
        for i in small:
            for j in small:
                if i < j:
                    # 4x2 grid for small
                    si = i - 2
                    sj = j - 2
                    ri = si // 2
                    ci = si % 2
                    rj = sj // 2
                    cj = sj % 2
                    if ri == rj:
                        topo6[(i,j)] = 'h'
                    elif ci == cj:
                        topo6[(i,j)] = 'v'
                    else:
                        if ri < rj and ci <= cj:
                            topo6[(i,j)] = 'h'
                        else:
                            topo6[(i,j)] = 'v'
        for i in big:
            for j in small:
                if i < j:
                    topo6[(i,j)] = 'h'
                else:
                    topo6[(j,i)] = 'h'
        # Fix: ensure keys are (min, max)
        topo6_fixed = {}
        for (i,j), d in topo6.items():
            a, b = min(i,j), max(i,j)
            topo6_fixed[(a,b)] = d
        topologies.append(topo6_fixed)
        
        # Topology 7: 3 big + 7 small
        # Squares 0,1,2 are big (top row), 3-9 are small (bottom)
        topo7 = {}
        big = [0, 1, 2]
        small = list(range(3, 10))
        for i in big:
            for j in big:
                if i < j:
                    topo7[(i,j)] = 'h'  # side by side
        for i in small:
            for j in small:
                if i < j:
                    # 7 squares in bottom: 4+3 or 3+4
                    si = i - 3
                    sj = j - 3
                    # Use 4 columns for first 4, 3 for rest
                    if si < 4 and sj < 4:
                        ri = 0
                        rj = 0
                        topo7[(i,j)] = 'h'
                    elif si < 4:
                        # i in top sub-row, j in bottom
                        topo7[(i,j)] = 'v'
                    elif sj < 4:
                        topo7[(i,j)] = 'v'
                    else:
                        topo7[(i,j)] = 'h'
        for i in big:
            for j in small:
                if i < j:
                    topo7[(i,j)] = 'v'
                else:
                    topo7[(j,i)] = 'v'
        topo7_fixed = {}
        for (i,j), d in topo7.items():
            a, b = min(i,j), max(i,j)
            topo7_fixed[(a,b)] = d
        topologies.append(topo7_fixed)
        
        # Topology 8: 1 big (top-left) + 9 small (rest)
        topo8 = {}
        big = [0]
        small = list(range(1, 10))
        for i in small:
            for j in small:
                if i < j:
                    # 3x3 grid for small
                    si = i - 1
                    sj = j - 1
                    ri = si // 3
                    ci = si % 3
                    rj = sj // 3
                    cj = sj % 3
                    if ri == rj:
                        topo8[(i,j)] = 'h'
                    elif ci == cj:
                        topo8[(i,j)] = 'v'
                    else:
                        if ri < rj and ci <= cj:
                            topo8[(i,j)] = 'h'
                        else:
                            topo8[(i,j)] = 'v'
        for i in big:
            for j in small:
                if i < j:
                    topo8[(i,j)] = 'h'  # big is to the left
        topologies.append(topo8)
        
        # Topology 9: 4 big (2x2) + 6 small (3x2)
        topo9 = {}
        big = [0, 1, 2, 3]
        small = list(range(4, 10))
        # Big squares in 2x2 grid
        for i in big:
            for j in big:
                if i < j:
                    bi = i % 2
                    bj = j % 2
                    if bi == bj:
                        topo9[(i,j)] = 'v'  # same column
                    else:
                        topo9[(i,j)] = 'h'  # same row
        # Small squares in 3x2 grid
        for i in small:
            for j in small:
                if i < j:
                    si = i - 4
                    sj = j - 4
                    ri = si // 2
                    ci = si % 2
                    rj = sj // 2
                    cj = sj % 2
                    if ri == rj:
                        topo9[(i,j)] = 'h'
                    elif ci == cj:
                        topo9[(i,j)] = 'v'
                    else:
                        if ri < rj and ci <= cj:
                            topo9[(i,j)] = 'h'
                        else:
                            topo9[(i,j)] = 'v'
        # Big vs small: big is on top
        for i in big:
            for j in small:
                if i < j:
                    topo9[(i,j)] = 'v'
        topologies.append(topo9)
        
        # Add random topologies
        np.random.seed(42)
        for _ in range(100):
            if time.time() > deadline:
                break
            topo = {}
            for i in range(n):
                for j in range(i+1, n):
                    topo[(i,j)] = 'h' if np.random.random() < 0.5 else 'v'
            topologies.append(topo)
        
        return topologies

    def refine_with_local_search(c, s, max_rounds=100):
        """
        After LP gives an optimal solution for a topology, try to improve
        by relaxing the topology constraints locally.
        """
        c = c.copy()
        s = s.copy()
        best_val = objective(c, s)
        
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            order = np.random.permutation(n)
            for i in order:
                old_c = c[i].copy()
                old_s = s[i]
                
                # Try to increase side length
                max_s = min(2*old_c[0], 2*(1-old_c[0]), 2*old_c[1], 2*(1-old_c[1])) + 2*tol
                max_s = min(max_s, 1.0)
                
                if max_s <= old_s + 1e-12:
                    continue
                
                best_s = old_s
                best_c = old_c.copy()
                
                lo, hi = old_s, max_s
                for _ in range(40):
                    mid = (lo + hi) / 2
                    if mid <= best_s + 1e-13:
                        break
                    
                    found = False
                    # Try current position
                    ok = True
                    for j in range(n):
                        if j == i:
                            continue
                        t = (mid + s[j]) / 2 - tol
                        if abs(old_c[0] - c[j][0]) < t and abs(old_c[1] - c[j][1]) < t:
                            ok = False
                            break
                    if ok and old_c[0] - mid/2 >= -tol and old_c[0] + mid/2 <= 1+tol \
                            and old_c[1] - mid/2 >= -tol and old_c[1] + mid/2 <= 1+tol:
                        found = True
                        best_c = old_c.copy()
                    
                    if not found:
                        # Try random positions
                        for _ in range(100):
                            nx = np.random.uniform(mid/2, 1-mid/2)
                            ny = np.random.uniform(mid/2, 1-mid/2)
                            ok = True
                            for j in range(n):
                                if j == i:
                                    continue
                                t = (mid + s[j]) / 2 - tol
                                if abs(nx - c[j][0]) < t and abs(ny - c[j][1]) < t:
                                    ok = False
                                    break
                            if ok:
                                found = True
                                best_c = np.array([nx, ny])
                                break
                    
                    if found:
                        best_s = mid
                        lo = mid
                    else:
                        hi = mid
                
                if best_s > s[i] + 1e-10:
                    s[i] = best_s
                    c[i] = best_c
                    improved = True
            
            if not improved:
                break
        
        return c, s

    # Main optimization
    best_c = None
    best_s = None
    best_val = -1
    
    # Generate topologies
    topologies = generate_topologies()
    
    # Solve LP for each topology
    for idx, topo in enumerate(topologies):
        if time.time() > deadline:
            break
        
        c, s, val = solve_topology(topo)
        if c is not None and is_feasible(c, s):
            val = objective(c, s)
            if val > best_val:
                best_val = val
                best_c = c.copy()
                best_s = s.copy()
                
                # Try local refinement
                c_ref, s_ref = refine_with_local_search(c, s, max_rounds=50)
                if is_feasible(c_ref, s_ref):
                    val_ref = objective(c_ref, s_ref)
                    if val_ref > best_val:
                        best_val = val_ref
                        best_c = c_ref.copy()
                        best_s = s_ref.copy()
    
    # If no topology worked, use fallback
    if best_c is None:
        centers = np.array([
            [0.25, 0.25], [0.75, 0.25],
            [0.125, 0.875], [0.375, 0.875], [0.625, 0.875], [0.875, 0.875],
            [0.125, 0.625], [0.375, 0.625], [0.625, 0.625], [0.875, 0.625],
        ], dtype=np.float64)
        sides = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        best_c = centers
        best_s = sides
    
    # Final local search polish with more rounds
    c_final, s_final = refine_with_local_search(best_c, best_s, max_rounds=300)
    if is_feasible(c_final, s_final):
        val_final = objective(c_final, s_final)
        if val_final > best_val:
            best_val = val_final
            best_c = c_final.copy()
            best_s = s_final.copy()
    
    best_s = np.maximum(best_s, 1e-9)
    
    return best_c, best_s

# EVOLVE-BLOCK-END
