"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

BASE = 51975
DEADLINE = 145  # safety margin before 150s kill

def get_divisors(n):
    divs = []
    for d in range(3, n + 1, 2):
        if n % d == 0:
            divs.append(d)
    return sorted(divs)

MODULI = get_divisors(BASE)

def count_uncovered(classes):
    """Count residues not covered by any class. classes: list of (m, r)."""
    covered = np.zeros(BASE, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return int(np.sum(covered == 0))

def greedy_init(order=None):
    """Greedy: for each modulus in order, pick residue covering most uncovered."""
    if order is None:
        order = sorted(MODULI)
    covered = np.zeros(BASE, dtype=np.uint8)
    classes = []
    for m in order:
        best_r = 0
        best_count = -1
        for r in range(m):
            cnt = int(np.sum(covered[r::m] == 0))
            if cnt > best_count:
                best_count = cnt
                best_r = r
        classes.append((m, best_r))
        covered[best_r::m] = 1
    return classes

def local_search(classes, time_budget):
    """Local search: try changing residue of each class, accept if improves."""
    classes = list(classes)
    uncovered = count_uncovered(classes)
    start = time.time()
    
    while time.time() - start < time_budget:
        improved = False
        idx_order = list(range(len(classes)))
        random.shuffle(idx_order)
        
        for i in idx_order:
            if time.time() - start >= time_budget:
                break
            m, r = classes[i]
            best_r = r
            best_unc = uncovered
            
            candidates = [r]
            for _ in range(min(5, m - 1)):
                candidates.append(random.randrange(m))
            
            for r_new in candidates:
                if r_new == r:
                    continue
                new_classes = list(classes)
                new_classes[i] = (m, r_new)
                unc = count_uncovered(new_classes)
                if unc < best_unc:
                    best_unc = unc
                    best_r = r_new
            
            if best_r != r:
                classes[i] = (m, best_r)
                uncovered = best_unc
                improved = True
        
        if not improved:
            i = random.randrange(len(classes))
            m, r = classes[i]
            r_new = random.randrange(m)
            if r_new != r:
                new_classes = list(classes)
                new_classes[i] = (m, r_new)
                unc = count_uncovered(new_classes)
                if unc <= uncovered + 2:
                    classes = new_classes
                    uncovered = unc
                    improved = True
            if not improved:
                break
    
    return classes, uncovered

def solve_ilp(time_budget):
    """
    Formulate as ILP:
    Variables: x_{m,r} in {0,1} for each modulus m and residue r in [0, m)
    Objective: maximize sum_{pos} y_pos where y_pos = 1 if any class covers pos
    Constraints:
      - For each modulus m: sum_r x_{m,r} <= 1 (use each modulus at most once)
      - For each position pos: y_pos <= sum_{m,r: pos ≡ r mod m} x_{m,r}
    
    This is a large ILP. We'll use scipy.optimize.milp with CBC solver.
    """
    start_time = time.time()
    
    # Build variable mapping: (m, r) -> var_index
    var_map = {}
    var_list = []
    for m in MODULI:
        for r in range(m):
            var_map[(m, r)] = len(var_list)
            var_list.append((m, r))
    
    n_vars = len(var_list)
    
    # y_pos variables: one per position
    # Total variables: n_vars + BASE
    n_y = BASE
    total_vars = n_vars + n_y
    
    # Objective: maximize sum of y_pos => minimize -sum(y_pos)
    # c vector: [0 for x vars, -1 for y vars]
    c = np.zeros(total_vars)
    c[n_vars:] = -1.0
    
    # Constraints:
    # 1) For each modulus m: sum_r x_{m,r} <= 1
    # 2) For each position pos: y_pos - sum_{m,r: pos≡r mod m} x_{m,r} <= 0
    
    # Build constraint matrix (sparse)
    rows = []
    cols = []
    vals = []
    lower = []
    upper = []
    
    # Constraint 1: modulus usage (one per modulus)
    for m in MODULI:
        row_idx = len(lower)
        for r in range(m):
            var_idx = var_map[(m, r)]
            rows.append(row_idx)
            cols.append(var_idx)
            vals.append(1.0)
        lower.append(0.0)
        upper.append(1.0)
    
    # Constraint 2: coverage constraints
    # For each position pos: y_pos - sum_{(m,r) where pos ≡ r (mod m)} x_{m,r} <= 0
    # i.e., y_pos <= sum of x_{m,r} for classes covering pos
    # Rewrite: y_pos - sum x <= 0, lower=-inf, upper=0
    
    # This is expensive to build naively. Let's be smart:
    # For each (m, r), it covers positions r, r+m, r+2m, ...
    # So for each position pos, the classes covering it are those (m, r) where pos % m == r
    
    # Build mapping: pos -> list of (m, r) that cover it
    # This is O(BASE * num_moduli) which is ~51975 * 48 ≈ 2.5M, feasible
    
    # Actually, let's think about the size:
    # n_vars = sum of all moduli = sum_{m|51975, m>1} m
    # Let's compute this
    n_vars = sum(MODULI)
    total_vars = n_vars + BASE
    
    print(f"ILP: {n_vars} x-vars, {BASE} y-vars, total {total_vars}")
    
    # The constraint matrix will have:
    # - len(MODULI) rows for modulus constraints
    # - BASE rows for coverage constraints
    # Total rows: len(MODULI) + BASE
    
    # Each coverage constraint row has entries for each (m,r) where r = pos % m
    # That's one entry per modulus, so ~len(MODULI) nonzeros per row
    
    # Total nonzeros: len(MODULI) * sum_m 1 (for modulus constraints) + BASE * len(MODULI) (for coverage)
    # ≈ 2 * BASE * len(MODULI)
    
    num_moduli = len(MODULI)
    num_rows = num_moduli + BASE
    
    # Build sparse matrix
    row_idx = []
    col_idx = []
    data = []
    
    # Modulus constraints: for each m, sum_r x_{m,r} <= 1
    # Variable indexing: x vars are 0..n_vars-1, y vars are n_vars..total_vars-1
    # We need to know which variable index corresponds to which (m,r)
    
    var_index = {}
    vi = 0
    for m in MODULI:
        for r in range(m):
            var_index[(m, r)] = vi
            vi += 1
    
    # Modulus constraints
    for m in MODULI:
        r_idx = len(lower)
        for r in range(m):
            v = var_index[(m, r)]
            row_idx.append(r_idx)
            col_idx.append(v)
            data.append(1.0)
        lower.append(0.0)
        upper.append(1.0)
    
    # Coverage constraints: for each pos, y_pos - sum_{m} x_{m, pos%m} <= 0
    for pos in range(BASE):
        r_idx = len(lower)
        # y_pos variable
        y_var = n_vars + pos
        row_idx.append(r_idx)
        col_idx.append(y_var)
        data.append(1.0)
        
        # x variables: for each m, the residue is pos % m
        for m in MODULI:
            r = pos % m
            v = var_index[(m, r)]
            row_idx.append(r_idx)
            col_idx.append(v)
            data.append(-1.0)
        
        lower.append(-np.inf)
        upper.append(0.0)
    
    A = csr_matrix((data, (row_idx, col_idx)), shape=(num_rows, total_vars))
    
    constraints = LinearConstraint(A, lower, upper)
    
    # Bounds: x vars in [0,1] integer, y vars in [0,1] integer
    integrality = np.ones(total_vars)
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)
    
    bounds = Bounds(lb, ub)
    
    # Solve
    t_start = time.time()
    result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    t_elapsed = time.time() - t_start
    
    if result.status == 0:  # optimal
        # Extract solution
        x_sol = result.x[:n_vars]
        classes = []
        vi = 0
        for m in MODULI:
            for r in range(m):
                if x_sol[vi] > 0.5:
                    classes.append((m, r))
                vi += 1
        score = count_uncovered(classes)
        print(f"ILP solved in {t_elapsed:.1f}s, score={score}, classes={len(classes)}")
        return classes, score
    else:
        print(f"ILP status={result.status}, message={result.message}")
        return None, None

def construct():
    start_time = time.time()
    best_classes = None
    best_score = BASE  # worst case
    
    # Try multiple greedy initializations with different orderings
    rng = random.Random(42)
    
    # 1. Greedy with sorted moduli
    classes = greedy_init(sorted(MODULI))
    score = count_uncovered(classes)
    if score < best_score:
        best_score = score
        best_classes = list(classes)
    
    # 2. Greedy with reverse sorted
    if time.time() - start_time < DEADLINE:
        classes = greedy_init(sorted(MODULI, reverse=True))
        score = count_uncovered(classes)
        if score < best_score:
            best_score = score
            best_classes = list(classes)
    
    # 3. Greedy with random orderings
    for _ in range(5):
        if time.time() - start_time >= DEADLINE:
            break
        order = MODULI[:]
        rng.shuffle(order)
        classes = greedy_init(order)
        score = count_uncovered(classes)
        if score < best_score:
            best_score = score
            best_classes = list(classes)
    
    # 4. Try ILP if time permits
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 60:
        # ILP might take a while; give it up to 80s
        ilp_time_budget = min(80, DEADLINE - elapsed - 10)
        if ilp_time_budget > 15:
            ilp_classes, ilp_score = solve_ilp(ilp_time_budget)
            if ilp_classes is not None and ilp_score < best_score:
                best_score = ilp_score
                best_classes = ilp_classes
    
    # 5. Local search from best
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        remaining = DEADLINE - elapsed - 2
        if remaining > 5:
            classes, score = local_search(best_classes, remaining)
            if score < best_score:
                best_score = score
                best_classes = classes
    
    # 6. More restarts if time remains
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        for trial in range(20):
            if time.time() - start_time >= DEADLINE - 5:
                break
            order = MODULI[:]
            rng.shuffle(order)
            classes = greedy_init(order)
            score = count_uncovered(classes)
            if score < best_score:
                best_score = score
                best_classes = list(classes)
            # Quick local search
            remaining = DEADLINE - (time.time() - start_time)
            if remaining > 15 and score <= best_score + 50:
                classes, score = local_search(classes, min(remaining * 0.5, 10))
                if score < best_score:
                    best_score = score
                    best_classes = classes
    
    return [[m, r] for m, r in best_classes]
# EVOLVE-BLOCK-END
