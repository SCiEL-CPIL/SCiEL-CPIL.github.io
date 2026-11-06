"""Additive bases with few representations - integer linear programming rewrite.

We model the problem as a binary integer program:
  Variables: x_i in {0,1} for i in [0, 128]
  Objective: minimize peak (max representation multiplicity in [128, 256])
  
The unordered pair representation count for sum k is:
  U_k = (ordered_count[k] + diag[k]) / 2
where ordered_count[k] = sum_{i+j=k} x_i * x_j
and diag[k] = 1 if k is even and x_{k/2} = 1, else 0.

We minimize the peak using successive binary search on the peak value,
solving an ILP at each step with scipy.optimize.milp.
"""
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

_N = 128
_LEN = 129  # indices 0..128
_T0 = 128
_T1 = 256
_WIN = _T1 - _T0 + 1  # 129 targets

_START = time.time()
_DEADLINE = 85.0  # safety margin before the 90s kill


def _elapsed():
    return time.time() - _START


def _build_ordered_coefficients():
    """
    Build the coefficient matrix for ordered pair counts.
    For each target k in [128, 256], ordered_count[k] = sum_{i+j=k} x_i * x_j.
    
    We linearize: x_i * x_j <= x_i, x_i * x_j <= x_j, x_i * x_j >= x_i + x_j - 1.
    But this creates too many variables. Instead, we use a different approach.
    
    Actually, for the ILP, we can use the fact that x_i * x_j is a bilinear term.
    Since x_i, x_j in {0,1}, we can introduce auxiliary variables z_{i,j} = x_i * x_j
    for i <= j (to handle unordered pairs), or just work with ordered and divide.
    
    Let me use a cleaner formulation:
    For unordered pairs, the count for sum k is:
      U_k = sum_{i <= j, i+j=k} x_i * x_j
    
    We introduce z_{i,j} for all 0 <= i <= j <= 128 with i+j in [128, 256].
    z_{i,j} = x_i * x_j.
    
    Linearization constraints for each (i,j) with i < j:
      z_{i,j} <= x_i
      z_{i,j} <= x_j
      z_{i,j} >= x_i + x_j - 1
      z_{i,j} >= 0
    
    For i == j:
      z_{i,i} = x_i (since x_i * x_i = x_i for binary)
    
    Then U_k = sum_{i <= j, i+j=k} z_{i,j}.
    
    Objective: minimize peak M such that U_k <= M for all k in [128, 256],
               and U_k >= 1 for all k in [128, 256].
    """
    pass


def _solve_ilp(peak_limit, timeout=10.0):
    """
    Solve the ILP to find a subset A with peak unordered multiplicity <= peak_limit.
    Returns (success, x_array) or (False, None).
    """
    # Build the problem
    # Variables:
    #   x_0, ..., x_128 (129 binary variables)
    #   z_{i,j} for 0 <= i <= j <= 128, i+j in [128, 256]
    #   M (peak variable, but we fix M = peak_limit for feasibility check)
    
    # Let's enumerate all (i,j) pairs with i <= j and i+j in [128, 256]
    pairs = []
    for i in range(_LEN):
        for j in range(i, _LEN):
            s = i + j
            if _T0 <= s <= _T1:
                pairs.append((i, j, s))
    
    n_pairs = len(pairs)
    n_vars = _LEN + n_pairs  # x vars + z vars
    
    # Objective: minimize sum of x_i (to keep the set small, as a tiebreaker)
    # Or just feasibility. Let's minimize sum x_i.
    c = np.zeros(n_vars)
    c[:_LEN] = 1.0  # minimize |A|
    
    # Bounds
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    integrality = np.ones(n_vars)  # all integer
    
    # Constraints:
    # 1. For each pair (i, j, s) with i < j:
    #    z_{i,j} <= x_i  =>  -z_{i,j} + x_i >= 0
    #    z_{i,j} <= x_j  =>  -z_{i,j} + x_j >= 0
    #    z_{i,j} >= x_i + x_j - 1  =>  -x_i - x_j + z_{i,j} >= -1
    # 2. For each pair (i, i, s): z_{i,i} = x_i  =>  z_{i,i} - x_i = 0
    # 3. For each target k in [128, 256]:
    #    U_k = sum_{(i,j,s=k)} z_{i,j} >= 1
    #    U_k = sum_{(i,j,s=k)} z_{i,j} <= peak_limit
    
    # Build constraint matrix
    # We'll collect all constraints in a list, then build the matrix
    
    constraints = []  # list of (row_vector, lower, upper)
    
    # Linearization constraints
    for idx, (i, j, s) in enumerate(pairs):
        z_idx = _LEN + idx
        if i < j:
            # z <= x_i: -z + x_i >= 0
            row = np.zeros(n_vars)
            row[z_idx] = -1.0
            row[i] = 1.0
            constraints.append((row, 0.0, np.inf))
            
            # z <= x_j: -z + x_j >= 0
            row = np.zeros(n_vars)
            row[z_idx] = -1.0
            row[j] = 1.0
            constraints.append((row, 0.0, np.inf))
            
            # z >= x_i + x_j - 1: -x_i - x_j + z >= -1
            row = np.zeros(n_vars)
            row[i] = -1.0
            row[j] = -1.0
            row[z_idx] = 1.0
            constraints.append((row, -1.0, np.inf))
        else:
            # i == j: z = x_i => z - x_i = 0
            row = np.zeros(n_vars)
            row[z_idx] = 1.0
            row[i] = -1.0
            constraints.append((row, 0.0, 0.0))
    
    # Coverage and peak constraints
    # For each target k, collect the z indices
    target_pairs = {k: [] for k in range(_T0, _T1 + 1)}
    for idx, (i, j, s) in enumerate(pairs):
        target_pairs[s].append(_LEN + idx)
    
    for k in range(_T0, _T1 + 1):
        z_indices = target_pairs[k]
        if not z_indices:
            # This target cannot be covered - infeasible
            return False, None
        row = np.zeros(n_vars)
        for zi in z_indices:
            row[zi] = 1.0
        # U_k >= 1 and U_k <= peak_limit
        constraints.append((row, 1.0, float(peak_limit)))
    
    # Build the linear constraint
    if not constraints:
        return False, None
    
    A_mat = np.array([c[0] for c in constraints])
    lb_vec = np.array([c[1] for c in constraints])
    ub_vec = np.array([c[2] for c in constraints])
    
    lc = LinearConstraint(A_mat, lb_vec, ub_vec)
    bounds = Bounds(lb, ub)
    
    try:
        res = milp(
            c=c,
            constraints=lc,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": timeout, "presolve": True}
        )
    except Exception:
        return False, None
    
    if res.status != 0 and res.status != 1:  # 0=optimal, 1=feasible
        return False, None
    
    if res.x is None:
        return False, None
    
    x = res.x[:_LEN].round().astype(int)
    return True, x


def _unordered_counts(x):
    """Compute unordered pair counts for targets [128, 256]."""
    y = np.convolve(x.astype(np.int64), x.astype(np.int64), mode="full")
    w = y[_T0:_T1 + 1].copy()
    ks = np.arange(_T0, _T1 + 1)
    is_even = (ks % 2 == 0)
    mids = ks // 2
    diag_corr = np.where(is_even, x[mids], 0)
    u_counts = (w + diag_corr) // 2
    return u_counts


def _valid(u_counts):
    return bool(np.all(u_counts > 0))


def _peak(u_counts):
    return int(u_counts.max())


def _make_greedy_seed(rng):
    """Build a valid seed using greedy coverage."""
    x = np.zeros(_LEN, dtype=np.int64)
    candidates = np.arange(_LEN)
    rng.shuffle(candidates)
    
    u = _unordered_counts(x)
    while not _valid(u):
        uncovered = np.where(u == 0)[0] + _T0
        if len(uncovered) == 0:
            break
        
        best_c = None
        best_gain = -1
        for c in candidates:
            if x[c] == 1:
                continue
            gain = 0
            for t in uncovered:
                a = t - c
                if 0 <= a < _LEN and x[a] == 1:
                    gain += 1
                if 2 * c == t:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
        
        if best_c is None or best_gain <= 0:
            best_c = candidates[0]
        
        x[best_c] = 1
        u = _unordered_counts(x)
    
    return x


def _hill_climb(x, rng, max_time):
    """Local search to reduce peak."""
    u = _unordered_counts(x)
    if not _valid(u):
        return x
    best_peak = _peak(u)
    best_x = x.copy()
    current_peak = best_peak
    
    while _elapsed() < max_time:
        improved = False
        in_A = np.where(x == 1)[0]
        not_in_A = np.where(x == 0)[0]
        
        if len(in_A) == 0 or len(not_in_A) == 0:
            break
        
        # Try random swaps
        for _ in range(100):
            if _elapsed() > max_time:
                break
            i = rng.choice(in_A)
            j = rng.choice(not_in_A)
            
            x_new = x.copy()
            x_new[i] = 0
            x_new[j] = 1
            
            u_new = _unordered_counts(x_new)
            if not _valid(u_new):
                continue
            
            peak_new = _peak(u_new)
            if peak_new < current_peak:
                x = x_new
                u = u_new
                current_peak = peak_new
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_x = x.copy()
                improved = True
        
        if not improved:
            break
    
    return best_x


def construct():
    global _START
    _START = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Get a valid seed
    x = _make_greedy_seed(rng)
    u = _unordered_counts(x)
    if not _valid(u):
        # Fallback: full set
        x = np.ones(_LEN, dtype=np.int64)
        u = _unordered_counts(x)
    
    current_peak = _peak(u)
    best_x = x.copy()
    best_peak = current_peak
    
    # Phase 2: Local search to improve
    remaining = _DEADLINE - _elapsed() - 5
    if remaining > 1:
        x_opt = _hill_climb(x, rng, max_time=min(remaining, 20))
        u_opt = _unordered_counts(x_opt)
        if _valid(u_opt):
            peak_opt = _peak(u_opt)
            if peak_opt < best_peak:
                best_peak = peak_opt
                best_x = x_opt.copy()
    
    # Phase 3: Try ILP to find a better peak
    remaining = _DEADLINE - _elapsed() - 3
    if remaining > 2:
        # Binary search on peak
        lo = 1
        hi = best_peak
        found_better = False
        
        while lo <= hi and _elapsed() < _DEADLINE - 3:
            mid = (lo + hi) // 2
            success, x_ilp = _solve_ilp(mid, timeout=min(15.0, _DEADLINE - _elapsed() - 2))
            
            if success:
                u_ilp = _unordered_counts(x_ilp)
                if _valid(u_ilp):
                    peak_ilp = _peak(u_ilp)
                    if peak_ilp < best_peak:
                        best_peak = peak_ilp
                        best_x = x_ilp.copy()
                        found_better = True
                    hi = mid - 1
                else:
                    hi = mid - 1
            else:
                lo = mid + 1
        
        # If ILP found something, do local search on it
        if found_better and _elapsed() < _DEADLINE - 2:
            x_final = _hill_climb(best_x, rng, max_time=_DEADLINE - _elapsed() - 1)
            u_final = _unordered_counts(x_final)
            if _valid(u_final):
                peak_final = _peak(u_final)
                if peak_final < best_peak:
                    best_peak = peak_final
                    best_x = x_final.copy()
    
    # Convert to sorted list
    result = [int(i) for i in np.where(best_x == 1)[0]]
    return result
