<<<<<<< SEARCH
"""Model-and-optimize: enumerate allocation patterns, compute minimal IC payments, pick best revenue."""
import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    # Parse distribution
    values = [0, 1, 2, 3]
    
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            else:
                probs = None
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = None
    else:
        probs = None
    
    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = probs / probs.sum()
    
    # Parse support
    if support is not None:
        if isinstance(support, (list, tuple)):
            if len(support) == 16:
                types = [tuple(t) for t in support]
            elif len(support) == 2:
                v1_vals = list(support[0])
                v2_vals = list(support[1])
                types = [(v1, v2) for v1 in v1_vals for v2 in v2_vals]
            else:
                types = [(i, j) for i in values for j in values]
        else:
            types = [(i, j) for i in values for j in values]
    else:
        types = [(i, j) for i in values for j in values]
    
    n = 16
    assert len(types) == n, f"Expected 16 types, got {len(types)}"
    
    # Value matrix: val[t] = (v1, v2)
    val_matrix = np.array(types, dtype=float)  # shape (16, 2)
    
    # For a single additive bidder, the IC payment for a given allocation rule x(t)
    # is uniquely determined (up to IR). The minimal payment satisfying IC and IR is:
    # p(t) = max(0, max_{t'} [v(t)·x(t') - v(t')·x(t') + p(t')])  -- but this is recursive.
    #
    # Actually, for a single agent, the IC constraint gives:
    # p(t) <= v(t)·x(t) - v(t)·x(t') + v(t')·x(t') + p(t') for all t'
    # And IR gives p(t) <= v(t)·x(t).
    # The maximal feasible p(t) (minimizing revenue loss) given the allocation is found by
    # solving a system of linear inequalities. But actually, the payment is uniquely
    # determined by the allocation rule via the "critical type" formula.
    #
    # For a single agent, the mechanism is fully determined by the allocation rule.
    # The payment p(t) is the unique solution to:
    # p(t) = max(0, max_{t'} [v(t)·(x(t') - x(t)) + v(t')·x(t')])
    # No, that's not right either. Let me think more carefully.
    #
    # The IC constraint: for type t reporting t', utility must not increase:
    # v(t)·x(t) - p(t) >= v(t)·x(t') - p(t')
    # => p(t) - p(t') <= v(t)·(x(t) - x(t'))
    #
    # This is a system of linear inequalities in the payment vector p. Given the allocation x,
    # we want the maximum revenue, i.e., maximize sum_t probs[t]*p(t) subject to:
    # 1. p(t) - p(t') <= v(t)·(x(t) - x(t')) for all t, t'
    # 2. p(t) <= v(t)·x(t) for all t  (IR)
    # 3. p(t) >= 0 for all t
    #
    # This is an LP in 16 variables (the payments), given the allocation.
    #
    # Key insight: For a single agent, the allocation rule x(t) can be any function from
    # T to [0,1]^2. But since we have a finite support, we can enumerate or optimize.
    #
    # Strategy: The allocation rule for each good independently determines part of the
    # mechanism. For good 1, x1(t) depends on t=(v1,v2). The IC constraint couples the
    # two goods through the payment.
    #
    # Better approach: Use the fact that for a single agent, the optimal mechanism
    # (maximizing revenue subject to IC and IR) has a specific structure. The payment
    # is:
    # p(t) = v(t)·x(t) - u(t)
    # where u(t) is the utility, and u(t) = max(0, max_{t'} [v(t)·(x(t') - x(t)) + u(t')])
    # This is still recursive.
    #
    # Practical approach: For each allocation pattern (binary x1, x2 for each type),
    # solve the 16-variable LP for the optimal payments.
    #
    # But 2^32 allocation patterns is too many. Instead, note that x1(t) for good 1
    # should be non-decreasing in v1 (monotonicity), and x2(t) non-decreasing in v2.
    # This reduces the search space significantly.
    #
    # For monotone allocations: x1(v1, v2) depends only on v1 (for a single good,
    # the optimal allocation for good 1 is a function of v1 only, by symmetry/structure).
    # Actually, with two goods, x1 can depend on both v1 and v2, but in the optimal
    # mechanism for a single agent with additive values, x1(t) = f1(v1) and x2(t) = f2(v2)
    # for some threshold functions.
    #
    # Let's assume x1(t) = 1{v1 >= a} and x2(t) = 1{v2 >= b} for some thresholds a, b.
    # This gives 4*4 = 16 candidate mechanisms. We can also consider partial allocations.
    #
    # Even better: enumerate all monotone binary allocation rules. For good 1, x1 is
    # a non-decreasing function of v1 (ignoring v2 for now). The possible x1 patterns
    # over v1 in {0,1,2,3} are: all 0, {v1>=1}, {v1>=2}, {v1>=3}, all 1. That's 5 patterns.
    # Similarly for x2. So 5*5 = 25 candidate mechanisms.
    #
    # But x1 can also depend on v2. Let's be more general: for each type t=(v1,v2),
    # x1(t) can be 0 or 1, subject to monotonicity: x1 is non-decreasing in v1 for fixed v2,
    # and non-decreasing in v2 for fixed v1? No, x1 doesn't need to be monotone in v2.
    # Actually, for IC, x1(t) must be non-decreasing in v1 when v2 is held fixed (single-crossing).
    #
    # Let me just try the threshold approach first: x1(t) = 1{v1 >= a}, x2(t) = 1{v2 >= b}.
    
    best_revenue = -np.inf
    best_alloc = None
    best_pay = None
    
    # Candidate thresholds
    thresholds = [0, 1, 2, 3, 4]  # 4 means never allocate
    
    for a in thresholds:
        for b in thresholds:
            if time.time() > deadline:
                break
            # Allocation rule
            alloc = np.zeros((16, 2))
            for t in range(16):
                v1_t, v2_t = val_matrix[t]
                alloc[t, 0] = 1.0 if v1_t >= a else 0.0
                alloc[t, 1] = 1.0 if v2_t >= b else 0.0
            
            # Solve for optimal payments
            pay = _solve_payments(alloc, val_matrix, probs)
            if pay is None:
                continue
            
            revenue = np.sum(probs * pay)
            if revenue > best_revenue:
                best_revenue = revenue
                best_alloc = alloc.copy()
                best_pay = pay.copy()
        
        if time.time() > deadline:
            break
    
    # If no threshold mechanism found, try a more general LP-based approach
    if best_alloc is None:
        best_alloc = np.ones((16, 2))
        best_pay = np.zeros(16)
    
    # Verify and return
    best_pay = np.maximum(best_pay, 0)
    best_alloc = np.clip(best_alloc, 0, 1)
    
    return {'allocations': best_alloc, 'payments': best_pay}
=======
"""Model-and-optimize: enumerate allocation patterns, compute minimal IC payments, pick best revenue."""
import numpy as np
from scipy.optimize import linprog
import time
from itertools import product

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    # Parse distribution
    values = [0, 1, 2, 3]
    
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            else:
                probs = None
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = None
    else:
        probs = None
    
    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = probs / probs.sum()
    
    # Parse support
    if support is not None:
        if isinstance(support, (list, tuple)):
            if len(support) == 16:
                types = [tuple(t) for t in support]
            elif len(support) == 2:
                v1_vals = list(support[0])
                v2_vals = list(support[1])
                types = [(v1, v2) for v1 in v1_vals for v2 in v2_vals]
            else:
                types = [(i, j) for i in values for j in values]
        else:
            types = [(i, j) for i in values for j in values]
    else:
        types = [(i, j) for i in values for j in values]
    
    n = 16
    assert len(types) == n, f"Expected 16 types, got {len(types)}"
    
    # Value matrix: val[t] = (v1, v2)
    val_matrix = np.array(types, dtype=float)  # shape (16, 2)
    
    best_revenue = -np.inf
    best_alloc = None
    best_pay = None
    
    # Phase 1: Threshold mechanisms (fast baseline)
    thresholds = [0, 1, 2, 3, 4]
    
    for a in thresholds:
        for b in thresholds:
            if time.time() > deadline:
                break
            alloc = np.zeros((16, 2))
            for t in range(16):
                v1_t, v2_t = val_matrix[t]
                alloc[t, 0] = 1.0 if v1_t >= a else 0.0
                alloc[t, 1] = 1.0 if v2_t >= b else 0.0
            
            pay = _solve_payments(alloc, val_matrix, probs)
            if pay is None:
                continue
            
            revenue = np.sum(probs * pay)
            if revenue > best_revenue:
                best_revenue = revenue
                best_alloc = alloc.copy()
                best_pay = pay.copy()
        
        if time.time() > deadline:
            break
    
    # Phase 2: Enumerate all monotone binary allocation patterns for each good
    # For good 1: x1(v1, v2) must be non-decreasing in v1 for fixed v2
    # For good 2: x2(v1, v2) must be non-decreasing in v2 for fixed v1
    # 
    # A monotone pattern for good 1 is determined by a threshold for each v2 level:
    # x1(v1, v2) = 1 if v1 >= threshold[v2], where threshold[v2] in {0,1,2,3,4}
    # This gives 5^4 = 625 patterns per good, so 625^2 = 390625 total.
    # But we can be smarter: many of these are equivalent or dominated.
    #
    # Actually, let's enumerate all monotone binary patterns more directly.
    # For good 1, the allocation x1 is a 4x4 matrix (rows=v1, cols=v2) that is
    # non-decreasing down each column. The number of such matrices is C(8,4) = 70.
    # Similarly for good 2: non-decreasing across each row, also 70 patterns.
    # Total: 70*70 = 4900 candidate mechanisms.
    
    # Generate all monotone binary patterns for a 4x4 grid (non-decreasing in one direction)
    # For good 1: non-decreasing in v1 (down columns)
    # For good 2: non-decreasing in v2 (across rows)
    
    def _generate_monotone_patterns_1d(n_levels, n_cols):
        """Generate all non-decreasing binary patterns of length n_levels for n_cols columns.
        Each column is independently non-decreasing. Returns list of (n_cols, n_levels) arrays."""
        # For a single column of length n_levels, non-decreasing binary patterns are:
        # 000...0, 000...01, 000...011, ..., 111...1
        # There are n_levels+1 such patterns.
        # For n_cols independent columns, there are (n_levels+1)^n_cols patterns.
        patterns = []
        for combo in product(range(n_levels + 1), repeat=n_cols):
            pat = np.zeros((n_cols, n_levels))
            for col, thresh in enumerate(combo):
                # thresh = number of zeros, so ones start at index thresh
                pat[col, thresh:] = 1.0
            patterns.append(pat)
        return patterns
    
    # For good 1: x1[v1, v2] non-decreasing in v1 (for each v2)
    # So for each v2 column, the pattern is non-decreasing in v1
    # Pattern shape: (4 v1 levels, 4 v2 cols), each column non-decreasing
    patterns_good1 = _generate_monotone_patterns_1d(4, 4)  # 5^4 = 625 patterns
    # patterns_good1[i] has shape (4, 4) where patterns_good1[i][v1, v2] = x1(v1, v2)
    
    # For good 2: x2[v1, v2] non-decreasing in v2 (for each v1)
    # So for each v1 row, the pattern is non-decreasing in v2
    # We need: for each row v1, x2[v1, :] is non-decreasing
    # This is equivalent to: transpose, then non-decreasing in v1 for each v2 col
    patterns_good2 = _generate_monotone_patterns_1d(4, 4)  # 5^4 = 625 patterns
    # patterns_good2[j] has shape (4, 4) where patterns_good2[j][v1, v2] = x2(v1, v2)
    # But we need it non-decreasing in v2 for fixed v1, which means:
    # patterns_good2[j][v1, v2] should be non-decreasing in v2
    # Our _generate_monotone_patterns_1d gives non-decreasing in the first index (v1)
    # So we need to transpose: if p = _generate_monotone_patterns_1d(4,4)[j],
    # then p.T has shape (4,4) where p.T[v2, v1] is non-decreasing in v2 for fixed v1
    # We want x2[v1, v2] non-decreasing in v2, so x2 = p.T
    
    if time.time() < deadline:
        for p1_idx in range(len(patterns_good1)):
            if time.time() > deadline:
                break
            x1_pat = patterns_good1[p1_idx]  # shape (4, 4): x1[v1, v2]
            
            for p2_idx in range(len(patterns_good2)):
                if time.time() > deadline:
                    break
                x2_pat = patterns_good2[p2_idx].T  # shape (4, 4): x2[v1, v2]
                
                # Build allocation matrix
                alloc = np.zeros((16, 2))
                for t in range(16):
                    v1_t, v2_t = int(val_matrix[t, 0]), int(val_matrix[t, 1])
                    alloc[t, 0] = x1_pat[v1_t, v2_t]
                    alloc[t, 1] = x2_pat[v1_t, v2_t]
                
                pay = _solve_payments(alloc, val_matrix, probs)
                if pay is None:
                    continue
                
                revenue = np.sum(probs * pay)
                if revenue > best_revenue:
                    best_revenue = revenue
                    best_alloc = alloc.copy()
                    best_pay = pay.copy()
    
    # If no mechanism found, fallback
    if best_alloc is None:
        best_alloc = np.ones((16, 2))
        best_pay = np.zeros(16)
    
    # Verify and return
    best_pay = np.maximum(best_pay, 0)
    best_alloc = np.clip(best_alloc, 0, 1)
    
    return {'allocations': best_alloc, 'payments': best_pay}
>>>>>>> REPLACE
