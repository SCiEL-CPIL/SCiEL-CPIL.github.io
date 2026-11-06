import numpy as np
import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    N = 129
    TARGET_LOW = 128
    TARGET_HIGH = 256
    TARGET_SLICE = slice(TARGET_LOW, TARGET_HIGH + 1)
    
    best_A = None
    best_peak = float('inf')
    
    def get_unordered_counts(A_vals):
        """Compute unordered pair counts for targets [128, 256]."""
        if len(A_vals) == 0:
            return np.zeros(257, dtype=np.int64)
        x = np.zeros(N, dtype=np.int64)
        x[A_vals] = 1
        y = np.convolve(x, x, mode="full")
        w = y[TARGET_LOW:TARGET_HIGH + 1].copy()
        ks = np.arange(TARGET_LOW, TARGET_HIGH + 1)
        is_even = (ks % 2 == 0)
        mids = ks // 2
        diag_corr = np.where(is_even, x[mids], 0)
        u_counts = (w + diag_corr) // 2
        return u_counts
    
    def evaluate_peak(u_counts):
        window = u_counts
        if np.min(window) == 0:
            return float('inf')
        return int(np.max(window))
    
    def make_structured_seed(rng, size_hint=50):
        """2-level even/odd construction for guaranteed coverage."""
        x = np.zeros(N, dtype=np.int64)
        evens = np.array([2 * k for k in range(65)])
        odds = np.array([2 * k + 1 for k in range(64)])
        rng.shuffle(evens)
        rng.shuffle(odds)
        
        n_even = max(20, size_hint // 2)
        n_odd = max(20, size_hint - n_even)
        n_even = min(n_even, len(evens))
        n_odd = min(n_odd, len(odds))
        
        x[evens[:n_even]] = 1
        x[odds[:n_odd]] = 1
        
        # Fix coverage
        A_vals = np.where(x == 1)[0]
        u_counts = get_unordered_counts(A_vals)
        while np.min(u_counts[TARGET_SLICE]) == 0:
            uncovered = np.where(u_counts[TARGET_SLICE] == 0)[0] + TARGET_LOW
            if len(uncovered) == 0:
                break
            best_c = None
            best_gain = -1
            for t in uncovered[:10]:
                lo = max(0, t - 128)
                hi = min(128, t)
                for c in range(lo, hi + 1):
                    if x[c] == 1:
                        continue
                    gain = 0
                    if TARGET_LOW <= c + c <= TARGET_HIGH and u_counts[c + c - TARGET_LOW] == 0:
                        gain += 1
                    for yy in A_vals:
                        if yy == c:
                            continue
                        s = c + yy
                        if TARGET_LOW <= s <= TARGET_HIGH and u_counts[s - TARGET_LOW] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_c = c
            if best_c is None or best_gain <= 0:
                break
            x[best_c] = 1
            A_vals = np.append(A_vals, best_c)
            u_counts = get_unordered_counts(A_vals)
        
        return x, A_vals
    
    def solve_ilp(peak_limit, timeout=10.0):
        """
        Solve the ILP to find a subset A with peak unordered multiplicity <= peak_limit.
        Returns (success, x_array) or (False, None).
        """
        # Enumerate all (i,j) pairs with i <= j and i+j in [128, 256]
        pairs = []
        for i in range(N):
            for j in range(i, N):
                s = i + j
                if TARGET_LOW <= s <= TARGET_HIGH:
                    pairs.append((i, j, s))
        
        n_pairs = len(pairs)
        n_vars = N + n_pairs  # x vars + z vars
        
        # Objective: minimize sum of x_i (to keep the set small)
        c = np.zeros(n_vars)
        c[:N] = 1.0
        
        # Bounds
        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)
        integrality = np.ones(n_vars)
        
        # Build constraint matrix
        constraints = []
        
        # Linearization constraints
        for idx, (i, j, s) in enumerate(pairs):
            z_idx = N + idx
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
        target_pairs = {k: [] for k in range(TARGET_LOW, TARGET_HIGH + 1)}
        for idx, (i, j, s) in enumerate(pairs):
            target_pairs[s].append(N + idx)
        
        for k in range(TARGET_LOW, TARGET_HIGH + 1):
            z_indices = target_pairs[k]
            if not z_indices:
                return False, None
            row = np.zeros(n_vars)
            for zi in z_indices:
                row[zi] = 1.0
            constraints.append((row, 1.0, float(peak_limit)))
        
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
        
        if res.status != 0 and res.status != 1:
            return False, None
        
        if res.x is None:
            return False, None
        
        x = res.x[:N].round().astype(int)
        return True, x
    
    def hill_climb(x, rng, max_time=5.0):
        """Local search to reduce peak."""
        A_vals = np.where(x == 1)[0]
        u_counts = get_unordered_counts(A_vals)
        current_peak = evaluate_peak(u_counts)
        if current_peak == float('inf'):
            return x
        
        best_peak = current_peak
        best_x = x.copy()
        
        start = time.time()
        T = 5.0
        T_min = 0.01
        cooling = 0.999
        
        while time.time() - start < max_time:
            if random.random() < 0.5:
                # Add element
                candidates = np.where(x == 0)[0]
                if len(candidates) == 0:
                    break
                k = candidates[random.randint(0, len(candidates) - 1)]
                
                x[k] = 1
                A_vals = np.append(A_vals, k)
                new_counts = get_unordered_counts(A_vals)
                new_peak = evaluate_peak(new_counts)
                
                delta = new_peak - current_peak
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_x = x.copy()
                else:
                    x[k] = 0
                    A_vals = A_vals[A_vals != k]
                    new_counts = u_counts
            else:
                # Remove element
                if len(A_vals) == 0:
                    break
                k_idx = random.randint(0, len(A_vals) - 1)
                k = A_vals[k_idx]
                
                x[k] = 0
                A_vals = np.delete(A_vals, k_idx)
                new_counts = get_unordered_counts(A_vals)
                new_peak = evaluate_peak(new_counts)
                
                delta = new_peak - current_peak
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_x = x.copy()
                else:
                    x[k] = 1
                    A_vals = np.append(A_vals, k)
                    new_counts = u_counts
            
            T *= cooling
            if T < T_min:
                break
        
        return best_x
    
    # Strategy 1: Try ILP with decreasing peak limits
    for peak_limit in [4, 5, 6, 7, 8]:
        if time.time() > deadline - 20:
            break
        
        success, x = solve_ilp(peak_limit, timeout=15.0)
        if success:
            A_vals = np.where(x == 1)[0]
            u_counts = get_unordered_counts(A_vals)
            actual_peak = evaluate_peak(u_counts)
            
            if actual_peak != float('inf') and actual_peak < best_peak:
                best_peak = actual_peak
                best_A = x.copy()
                
                # Refine with hill climbing
                if time.time() < deadline - 10:
                    rng = random.Random(int(time.time() * 1000))
                    refined_x = hill_climb(x.copy(), rng, max_time=5.0)
                    refined_vals = np.where(refined_x == 1)[0]
                    refined_counts = get_unordered_counts(refined_vals)
                    refined_peak = evaluate_peak(refined_counts)
                    
                    if refined_peak != float('inf') and refined_peak < best_peak:
                        best_peak = refined_peak
                        best_A = refined_x.copy()
    
    # Strategy 2: Structured seeds with local search
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        size = random.randint(38, 58)
        rng = random.Random(restart * 1000 + size)
        A_mask, A_vals = make_structured_seed(rng, size)
        u_counts = get_unordered_counts(A_vals)
        current_peak = evaluate_peak(u_counts)
        
        if current_peak != float('inf'):
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = A_mask.copy()
            
            # Hill climb
            if time.time() < deadline - 5:
                rng2 = random.Random(int(time.time() * 1000) + restart)
                refined_x = hill_climb(A_mask.copy(), rng2, max_time=3.0)
                refined_vals = np.where(refined_x == 1)[0]
                refined_counts = get_unordered_counts(refined_vals)
                refined_peak = evaluate_peak(refined_counts)
                
                if refined_peak != float('inf') and refined_peak < best_peak:
                    best_peak = refined_peak
                    best_A = refined_x.copy()
    
    if best_A is None:
        return list(range(64, 129))
    
    return np.where(best_A)[0].tolist()
