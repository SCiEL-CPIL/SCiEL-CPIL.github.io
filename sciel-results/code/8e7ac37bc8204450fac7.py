# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 292  # safety margin before 330s kill
    m = len(clauses)

    # --- Vectorized representation ---
    # For each variable v, store:
    #   var_clause_idx[v]: array of clause indices containing v
    #   var_clause_sign[v]: array of signs (+1/-1) of v in those clauses
    # A clause is satisfied by literal (v, s) iff assignment[v] == (s > 0)
    # sat_counts[c] = number of satisfied literals in clause c

    var_clause_idx = [None] * n
    var_clause_sign = [None] * n
    for v in range(n):
        idxs = []
        signs = []
        for c in range(m):
            # find if v is in clause c
            for lit in clauses[c]:
                if abs(lit) - 1 == v:
                    idxs.append(c)
                    signs.append(1 if lit > 0 else -1)
                    break
        var_clause_idx[v] = np.array(idxs, dtype=np.int32) if idxs else np.array([], dtype=np.int32)
        var_clause_sign[v] = np.array(signs, dtype=np.int8) if signs else np.array([], dtype=np.int8)

    # clause literal info: for each clause, list of (var, sign)
    clause_vars = []
    for c in clauses:
        cv = [(abs(lit) - 1, 1 if lit > 0 else -1) for lit in c]
        clause_vars.append(cv)

    # clause_var_array[c] = array of var indices, clause_sign_array[c] = array of signs
    clause_var_array = [np.array([v for v, s in cv], dtype=np.int32) for cv in clause_vars]
    clause_sign_array = [np.array([s for v, s in cv], dtype=np.int8) for cv in clause_vars]

    def compute_sat_counts(assignment):
        """Vectorized: for each clause, count satisfied literals."""
        counts = np.zeros(m, dtype=np.int32)
        for c in range(m):
            vars_c = clause_var_array[c]
            signs_c = clause_sign_array[c]
            # satisfied iff assignment[var] == (sign > 0)
            # assignment is boolean array; sign>0 means we want True, sign<0 means we want False
            sat = (assignment[vars_c] == (signs_c > 0))
            counts[c] = sat.sum()
        return counts

    def flip_update(assignment, v, sat_counts):
        """Flip variable v in assignment and update sat_counts in place. Returns new assignment value."""
        old_val = assignment[v]
        assignment[v] = not old_val
        idxs = var_clause_idx[v]
        signs = var_clause_sign[v]
        if len(idxs) == 0:
            return
        # For each clause containing v, recompute its sat count quickly
        # But we can do it incrementally: the literal for v in clause c was
        # satisfied iff old_val == (sign > 0). After flip, it's satisfied iff
        # (not old_val) == (sign > 0), i.e., the opposite.
        # So for each such clause, if the v-literal was satisfied before, decrement; else increment.
        was_sat = (old_val == (signs > 0))
        sat_counts[idxs] = sat_counts[idxs] + np.where(was_sat, -1, 1).astype(np.int32)

    rng = random.Random(12345)
    np_rng = np.random.default_rng(12345)

    def random_assignment():
        return np_rng.random(n) < 0.5

    def is_satisfied(sat_counts):
        return sat_counts.min() > 0

    def walksat_run(assignment, sat_counts, max_steps, p_greedy, rng_local):
        """Run WalkSAT on the given state. Returns (solved, assignment, sat_counts)."""
        for step in range(max_steps):
            if is_satisfied(sat_counts):
                return True, assignment, sat_counts
            unsat = np.where(sat_counts == 0)[0]
            if len(unsat) == 0:
                return True, assignment, sat_counts
            c = unsat[rng_local.randint(0, len(unsat) - 1)]
            vars_c = clause_var_array[c]
            signs_c = clause_sign_array[c]
            if rng_local.random() < p_greedy:
                # Greedy: pick literal that maximizes newly-satisfied clauses
                # Only consider literals that are currently unsatisfied
                cur_sat = (assignment[vars_c] == (signs_c > 0))
                candidates = np.where(~cur_sat)[0]
                if len(candidates) == 0:
                    # all literals satisfied? shouldn't happen if clause is unsat
                    v = vars_c[rng_local.randint(0, len(vars_c) - 1)]
                else:
                    best_score = -1
                    best_v = -1
                    for ci in candidates:
                        v = vars_c[ci]
                        idxs = var_clause_idx[v]
                        signs = var_clause_sign[v]
                        # Clauses that would become satisfied: those with sat_count == 1
                        # and where the v-literal is currently unsatisfied
                        one_sat = (sat_counts[idxs] == 1)
                        v_lit_unsat = (assignment[v] != (signs > 0))
                        score = int((one_sat & v_lit_unsat).sum())
                        if score > best_score:
                            best_score = score
                            best_v = v
                    v = best_v if best_v >= 0 else vars_c[rng_local.randint(0, len(vars_c) - 1)]
            else:
                v = vars_c[rng_local.randint(0, len(vars_c) - 1)]
            flip_update(assignment, v, sat_counts)
        return is_satisfied(sat_counts), assignment, sat_counts

    def simulated_annealing_run(assignment, sat_counts, max_steps, rng_local, np_rng_local):
        """SA with temperature cooling. Returns (solved, assignment, sat_counts)."""
        T = 5.0
        cooling = 0.9995
        for step in range(max_steps):
            if is_satisfied(sat_counts):
                return True, assignment, sat_counts
            # Pick a random variable
            v = int(np_rng_local.integers(n))
            old_unsat = int((sat_counts == 0).sum())
            flip_update(assignment, v, sat_counts)
            new_unsat = int((sat_counts == 0).sum())
            delta = new_unsat - old_unsat
            if delta <= 0 or (T > 0.01 and rng_local.random() < np.exp(-delta / T)):
                T *= cooling
            else:
                # Revert
                flip_update(assignment, v, sat_counts)
                T *= cooling
        return is_satisfied(sat_counts), assignment, sat_counts

    def walksat_random_restart_portfolio(time_limit):
        """Run multiple WalkSAT walks with different parameters."""
        best_unsat = float('inf')
        best_assign = None
        best_counts = None
        params_list = [
            (0.5, 3000),
            (0.7, 3000),
            (0.9, 2000),
            (0.3, 4000),
            (0.6, 2500),
        ]
        while time.time() < time_limit:
            for p_g, steps in params_list:
                if time.time() >= time_limit:
                    break
                assignment = random_assignment()
                sat_counts = compute_sat_counts(assignment)
                if is_satisfied(sat_counts):
                    return assignment
                cur_unsat = int((sat_counts == 0).sum())
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assign = assignment.copy()
                    best_counts = sat_counts.copy()
                solved, assignment, sat_counts = walksat_run(assignment, sat_counts, steps, p_g, rng)
                if solved:
                    return assignment
                cur_unsat = int((sat_counts == 0).sum())
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assign = assignment.copy()
                    best_counts = sat_counts.copy()
        if best_assign is not None:
            return best_assign
        return None

    def sa_portfolio(time_limit):
        """Run multiple SA runs."""
        best_unsat = float('inf')
        best_assign = None
        while time.time() < time_limit:
            assignment = random_assignment()
            sat_counts = compute_sat_counts(assignment)
            if is_satisfied(sat_counts):
                return assignment
            cur_unsat = int((sat_counts == 0).sum())
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assign = assignment.copy()
            solved, assignment, sat_counts = simulated_annealing_run(
                assignment, sat_counts, 5000, rng, np_rng
            )
            if solved:
                return assignment
            cur_unsat = int((sat_counts == 0).sum())
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assign = assignment.copy()
        if best_assign is not None:
            return best_assign
        return None

    # --- Preprocessing: pure literal elimination + unit propagation ---
    def preprocess(clauses_list, n_vars):
        """Apply pure literal elimination and unit propagation.
        Returns (reduced_clauses, fixed_assignments) where fixed_assignments is dict var->bool,
        or None if contradiction."""
        # Build active clause set
        active = list(clauses_list)
        fixed = {}
        # Pure literal elimination
        changed = True
        while changed:
            changed = False
            pos_vars = set()
            neg_vars = set()
            for c in active:
                for lit in c:
                    v = abs(lit) - 1
                    if lit > 0:
                        pos_vars.add(v)
                    else:
                        neg_vars.add(v)
            for v in pos_vars:
                if v not in neg_vars and v not in fixed:
                    fixed[v] = True
                    changed = True
            for v in neg_vars:
                if v not in pos_vars and v not in fixed:
                    fixed[v] = False
                    changed = True
            if changed:
                # Remove satisfied clauses, remove fixed literals
                new_active = []
                for c in active:
                    sat = False
                    new_c = []
                    for lit in c:
                        v = abs(lit) - 1
                        if v in fixed:
                            if (lit > 0) == fixed[v]:
                                sat = True
                                break
                        else:
                            new_c.append(lit)
                    if not sat:
                        if len(new_c) == 0:
                            return None  # contradiction
                        new_active.append(new_c)
                active = new_active
        # Unit propagation
        changed = True
        while changed:
            changed = False
            for i, c in enumerate(active):
                if len(c) == 1:
                    lit = c[0]
                    v = abs(lit) - 1
                    val = lit > 0
                    if v in fixed:
                        if fixed[v] != val:
                            return None
                    else:
                        fixed[v] = val
                        changed = True
            if changed:
                new_active = []
                for c in active:
                    sat = False
                    new_c = []
                    for lit in c:
                        v = abs(lit) - 1
                        if v in fixed:
                            if (lit > 0) == fixed[v]:
                                sat = True
                                break
                        else:
                            new_c.append(lit)
                    if not sat:
                        if len(new_c) == 0:
                            return None
                        new_active.append(new_c)
                active = new_active
        return active, fixed

    # Apply preprocessing
    reduced_clauses, fixed_assign = preprocess(clauses, n)
    if reduced_clauses is None:
        # Contradiction from preprocessing - shouldn't happen for planted SAT
        # Fall back to full search
        reduced_clauses = list(clauses)
        fixed_assign = {}

    # Build solution from fixed assignments
    final_assign = [None] * n
    for v, val in fixed_assign.items():
        final_assign[v] = val

    # If all variables are fixed, return
    unfixed_vars = [v for v in range(n) if final_assign[v] is None]
    if len(unfixed_vars) == 0:
        return [bool(x) for x in final_assign]

    # Check if reduced instance is empty
    if len(reduced_clauses) == 0:
        for v in range(n):
            if final_assign[v] is None:
                final_assign[v] = False
        return [bool(x) for x in final_assign]

    # --- Portfolio search on the reduced instance ---
    # We need to map the reduced instance back. For simplicity, we'll run the
    # portfolio on the original instance but initialize with fixed assignments.
    
    def make_initial_assignment():
        """Create an assignment respecting fixed variables, random for others."""
        assign = np.zeros(n, dtype=bool)
        for v, val in fixed_assign.items():
            assign[v] = val
        for v in unfixed_vars:
            assign[v] = np_rng.random() < 0.5
        return assign

    best_unsat_global = float('inf')
    best_assign_global = None

    def track_best(assignment, sat_counts):
        nonlocal best_unsat_global, best_assign_global
        cur = int((sat_counts == 0).sum())
        if cur < best_unsat_global:
            best_unsat_global = cur
            best_assign_global = assignment.copy()

    # Time-sliced portfolio: 40% WalkSAT, 30% SA, 30% WalkSAT with different params
    t_start = time.time()
    t_walk1_end = t_start + (deadline - t_start) * 0.40
    t_sa_end = t_walk1_end + (deadline - t_start) * 0.30
    # t_walk2_end = deadline

    # Phase 1: WalkSAT with p_greedy in [0.4, 0.8]
    while time.time() < t_walk1_end:
        assignment = make_initial_assignment()
        sat_counts = compute_sat_counts(assignment)
        if is_satisfied(sat_counts):
            return [bool(x) for x in assignment]
        track_best(assignment, sat_counts)
        p_g = rng.uniform(0.4, 0.8)
        solved, assignment, sat_counts = walksat_run(assignment, sat_counts, 3000, p_g, rng)
        if solved:
            return [bool(x) for x in assignment]
        track_best(assignment, sat_counts)

    # Phase 2: Simulated Annealing
    while time.time() < t_sa_end:
        assignment = make_initial_assignment()
        sat_counts = compute_sat_counts(assignment)
        if is_satisfied(sat_counts):
            return [bool(x) for x in assignment]
        track_best(assignment, sat_counts)
        solved, assignment, sat_counts = simulated_annealing_run(
            assignment, sat_counts, 4000, rng, np_rng
        )
        if solved:
            return [bool(x) for x in assignment]
        track_best(assignment, sat_counts)

    # Phase 3: WalkSAT with high randomness (exploration)
    while time.time() < deadline:
        assignment = make_initial_assignment()
        sat_counts = compute_sat_counts(assignment)
        if is_satisfied(sat_counts):
            return [bool(x) for x in assignment]
        track_best(assignment, sat_counts)
        p_g = rng.uniform(0.1, 0.5)
        solved, assignment, sat_counts = walksat_run(assignment, sat_counts, 2500, p_g, rng)
        if solved:
            return [bool(x) for x in assignment]
        track_best(assignment, sat_counts)

    # Return best found
    if best_assign_global is not None:
        # Apply fixed assignments
        for v, val in fixed_assign.items():
            best_assign_global[v] = val
        return [bool(x) for x in best_assign_global]
    
    # Fallback
    return [bool(x) for x in make_initial_assignment()]
# EVOLVE-BLOCK-END
