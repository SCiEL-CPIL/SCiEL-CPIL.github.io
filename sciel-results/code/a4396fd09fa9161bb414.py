"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def _build_clause_data(n, clauses):
    """Precompute clause structure as numpy arrays for vectorized operations."""
    m = len(clauses)
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_sgn = np.zeros((m, 3), dtype=np.int8)
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            clause_vars[i, j] = abs(lit) - 1
            clause_sgn[i, j] = 1 if lit > 0 else -1
    # For each variable, precompute numpy index arrays for fast vectorized gain
    var_cidx = [None] * n
    var_lidx = [None] * n
    for v in range(n):
        var_cidx[v] = []
        var_lidx[v] = []
    for i in range(m):
        for j in range(3):
            v = int(clause_vars[i, j])
            var_cidx[v].append(i)
            var_lidx[v].append(j)
    # Convert to numpy arrays once
    var_cidx_arr = [np.array(var_cidx[v], dtype=np.int32) if var_cidx[v] else np.zeros(0, dtype=np.int32) for v in range(n)]
    var_lidx_arr = [np.array(var_lidx[v], dtype=np.int32) if var_lidx[v] else np.zeros(0, dtype=np.int32) for v in range(n)]
    return clause_vars, clause_sgn, var_cidx_arr, var_lidx_arr


def _compute_sat_state(clause_vars, clause_sgn, assignment):
    """Compute per-clause satisfaction state: sat_count (how many literals satisfied)."""
    var_idx = clause_vars
    signs = clause_sgn
    truth = assignment[var_idx]
    sat_lit = (signs == 1) & truth | (signs == -1) & (~truth)
    sat_count = sat_lit.sum(axis=1)
    return sat_count


def _phase_seed(n, clauses, rng):
    """Initialize assignment using phase information from clause polarities."""
    pos_count = [0] * n
    neg_count = [0] * n
    for clause in clauses:
        for lit in clause:
            if lit > 0:
                pos_count[lit - 1] += 1
            else:
                neg_count[-lit - 1] += 1
    assignment = []
    for i in range(n):
        if pos_count[i] > neg_count[i]:
            assignment.append(True)
        elif neg_count[i] > pos_count[i]:
            assignment.append(False)
        else:
            assignment.append(rng.random() < 0.5)
    return assignment


def _count_unsat(assignment, clauses):
    """Count number of unsatisfied clauses."""
    count = 0
    for clause in clauses:
        sat = False
        for lit in clause:
            if assignment[abs(lit) - 1] == (lit > 0):
                sat = True
                break
        if not sat:
            count += 1
    return count


def _vectorized_impact_walksat(assignment, clause_vars, clause_sgn, var_cidx_arr, var_lidx_arr, n, m, rng, max_steps, deadline):
    """
    Vectorized WalkSAT with impact-based literal selection using precomputed index arrays.
    """
    assignment = np.array(assignment, dtype=bool)
    sat_count = _compute_sat_state(clause_vars, clause_sgn, assignment)

    for step in range(max_steps):
        if time.time() >= deadline:
            break

        unsat_mask = sat_count == 0
        n_unsat = int(unsat_mask.sum())
        if n_unsat == 0:
            break

        # Pick a random unsatisfied clause
        unsat_indices = np.where(unsat_mask)[0]
        ci = int(rng.choice(unsat_indices.tolist()))

        lits_vars = clause_vars[ci]  # (3,)

        # U-shaped noise: high early for exploration, low middle for exploitation, high late for escape
        progress = step / max(1, max_steps)
        if progress < 0.2:
            noise = 0.40 - 0.20 * (progress / 0.2)
        elif progress < 0.7:
            noise = 0.20
        else:
            noise = 0.20 + 0.25 * ((progress - 0.7) / 0.3)

        if rng.random() < noise:
            li = rng.randint(0, 2)
            v = int(lits_vars[li])
        else:
            # Greedy: compute gain for each of the 3 literals using precomputed arrays
            gains = [0, 0, 0]
            for li in range(3):
                v = int(lits_vars[li])
                old_val = assignment[v]

                cidx = var_cidx_arr[v]
                lidx = var_lidx_arr[v]
                k = len(cidx)
                if k == 0:
                    gains[li] = 0
                    continue

                s2 = clause_sgn[cidx, lidx]

                if old_val:
                    # old_val=True: positive literals are sat, negative are not
                    old_lit_sat = (s2 == 1)
                    new_lit_sat = (s2 == -1)
                else:
                    old_lit_sat = (s2 == -1)
                    new_lit_sat = (s2 == 1)

                sc = sat_count[cidx]

                newly_sat = (sc == 0) & new_lit_sat
                newly_unsat = (sc == 1) & old_lit_sat & (~new_lit_sat)

                gains[li] = int(newly_sat.sum()) - int(newly_unsat.sum())

            # Pick max gain with random tie-break
            max_gain = max(gains)
            best_candidates = [li for li in range(3) if gains[li] == max_gain]
            li = best_candidates[rng.randint(0, len(best_candidates) - 1)]
            v = int(lits_vars[li])

        # Apply the flip
        old_val = assignment[v]
        assignment[v] = not assignment[v]

        # Update sat_count for all clauses containing v
        cidx = var_cidx_arr[v]
        lidx = var_lidx_arr[v]
        if len(cidx) > 0:
            s2 = clause_sgn[cidx, lidx]
            if old_val:
                # Was True, now False: positive lit goes sat->unsat, negative goes unsat->sat
                gained = (s2 == -1)
                lost = (s2 == 1)
            else:
                gained = (s2 == 1)
                lost = (s2 == -1)
            sat_count[cidx] += gained.astype(np.int32) - lost.astype(np.int32)

    return assignment.tolist()


def _vectorized_minconflicts(assignment, clauses, n, clause_vars, clause_sgn, var_cidx_arr, var_lidx_arr, rng, max_steps, deadline):
    """
    Vectorized MinConflicts using precomputed index arrays.
    """
    assignment = np.array(assignment, dtype=bool)
    sat_count = _compute_sat_state(clause_vars, clause_sgn, assignment)

    for step in range(max_steps):
        if time.time() >= deadline:
            break

        n_unsat = int((sat_count == 0).sum())
        if n_unsat == 0:
            break

        unsat_indices = np.where(sat_count == 0)[0]
        ci = int(rng.choice(unsat_indices.tolist()))

        best_var = -1
        best_cost = float('inf')

        for li in range(3):
            v = int(clause_vars[ci, li])
            old_val = assignment[v]

            cidx = var_cidx_arr[v]
            lidx = var_lidx_arr[v]
            k = len(cidx)
            if k == 0:
                cost = n_unsat
            else:
                s2 = clause_sgn[cidx, lidx]
                if old_val:
                    old_lit_sat = (s2 == 1)
                    new_lit_sat = (s2 == -1)
                else:
                    old_lit_sat = (s2 == -1)
                    new_lit_sat = (s2 == 1)

                sc = sat_count[cidx]
                gain = new_lit_sat & (~old_lit_sat)
                loss = old_lit_sat & (~new_lit_sat)
                sc_new = sc + gain.astype(np.int32) - loss.astype(np.int32)

                unsat_in_cidx_new = int((sc_new == 0).sum())
                unsat_in_cidx_old = int((sc == 0).sum())
                cost = n_unsat - unsat_in_cidx_old + unsat_in_cidx_new

            if cost < best_cost:
                best_cost = cost
                best_var = v
            elif cost == best_cost and rng.random() < 0.5:
                best_var = v

        if best_var >= 0:
            old_val = assignment[best_var]
            assignment[best_var] = not assignment[best_var]
            cidx = var_cidx_arr[best_var]
            lidx = var_lidx_arr[best_var]
            if len(cidx) > 0:
                s2 = clause_sgn[cidx, lidx]
                if old_val:
                    gained = (s2 == -1)
                    lost = (s2 == 1)
                else:
                    gained = (s2 == 1)
                    lost = (s2 == -1)
                sat_count[cidx] += gained.astype(np.int32) - lost.astype(np.int32)

    return assignment.tolist()


def _soft_min_polish(assignment, clause_vars, clause_sgn, n, rng, max_iters, deadline):
    """
    Smooth surrogate local search: minimize soft-min of clause satisfaction scores.
    Each clause gets a score in {0,1,2,3}. We use a log-sum-exp based soft-min
    to create a differentiable-ish objective, then do coordinate descent.
    """
    assignment = np.array(assignment, dtype=bool)
    m = clause_vars.shape[0]
    sat_count = _compute_sat_state(clause_vars, clause_sgn, assignment).astype(np.float64)

    for it in range(max_iters):
        if time.time() >= deadline:
            break

        n_unsat = int((sat_count == 0).sum())
        if n_unsat == 0:
            break

        # Soft-min surrogate: -log(sum(exp(-alpha * sat_count)))
        # Lower sat_count = higher contribution. We minimize this.
        alpha = 2.0
        # For numerical stability
        min_sc = sat_count.min()
        exp_terms = np.exp(-alpha * (sat_count - min_sc))
        soft_min_val = -np.log(exp_terms.sum()) + alpha * min_sc

        # Pick a random unsatisfied clause to focus on
        unsat_indices = np.where(sat_count == 0)[0]
        if len(unsat_indices) == 0:
            break
        ci = int(rng.choice(unsat_indices.tolist()))

        # Try flipping each variable in the clause; pick the one that best
        # improves the soft-min objective (approximated by gain/loss on sat_count)
        best_v = -1
        best_gain = -1
        for li in range(3):
            v = int(clause_vars[ci, li])
            old_val = assignment[v]

            # Compute delta in sat_count for all clauses containing v
            # We need var_cidx_arr/var_lidx_arr - pass them or recompute
            # For simplicity, do a local scan
            delta = np.zeros(m, dtype=np.int32)
            for j in range(3):
                for i2 in range(m):
                    if int(clause_vars[i2, j]) == v:
                        s2 = int(clause_sgn[i2, j])
                        if old_val:
                            # flipping True->False: pos lit loses sat, neg gains
                            if s2 == 1:
                                delta[i2] -= 1
                            else:
                                delta[i2] += 1
                        else:
                            if s2 == 1:
                                delta[i2] += 1
                            else:
                                delta[i2] -= 1
                        break

            new_sc = sat_count + delta.astype(np.float64)
            # Compute new soft-min
            min_sc2 = new_sc.min()
            exp_terms2 = np.exp(-alpha * (new_sc - min_sc2))
            soft_min_val2 = -np.log(exp_terms2.sum()) + alpha * min_sc2

            gain = soft_min_val - soft_min_val2  # positive = improvement
            if gain > best_gain:
                best_gain = gain
                best_v = v

        if best_v >= 0 and best_gain > 0:
            old_val = assignment[best_v]
            assignment[best_v] = not assignment[best_v]
            # Update sat_count
            for j in range(3):
                for i2 in range(m):
                    if int(clause_vars[i2, j]) == best_v:
                        s2 = int(clause_sgn[i2, j])
                        if old_val:
                            if s2 == 1:
                                sat_count[i2] -= 1
                            else:
                                sat_count[i2] += 1
                        else:
                            if s2 == 1:
                                sat_count[i2] += 1
                            else:
                                sat_count[i2] -= 1
                        break

    return assignment.tolist()


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 275  # Leave 55s margin before 330s kill

    m = len(clauses)

    # Precompute clause structure
    clause_vars, clause_sgn, var_cidx_arr, var_lidx_arr = _build_clause_data(n, clauses)

    best_assignment = None
    best_unsat_count = float('inf')

    def _try_update(assignment):
        nonlocal best_assignment, best_unsat_count
        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        return unsat_count == 0

    # Strategy 1: Phase-seeded vectorized impact WalkSAT with multiple restarts
    for attempt in range(200):
        if time.time() >= deadline:
            break
        seed = attempt * 7919 + 12345
        rng = random.Random(seed)

        if attempt == 0:
            assignment = _phase_seed(n, clauses, rng)
        elif attempt == 1:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        elif attempt == 2:
            # Inverted phase seed
            assignment = _phase_seed(n, clauses, rng)
            assignment = [not a for a in assignment]
        else:
            if best_assignment is not None:
                assignment = best_assignment[:]
                # Mutate a fraction proportional to how stuck we are
                frac = min(0.5, 0.1 + 0.02 * attempt)
                n_mut = int(n * frac)
                for _ in range(n_mut):
                    idx = rng.randrange(n)
                    assignment[idx] = not assignment[idx]
            else:
                assignment = [rng.random() < 0.5 for _ in range(n)]

        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]

        # Vectorized impact WalkSAT with adaptive steps
        steps = 2000 + attempt * 300
        assignment = _vectorized_impact_walksat(
            assignment, clause_vars, clause_sgn, var_cidx_arr, var_lidx_arr, n, m, rng, steps, deadline
        )

        if _try_update(assignment):
            return assignment

    # Strategy 2: Vectorized MinConflicts from best assignment found so far
    if best_assignment is not None and time.time() < deadline - 10:
        rng = random.Random(99999)
        assignment = best_assignment[:]
        assignment = _vectorized_minconflicts(
            assignment, clauses, n, clause_vars, clause_sgn, var_cidx_arr, var_lidx_arr, rng, 12000, deadline
        )
        if _try_update(assignment):
            return assignment

    # Strategy 3: More random-seed impact WalkSAT restarts
    for attempt in range(60):
        if time.time() >= deadline:
            break
        seed = 100000 + attempt * 31337
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]

        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]

        assignment = _vectorized_impact_walksat(
            assignment, clause_vars, clause_sgn, var_cidx_arr, var_lidx_arr, n, m, rng, 3000, deadline
        )

        if _try_update(assignment):
            return assignment

    # Strategy 4: Soft-min polish on best assignment (smooth surrogate local search)
    if best_assignment is not None and time.time() < deadline - 5:
        rng = random.Random(777777)
        assignment = best_assignment[:]
        assignment = _soft_min_polish(assignment, clause_vars, clause_sgn, n, rng, 5000, deadline)
        if _try_update(assignment):
            return assignment

    # Final: return best assignment found
    if best_assignment is not None:
        return best_assignment

    # Fallback: all False
    return [False] * n
# EVOLVE-BLOCK-END
