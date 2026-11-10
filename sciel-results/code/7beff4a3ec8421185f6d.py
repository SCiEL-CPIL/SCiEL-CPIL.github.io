"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np
import math


def _build_clause_data(n, clauses):
    """Precompute clause structure as numpy arrays for vectorized operations."""
    m = len(clauses)
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_sgn = np.zeros((m, 3), dtype=np.int8)
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            clause_vars[i, j] = abs(lit) - 1
            clause_sgn[i, j] = 1 if lit > 0 else -1
    # For each variable, precompute which (clause, position) pairs contain it
    var_to_clauses = {}
    for v in range(n):
        var_to_clauses[v] = []
    for i in range(m):
        for j in range(3):
            v = int(clause_vars[i, j])
            var_to_clauses[v].append((i, j))
    return clause_vars, clause_sgn, var_to_clauses


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


def _vectorized_impact_walksat(assignment, clause_vars, clause_sgn, var_to_clauses, n, m, rng, max_steps, deadline):
    """
    Vectorized WalkSAT with impact-based literal selection.
    """
    assignment = np.array(assignment, dtype=bool)
    sat_count = _compute_sat_state(clause_vars, clause_sgn, assignment)
    m_arr = clause_vars.shape[0]

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

        lits_vars = clause_vars[ci]
        lits_sgn = clause_sgn[ci]

        # Adaptive noise
        progress = step / max(1, max_steps)
        noise = 0.30 - 0.15 * progress

        if rng.random() < noise:
            li = rng.randint(0, 2)
            v = int(lits_vars[li])
        else:
            gains = np.zeros(3, dtype=np.int32)
            for li in range(3):
                v = int(lits_vars[li])
                old_val = assignment[v]
                new_val = not old_val

                vcl = var_to_clauses[v]
                if not vcl:
                    gains[li] = 0
                    continue

                cidx = np.array([p[0] for p in vcl], dtype=np.int32)
                lidx = np.array([p[1] for p in vcl], dtype=np.int32)
                s2 = clause_sgn[cidx, lidx]

                old_lit_sat = (s2 == 1) & old_val | (s2 == -1) & (~old_val)
                new_lit_sat = (s2 == 1) & new_val | (s2 == -1) & (~new_val)

                sc = sat_count[cidx]

                newly_sat = (sc == 0) & new_lit_sat
                newly_unsat = (sc == 1) & old_lit_sat & (~new_lit_sat)

                gains[li] = int(newly_sat.sum()) - int(newly_unsat.sum())

            max_gain = gains.max()
            best_candidates = np.where(gains == max_gain)[0]
            li = int(rng.choice(best_candidates.tolist()))
            v = int(lits_vars[li])

        # Apply the flip
        old_val = assignment[v]
        assignment[v] = not assignment[v]
        new_val = not old_val

        for (ci2, li2) in var_to_clauses[v]:
            ci2 = int(ci2)
            li2 = int(li2)
            s2 = int(clause_sgn[ci2, li2])
            if s2 == 1:
                old_lit_sat = old_val
                new_lit_sat = new_val
            else:
                old_lit_sat = not old_val
                new_lit_sat = not new_val
            if new_lit_sat and not old_lit_sat:
                sat_count[ci2] += 1
            elif old_lit_sat and not new_lit_sat:
                sat_count[ci2] -= 1

    return assignment.tolist()


def _vectorized_simulated_annealing(assignment, clause_vars, clause_sgn, var_to_clauses, n, m, rng, max_steps, deadline):
    """
    Vectorized Simulated Annealing.
    Uses Metropolis criterion to accept worse solutions to escape local minima.
    """
    assignment = np.array(assignment, dtype=bool)
    sat_count = _compute_sat_state(clause_vars, clause_sgn, assignment)
    
    # Initial temperature based on problem size
    T_init = 1.5
    T_final = 0.05
    # Cooling rate: geometric
    # We want T to go from T_init to T_final over max_steps
    # T_k = T_init * (T_final/T_init)^k
    # (T_final/T_init) = (T_final/T_init)^(1/max_steps)
    if max_steps <= 0:
        return assignment.tolist()
    cooling_factor = (T_final / T_init) ** (1.0 / max_steps)
    
    current_T = T_init
    current_unsat = int((sat_count == 0).sum())

    for step in range(max_steps):
        if time.time() >= deadline:
            break
            
        if current_unsat == 0:
            break

        # Pick a random unsatisfied clause
        unsat_indices = np.where(sat_count == 0)[0]
        if len(unsat_indices) == 0:
            break
        ci = int(rng.choice(unsat_indices.tolist()))

        # Pick a random variable from the clause
        li = rng.randint(0, 2)
        v = int(clause_vars[ci, li])
        
        old_val = assignment[v]
        new_val = not old_val

        # Calculate delta in unsat count
        # We need to check all clauses containing v
        vcl = var_to_clauses[v]
        delta_unsat = 0
        if vcl:
            cidx = np.array([p[0] for p in vcl], dtype=np.int32)
            lidx = np.array([p[1] for p in vcl], dtype=np.int32)
            s2 = clause_sgn[cidx, lidx]

            old_lit_sat = (s2 == 1) & old_val | (s2 == -1) & (~old_val)
            new_lit_sat = (s2 == 1) & new_val | (s2 == -1) & (~new_val)

            sc = sat_count[cidx]
            
            # Clauses becoming unsat: were sat-by-one (sc==1), old_lit_sat, not new_lit_sat
            newly_unsat = (sc == 1) & old_lit_sat & (~new_lit_sat)
            # Clauses becoming sat: were unsat (sc==0), new_lit_sat
            newly_sat = (sc == 0) & new_lit_sat
            
            delta_unsat = int(newly_unsat.sum()) - int(newly_sat.sum())

        # Metropolis Criterion
        if delta_unsat < 0:
            accept = True
        elif delta_unsat == 0:
            accept = True
        else:
            # Accept with probability exp(-delta / T)
            # If T is very small, this is ~0
            if current_T > 1e-9:
                prob = math.exp(-delta_unsat / current_T)
                accept = rng.random() < prob
            else:
                accept = False

        if accept:
            assignment[v] = new_val
            # Update sat_count
            for (ci2, li2) in vcl:
                ci2 = int(ci2)
                li2 = int(li2)
                s2 = int(clause_sgn[ci2, li2])
                if s2 == 1:
                    old_lit_sat = old_val
                    new_lit_sat = new_val
                else:
                    old_lit_sat = not old_val
                    new_lit_sat = not new_val
                if new_lit_sat and not old_lit_sat:
                    sat_count[ci2] += 1
                elif old_lit_sat and not new_lit_sat:
                    sat_count[ci2] -= 1
            current_unsat += delta_unsat

        # Cool down
        current_T *= cooling_factor

    return assignment.tolist()


def _vectorized_minconflicts(assignment, clauses, n, clause_vars, clause_sgn, var_to_clauses, rng, max_steps, deadline):
    """
    Vectorized MinConflicts.
    """
    assignment = np.array(assignment, dtype=bool)
    m = clause_vars.shape[0]
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
            new_val = not old_val

            vcl = var_to_clauses[v]
            if not vcl:
                cost = n_unsat
            else:
                cidx = np.array([p[0] for p in vcl], dtype=np.int32)
                lidx = np.array([p[1] for p in vcl], dtype=np.int32)
                s2 = clause_sgn[cidx, lidx]

                old_lit_sat = (s2 == 1) & old_val | (s2 == -1) & (~old_val)
                new_lit_sat = (s2 == 1) & new_val | (s2 == -1) & (~new_val)

                sc = sat_count[cidx].copy()
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
            new_val = not old_val
            for (ci2, li2) in var_to_clauses[best_var]:
                ci2 = int(ci2)
                li2 = int(li2)
                s2 = int(clause_sgn[ci2, li2])
                if s2 == 1:
                    old_lit_sat = old_val
                    new_lit_sat = new_val
                else:
                    old_lit_sat = not old_val
                    new_lit_sat = not new_val
                if new_lit_sat and not old_lit_sat:
                    sat_count[ci2] += 1
                elif old_lit_sat and not new_lit_sat:
                    sat_count[ci2] -= 1

    return assignment.tolist()


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280  # Leave 50s margin before 330s kill

    m = len(clauses)

    clause_vars, clause_sgn, var_to_clauses = _build_clause_data(n, clauses)

    best_assignment = None
    best_unsat_count = float('inf')

    def _try_update(assignment):
        nonlocal best_assignment, best_unsat_count
        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        return unsat_count == 0

    # Strategy 1: Phase-seeded vectorized impact WalkSAT
    for attempt in range(60):
        if time.time() >= deadline:
            break
        seed = attempt * 7919 + 12345
        rng = random.Random(seed)

        if attempt == 0:
            assignment = _phase_seed(n, clauses, rng)
        elif attempt == 1:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            if best_assignment is not None:
                assignment = best_assignment[:]
                for _ in range(n // 3):
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

        steps = 1000 + attempt * 100
        assignment = _vectorized_impact_walksat(
            assignment, clause_vars, clause_sgn, var_to_clauses, n, m, rng, steps, deadline
        )

        if _try_update(assignment):
            return assignment

    # Strategy 2: Vectorized Simulated Annealing (NEW STRUCTURAL CHANGE)
    # SA is good at escaping local minima that WalkSAT gets stuck in.
    for attempt in range(40):
        if time.time() >= deadline:
            break
        seed = 200000 + attempt * 4444
        rng = random.Random(seed)

        # Start SA from best found so far, or random if none
        if best_assignment is not None:
            # Perturb best assignment to provide diversity for SA
            assignment = best_assignment[:]
            # Flip ~20% of variables
            for _ in range(n // 5):
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

        # SA needs enough steps to cool down properly
        steps = 2000
        assignment = _vectorized_simulated_annealing(
            assignment, clause_vars, clause_sgn, var_to_clauses, n, m, rng, steps, deadline
        )

        if _try_update(assignment):
            return assignment

    # Strategy 3: Vectorized MinConflicts
    if best_assignment is not None and time.time() < deadline - 5:
        rng = random.Random(99999)
        assignment = best_assignment[:]
        assignment = _vectorized_minconflicts(
            assignment, clauses, n, clause_vars, clause_sgn, var_to_clauses, rng, 5000, deadline
        )
        if _try_update(assignment):
            return assignment

    # Final polish
    if best_assignment is not None and time.time() < deadline - 2:
        unsat_count = _count_unsat(best_assignment, clauses)
        if 0 < unsat_count <= 4:
            unsat_vars = set()
            for clause in clauses:
                sat = False
                for lit in clause:
                    if best_assignment[abs(lit) - 1] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    for lit in clause:
                        unsat_vars.add(abs(lit) - 1)
            unsat_vars = list(unsat_vars)
            max_flip = min(len(unsat_vars), 5)
            from itertools import combinations
            for r in range(1, max_flip + 1):
                if time.time() >= deadline - 1:
                    break
                for combo in combinations(unsat_vars, r):
                    candidate = best_assignment[:]
                    for v in combo:
                        candidate[v] = not candidate[v]
                    if _count_unsat(candidate, clauses) == 0:
                        return candidate
            for v in range(n):
                if time.time() >= deadline - 1:
                    break
                candidate = best_assignment[:]
                candidate[v] = not candidate[v]
                if _count_unsat(candidate, clauses) == 0:
                    return candidate

    if best_assignment is not None:
        return best_assignment

    return [False] * n
# EVOLVE-BLOCK-END
