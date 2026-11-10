"""Stronger planted 3-SAT solver: fast-indexed MinCon + adaptive-noise WalkSAT + min-conflicts refinement with restarts."""
# EVOLVE-BLOCK-START
import time
import random


def _build_indices(n, clauses):
    """Precompute clause satisfaction helpers and variable-to-clause indices."""
    m = len(clauses)
    # For each clause, store the three variable indices and their polarities
    clause_vars = []   # list of (v0, v1, v2)
    clause_pols = []   # list of (p0, p1, p2) where p=1 means positive, 0 means negative
    for c in clauses:
        vs = tuple(abs(l) - 1 for l in c)
        ps = tuple(1 if l > 0 else 0 for l in c)
        clause_vars.append(vs)
        clause_pols.append(ps)

    # var_to_clauses: for each variable, list of clause indices it appears in
    var_to_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for v in clause_vars[ci]:
            var_to_clauses[v].append(ci)

    return clause_vars, clause_pols, var_to_clauses


def _clause_sat_fast(ci, assign, clause_vars, clause_pols):
    vs = clause_vars[ci]
    ps = clause_pols[ci]
    return (assign[vs[0]] == ps[0]) or (assign[vs[1]] == ps[1]) or (assign[vs[2]] == ps[2])


def _count_unsat_fast(assign, clause_vars, clause_pols):
    cnt = 0
    for ci in range(len(clause_vars)):
        vs = clause_vars[ci]
        ps = clause_pols[ci]
        if not ((assign[vs[0]] == ps[0]) or (assign[vs[1]] == ps[1]) or (assign[vs[2]] == ps[2])):
            cnt += 1
    return cnt


def _phase_seed(n, clauses, rng):
    """Estimate a planted assignment from literal phase bias, with random tie-breaking."""
    pos = [0] * n
    neg = [0] * n
    for c in clauses:
        for l in c:
            if l > 0:
                pos[l - 1] += 1
            else:
                neg[-l - 1] += 1
    assign = []
    for i in range(n):
        p, q = pos[i], neg[i]
        if p > q:
            assign.append(True)
        elif q > p:
            assign.append(False)
        else:
            assign.append(rng.random() < 0.5)
    return assign


def _walksat_adaptive(clauses, assign, rng, deadline, clause_vars, clause_pols, var_to_clauses, max_steps):
    """WalkSAT with adaptive noise: greedy early, more random later."""
    m = len(clauses)
    n = len(assign)
    for step in range(max_steps):
        if time.time() >= deadline:
            break
        # Find unsat clauses
        unsat = []
        for ci in range(m):
            vs = clause_vars[ci]
            ps = clause_pols[ci]
            if not ((assign[vs[0]] == ps[0]) or (assign[vs[1]] == ps[1]) or (assign[vs[2]] == ps[2])):
                unsat.append(ci)
        if not unsat:
            return assign, True

        # Pick a random unsat clause
        ci = rng.choice(unsat)
        vs = clause_vars[ci]
        ps = clause_pols[ci]

        # Adaptive noise: starts low (greedy), increases over time
        progress = step / max_steps
        noise = 0.05 + 0.35 * progress

        if rng.random() < noise:
            # Random flip
            v = rng.choice(vs)
        else:
            # Score each variable by how many unsat clauses it appears in
            best_v = vs[0]
            best_s = -1
            for v in vs:
                s = 0
                for uc in var_to_clauses[v]:
                    uvs = clause_vars[uc]
                    ups = clause_pols[uc]
                    if not ((assign[uvs[0]] == ups[0]) or (assign[uvs[1]] == ups[1]) or (assign[uvs[2]] == ups[2])):
                        s += 1
                # Small random tie-break
                s += rng.random() * 0.01
                if s > best_s:
                    best_s = s
                    best_v = v
        assign[best_v] = not assign[best_v]

    return assign, _count_unsat_fast(assign, clause_vars, clause_pols) == 0


def _mincon_fast(clauses, assign, rng, deadline, clause_vars, clause_pols, var_to_clauses, max_steps):
    """MinCon: flip variable in an unsat clause that appears in most unsat clauses."""
    m = len(clauses)
    for step in range(max_steps):
        if time.time() >= deadline:
            break
        # Find unsat clauses and count variable appearances
        unsat = []
        unsat_var_count = [0] * len(assign)
        for ci in range(m):
            vs = clause_vars[ci]
            ps = clause_pols[ci]
            if not ((assign[vs[0]] == ps[0]) or (assign[vs[1]] == ps[1]) or (assign[vs[2]] == ps[2])):
                unsat.append(ci)
                for v in vs:
                    unsat_var_count[v] += 1
        if not unsat:
            return assign, True

        # Pick the unsat clause whose variables have highest total unsat count
        best_ci = None
        best_score = -1
        for ci in unsat:
            vs = clause_vars[ci]
            s = unsat_var_count[vs[0]] + unsat_var_count[vs[1]] + unsat_var_count[vs[2]]
            if s > best_score:
                best_score = s
                best_ci = ci

        vs = clause_vars[best_ci]
        # Flip the variable with highest unsat count in this clause
        best_v = vs[0]
        best_c = -1
        for v in vs:
            if unsat_var_count[v] > best_c:
                best_c = unsat_var_count[v]
                best_v = v
        assign[best_v] = not assign[best_v]

    return assign, _count_unsat_fast(assign, clause_vars, clause_pols) == 0


def _min_conflicts_fast(clauses, assign, rng, deadline, clause_vars, clause_pols, var_to_clauses, max_steps):
    """Min-conflicts: flip the variable that reduces unsat count the most."""
    m = len(clauses)
    n = len(assign)
    for step in range(max_steps):
        if time.time() >= deadline:
            break
        # Find unsat clauses
        unsat = []
        for ci in range(m):
            vs = clause_vars[ci]
            ps = clause_pols[ci]
            if not ((assign[vs[0]] == ps[0]) or (assign[vs[1]] == ps[1]) or (assign[vs[2]] == ps[2])):
                unsat.append(ci)
        if not unsat:
            return assign, True

        # Collect candidate variables (those in unsat clauses)
        cand_vars = set()
        for ci in unsat:
            vs = clause_vars[ci]
            cand_vars.add(vs[0])
            cand_vars.add(vs[1])
            cand_vars.add(vs[2])

        current_unsat = len(unsat)
        best_var = None
        best_delta = 0  # we want most negative delta (biggest reduction)

        for var in cand_vars:
            # Simulate flip
            assign[var] = not assign[var]
            new_unsat = 0
            for ci in range(m):
                vs = clause_vars[ci]
                ps = clause_pols[ci]
                if not ((assign[vs[0]] == ps[0]) or (assign[vs[1]] == ps[1]) or (assign[vs[2]] == ps[2])):
                    new_unsat += 1
            assign[var] = not assign[var]  # revert
            delta = new_unsat - current_unsat
            if delta < best_delta or best_var is None:
                best_delta = delta
                best_var = var
            if new_unsat == 0:
                # This flip solves it
                assign[var] = not assign[var]
                return assign, True

        if best_var is not None:
            assign[best_var] = not assign[best_var]

    return assign, _count_unsat_fast(assign, clause_vars, clause_pols) == 0


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start = time.time()
    # Safety margin: stop 8s before the 330s kill
    deadline = start + 322.0

    # Build fast indices
    clause_vars, clause_pols, var_to_clauses = _build_indices(n, clauses)

    # Precompute a phase-bias seed
    base_rng = random.Random(0)
    phase_assign = _phase_seed(n, clauses, base_rng)

    best_assign = phase_assign[:]
    best_unsat = _count_unsat_fast(best_assign, clause_vars, clause_pols)
    if best_unsat == 0:
        return best_assign

    # Portfolio: alternate strategies and seeds, growing effort, until deadline
    seed_counter = 0
    while time.time() < deadline:
        rng = random.Random(seed_counter)
        # Choose a starting assignment
        r = rng.random()
        if r < 0.4:
            assign = phase_assign[:]
        elif r < 0.6:
            assign = [rng.random() < 0.5 for _ in range(n)]
        elif r < 0.8:
            # Flip a few random vars from phase seed
            assign = phase_assign[:]
            for _ in range(rng.randint(1, 8)):
                assign[rng.randrange(n)] = not assign[rng.randrange(n)]
        else:
            # Start from best found, mutate
            assign = best_assign[:]
            for _ in range(rng.randint(2, 10)):
                assign[rng.randrange(n)] = not assign[rng.randrange(n)]

        # Choose strategy
        strategy = rng.choice(["walk", "mincon", "walk", "mincon"])  # walk slightly preferred
        remaining = deadline - time.time()
        # Scale steps based on remaining time; fast operations allow many steps
        max_steps = min(500000, int(remaining * 80000))  # ~80k steps/sec estimate, cap at 500k

        if strategy == "mincon":
            assign, solved = _mincon_fast(clauses, assign, rng, deadline, clause_vars, clause_pols, var_to_clauses, max_steps)
        else:
            assign, solved = _walksat_adaptive(clauses, assign, rng, deadline, clause_vars, clause_pols, var_to_clauses, max_steps)

        if solved:
            return assign

        u = _count_unsat_fast(assign, clause_vars, clause_pols)
        if u < best_unsat:
            best_unsat = u
            best_assign = assign[:]

        # Min-conflicts refinement on the best assignment when we have time
        if best_unsat > 0 and time.time() < deadline - 5.0:
            mc_assign = best_assign[:]
            mc_assign, mc_solved = _min_conflicts_fast(
                clauses, mc_assign, rng, deadline,
                clause_vars, clause_pols, var_to_clauses, 20000
            )
            if mc_solved:
                return mc_assign
            mc_u = _count_unsat_fast(mc_assign, clause_vars, clause_pols)
            if mc_u < best_unsat:
                best_unsat = mc_u
                best_assign = mc_assign[:]

        seed_counter += 1

    return best_assign
# EVOLVE-BLOCK-END
