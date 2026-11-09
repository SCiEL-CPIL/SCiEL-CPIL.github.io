"""Stronger planted 3-SAT solver: MinCon + phase-aware WalkSAT with restarts and a wall-clock deadline guard."""
# EVOLVE-BLOCK-START
import time
import random


def _clause_sat(clause, assign):
    for l in clause:
        if assign[abs(l) - 1] == (l > 0):
            return True
    return False


def _count_unsat(clauses, assign):
    return sum(0 if _clause_sat(c, assign) else 1 for c in clauses)


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


def _mincon_step(clauses, assign, rng):
    """One MinCon step: pick unsat clause with fewest variables, flip the var with highest MinCon score."""
    # Precompute per-variable count of unsat clauses it appears in (positive or negative)
    unsat_vars = [0] * len(assign)
    unsat_clauses = []
    for c in clauses:
        sat = False
        for l in c:
            if assign[abs(l) - 1] == (l > 0):
                sat = True
                break
        if not sat:
            unsat_clauses.append(c)
            for l in c:
                unsat_vars[abs(l) - 1] += 1
    if not unsat_clauses:
        return assign, True
    # Choose the unsat clause with the fewest variables (all 3 here, so pick one with most "useful" vars)
    # Prefer clause containing a variable with highest unsat count
    best_clause = None
    best_score = -1
    for c in unsat_clauses:
        s = sum(unsat_vars[abs(l) - 1] for l in c)
        if s > best_score:
            best_score = s
            best_clause = c
    # Flip the variable in best_clause with the highest unsat count (MinCon)
    best_var = max(best_clause, key=lambda l: unsat_vars[abs(l) - 1])
    assign[abs(best_var) - 1] = not assign[abs(best_var) - 1]
    return assign, False


def _run_one(clauses, assign, rng, deadline, strategy):
    """Run a single local search to completion or deadline. Returns (assign, solved)."""
    n = len(assign)
    if strategy == "mincon":
        for _ in range(200000):
            if time.time() >= deadline:
                return assign, _count_unsat(clauses, assign) == 0
            assign, done = _mincon_step(clauses, assign, rng)
            if done:
                return assign, True
    else:
        # Phase-aware WalkSAT: pick unsat clause, flip var with best "break/fix" heuristic
        for _ in range(200000):
            if time.time() >= deadline:
                return assign, _count_unsat(clauses, assign) == 0
            unsat = [c for c in clauses if not _clause_sat(c, assign)]
            if not unsat:
                return assign, True
            c = rng.choice(unsat)
            # Score each literal: prefer flipping a var that is in many unsat clauses
            # and that is currently not satisfying this clause
            best_l = None
            best_s = -1
            for l in c:
                v = abs(l) - 1
                # break: number of clauses that this var currently satisfies (would break)
                # fix: number of unsat clauses this var appears in
                s = rng.random() * 0.1
                # Heuristic: prefer flipping to satisfy this clause (it always does for a var in an unsat clause)
                # and prefer vars in more unsat clauses
                s += sum(1 for cc in unsat if v in [abs(x) - 1 for x in cc]) * 0.01
                if s > best_s:
                    best_s = s
                    best_l = l
            assign[abs(best_l) - 1] = not assign[abs(best_l) - 1]
    return assign, _count_unsat(clauses, assign) == 0


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start = time.time()
    # Safety margin: stop 8s before the 330s kill
    deadline = start + 322.0

    # Precompute a phase-bias seed (deterministic base) and use it
    base_rng = random.Random(0)
    phase_assign = _phase_seed(n, clauses, base_rng)

    best_assign = phase_assign[:]
    best_unsat = _count_unsat(clauses, best_assign)
    if best_unsat == 0:
        return best_assign

    # Portfolio: alternate strategies and seeds, growing effort, until deadline
    seed_counter = 0
    while time.time() < deadline:
        rng = random.Random(seed_counter)
        # Choose a starting assignment
        r = rng.random()
        if r < 0.5:
            assign = phase_assign[:]
        elif r < 0.75:
            assign = [rng.random() < 0.5 for _ in range(n)]
        else:
            # Flip a few random vars from phase seed
            assign = phase_assign[:]
            for _ in range(rng.randint(1, 5)):
                assign[rng.randrange(n)] = not assign[rng.randrange(n)]

        strategy = rng.choice(["mincon", "walk"])
        assign, solved = _run_one(clauses, assign, rng, deadline, strategy)
        if solved:
            return assign
        u = _count_unsat(clauses, assign)
        if u < best_unsat:
            best_unsat = u
            best_assign = assign[:]
        seed_counter += 1

    return best_assign
# EVOLVE-BLOCK-END
