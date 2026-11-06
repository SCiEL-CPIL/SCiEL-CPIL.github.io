"""WalkSAT for planted 3-SAT with incremental clause tracking, weighted flips,
restarts, and an explicit wall-clock deadline guard."""
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    deadline = time.monotonic() + 285.0  # safety margin vs 300s limit

    # Precompute per-clause variable lists and polarity bit per variable.
    clause_vars = []
    clause_polarity = []
    for c in clauses:
        vs = []
        ps = []
        for l in c:
            a = abs(l) - 1
            vs.append(a)
            ps.append(1 if l > 0 else 0)
        clause_vars.append(vs)
        clause_polarity.append(ps)

    # For each variable, list of (clause_index, polarity).
    var_clauses = [[] for _ in range(n)]
    for ci, (vs, ps) in enumerate(zip(clause_vars, clause_polarity)):
        for v, p in zip(vs, ps):
            var_clauses[v].append((ci, p))

    m = len(clauses)

    def make_assignment():
        return [rng.random() < 0.5 for _ in range(n)]

    def init_status(assignment):
        # num_sat[ci] = number of satisfied literals in clause ci
        num_sat = [0] * m
        for ci in range(m):
            s = 0
            for v, p in zip(clause_vars[ci], clause_polarity[ci]):
                if assignment[v] == bool(p):
                    s += 1
            num_sat[ci] = s
        return num_sat

    def num_unsat_of(num_sat):
        return sum(1 for s in num_sat if s == 0)

    # Best assignment seen (fewest unsat clauses).
    best_assignment = None
    best_unsat = None

    def record_if_better(assignment, unsat):
        nonlocal best_assignment, best_unsat
        if best_unsat is None or unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment[:]

    # Restart loop.
    restart_step_budget = 40000
    max_restarts = 400
    for restart in range(max_restarts):
        if time.monotonic() > deadline:
            break
        assignment = make_assignment()
        num_sat = init_status(assignment)
        unsat = num_unsat_of(num_sat)
        record_if_better(assignment, unsat)
        if unsat == 0:
            return assignment

        # Local search phase with weighted flips.
        steps = 0
        while steps < restart_step_budget:
            steps += 1
            if steps % 512 == 0 and time.monotonic() > deadline:
                record_if_better(assignment, unsat)
                break

            # Pick an unsatisfied clause.
            # Find one: scan from a random start for speed.
            start = rng.randrange(m)
            clause = None
            for k in range(m):
                ci = (start + k) % m
                if num_sat[ci] == 0:
                    clause = ci
                    break
            if clause is None:
                unsat = 0
                record_if_better(assignment, 0)
                return assignment

            vs = clause_vars[clause]
            ps = clause_polarity[clause]
            k = len(vs)

            # Choose flip variable.
            # 10% chance: min-ones repair (reduce satisfied literals in this clause).
            if rng.random() < 0.10:
                # pick among literals currently satisfied (keep clause unsat, lower ones)
                candidates = [i for i in range(k) if assignment[vs[i]] == bool(ps[i])]
                if candidates:
                    i = candidates[rng.randrange(len(candidates))]
                else:
                    i = rng.randrange(k)
            else:
                # prefer a literal that would satisfy the clause (min-ones bias)
                unsat_lits = [i for i in range(k) if assignment[vs[i]] != bool(ps[i])]
                if unsat_lits:
                    # weight: more unsat literals in clause -> pick among them
                    i = unsat_lits[rng.randrange(len(unsat_lits))]
                else:
                    i = rng.randrange(k)

            v = vs[i]
            p = ps[i]
            # Flip v; update num_sat for all clauses containing v.
            for (ci, cp) in var_clauses[v]:
                if assignment[v] == bool(cp):
                    # currently satisfied -> becomes unsat
                    num_sat[ci] -= 1
                    if num_sat[ci] == 0:
                        unsat += 1
                else:
                    # currently unsat -> becomes satisfied
                    num_sat[ci] += 1
                    if num_sat[ci] == 1:
                        unsat -= 1
            assignment[v] = not assignment[v]

            if unsat == 0:
                record_if_better(assignment, 0)
                return assignment

        record_if_better(assignment, unsat)

    # Deadline reached: return best seen (or last assignment).
    if best_assignment is None:
        best_assignment = assignment
    return best_assignment
