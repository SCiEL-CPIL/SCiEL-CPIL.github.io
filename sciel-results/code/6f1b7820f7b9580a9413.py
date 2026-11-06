# EVOLVE-BLOCK-START
import random


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(12345)

    # Precompute clause literal structures and per-variable clause incidence.
    # Represent each clause as (pos_vars, neg_vars) where pos_vars = set of var idx set True,
    # neg_vars = set of var idx set False. Variable indices are 0-based.
    clause_pos = []
    clause_neg = []
    var_clauses = [[] for _ in range(n)]  # var_clauses[v] = list of clause indices containing v (any polarity)
    var_polarity = {}  # (clause_idx, var) -> True if pos, False if neg

    for ci, clause in enumerate(clauses):
        pos = set()
        neg = set()
        for lit in clause:
            v = abs(lit) - 1
            if lit > 0:
                pos.add(v)
                var_polarity[(ci, v)] = True
            else:
                neg.add(v)
                var_polarity[(ci, v)] = False
            var_clauses[v].append(ci)
        clause_pos.append(pos)
        clause_neg.append(neg)

    m = len(clauses)

    def make_initial_assignment(bias=None):
        # bias: optional list of booleans to copy; otherwise random
        if bias is None:
            return [rng.random() < 0.5 for _ in range(n)]
        return list(bias)

    def evaluate(assignment):
        # Build state from assignment.
        true_vars = set()
        false_vars = set()
        for v in range(n):
            if assignment[v]:
                true_vars.add(v)
            else:
                false_vars.add(v)
        clause_sat = [False] * m
        unsat_clauses = set()
        unsat_count = [0] * n  # number of unsat clauses containing var v
        for ci in range(m):
            sat = False
            for v in clause_pos[ci]:
                if v in true_vars:
                    sat = True
                    break
            if not sat:
                for v in clause_neg[ci]:
                    if v in false_vars:
                        sat = True
                        break
            clause_sat[ci] = sat
            if not sat:
                unsat_clauses.add(ci)
                for v in clause_pos[ci]:
                    unsat_count[v] += 1
                for v in clause_neg[ci]:
                    unsat_count[v] += 1
        return true_vars, false_vars, clause_sat, unsat_clauses, unsat_count

    def is_satisfied_clause(ci, true_vars, false_vars):
        for v in clause_pos[ci]:
            if v in true_vars:
                return True
        for v in clause_neg[ci]:
            if v in false_vars:
                return True
        return False

    def flip_variable(v, true_vars, false_vars, clause_sat, unsat_clauses, unsat_count, assignment):
        # Remove v from current set, add to other.
        if v in true_vars:
            true_vars.remove(v)
            false_vars.add(v)
            assignment[v] = False
        else:
            false_vars.remove(v)
            true_vars.add(v)
            assignment[v] = True

        # Update each clause containing v.
        for ci in var_clauses[v]:
            # Determine new sat status of clause ci.
            new_sat = is_satisfied_clause(ci, true_vars, false_vars)
            old_sat = clause_sat[ci]
            if old_sat == new_sat:
                continue
            if new_sat:
                # clause becomes satisfied: decrement unsat_count for all its literals
                unsat_clauses.discard(ci)
                for v2 in clause_pos[ci]:
                    unsat_count[v2] -= 1
                for v2 in clause_neg[ci]:
                    unsat_count[v2] -= 1
            else:
                # clause becomes unsatisfied: increment unsat_count for all its literals
                unsat_clauses.add(ci)
                for v2 in clause_pos[ci]:
                    unsat_count[v2] += 1
                for v2 in clause_neg[ci]:
                    unsat_count[v2] += 1
            clause_sat[ci] = new_sat

    def pick_flip(unsat_clauses, unsat_count):
        # Pick an unsat clause, then a literal to flip with prob ∝ 1/(unsat_count+1).
        ci = rng.choice(list(unsat_clauses))
        # Collect candidate literals (var, polarity) in this clause.
        cands = []
        weights = []
        for v in clause_pos[ci]:
            cands.append(v)
            weights.append(1.0 / (unsat_count[v] + 1.0))
        for v in clause_neg[ci]:
            cands.append(v)
            weights.append(1.0 / (unsat_count[v] + 1.0))
        # With small probability, pick uniformly (noise) to escape flat regions.
        if rng.random() < 0.15:
            return rng.choice(cands)
        # Weighted selection.
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        for v, w in zip(cands, weights):
            acc += w
            if r <= acc:
                return v
        return cands[-1]

    # Main search with restarts.
    total_budget = 60000
    max_restarts = 20
    best_assignment = None
    best_unsat = None

    for restart in range(max_restarts):
        # On first restart use random; on later restarts bias toward previous best.
        if best_assignment is None:
            assignment = make_initial_assignment()
        else:
            # Bias: start from best, flip a few random vars to perturb.
            assignment = list(best_assignment)
            k = rng.randint(1, max(1, n // 10))
            for _ in range(k):
                v = rng.randrange(n)
                assignment[v] = not assignment[v]

        true_vars, false_vars, clause_sat, unsat_clauses, unsat_count = evaluate(assignment)

        if not unsat_clauses:
            return assignment

        if best_unsat is None or len(unsat_clauses) < best_unsat:
            best_unsat = len(unsat_clauses)
            best_assignment = list(assignment)

        # Flip budget per restart, decreasing slightly.
        flip_budget = max(2000, total_budget // max(1, (restart + 1)))
        no_progress = 0
        last_unsat = len(unsat_clauses)

        for _ in range(flip_budget):
            if not unsat_clauses:
                return assignment

            cur_unsat = len(unsat_clauses)
            if cur_unsat < last_unsat:
                last_unsat = cur_unsat
                no_progress = 0
                if best_unsat is None or cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment = list(assignment)
            else:
                no_progress += 1

            if no_progress > 3000:
                break  # restart

            v = pick_flip(unsat_clauses, unsat_count)
            flip_variable(v, true_vars, false_vars, clause_sat, unsat_clauses, unsat_count, assignment)

        if not unsat_clauses:
            return assignment

    # Return best found.
    return best_assignment if best_assignment is not None else [False] * n
# EVOLVE-BLOCK-END
