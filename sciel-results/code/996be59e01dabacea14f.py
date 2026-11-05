"""Rewritten 3-SAT solver with deadline guard and stronger local search."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    deadline = time.time() + 290.0  # safety margin below 330s kill

    num_clauses = len(clauses)

    # Precompute variable-to-clause index for faster min-confl
    # var_clauses[var] = list of (clause_idx, sign) where sign is +1 if literal is positive, -1 if negative
    var_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses):
        for l in c:
            var = abs(l) - 1
            sign = 1 if l > 0 else -1
            var_clauses[var].append((ci, sign))

    def make_random_assignment(seed):
        rng = random.Random(seed)
        return [rng.random() < 0.5 for _ in range(n)]

    def count_unsat(assignment):
        cnt = 0
        for c in clauses:
            sat = False
            for l in c:
                if assignment[abs(l) - 1] == (l > 0):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

    def unsat_clauses(assignment):
        result = []
        for i, c in enumerate(clauses):
            sat = False
            for l in c:
                if assignment[abs(l) - 1] == (l > 0):
                    sat = True
                    break
            if not sat:
                result.append(i)
        return result

    best_assignment = None
    best_unsat = num_clauses + 1

    # Try multiple random restarts with WalkSAT + min-confl
    for attempt in range(500):
        if time.time() > deadline:
            break
        assignment = make_random_assignment(attempt * 7919 + 13)
        cur_unsat = count_unsat(assignment)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        if cur_unsat == 0:
            return assignment

        # Local search: WalkSAT with min-confl
        rng = random.Random(attempt * 31337 + 42)
        steps = 0
        max_steps = 8000
        while cur_unsat > 0 and steps < max_steps:
            if time.time() > deadline:
                break
            steps += 1
            ucl = unsat_clauses(assignment)
            if not ucl:
                cur_unsat = 0
                break
            ci = rng.choice(ucl)
            clause = clauses[ci]

            # Min-confl: pick the literal in the clause that appears in the fewest other unsat clauses
            # For efficiency, count how many unsat clauses contain each variable
            # Use a simpler heuristic: for each literal in the clause, count how many unsat clauses it's in
            # But that's expensive. Instead, use the standard WalkSAT min-confl:
            # For each literal l in clause ci, count the number of unsatisfied clauses (other than ci)
            # that contain the variable of l. Pick the one with the minimum count.
            
            # Build a quick count of unsat clauses per variable
            # This is O(num_unsat * 3) which is acceptable
            unsat_vars_count = {}
            for uci in ucl:
                if uci == ci:
                    continue
                uc = clauses[uci]
                seen = set()
                for l in uc:
                    v = abs(l) - 1
                    if v not in seen:
                        seen.add(v)
                        unsat_vars_count[v] = unsat_vars_count.get(v, 0) + 1
            
            best_lit = None
            best_score = float('inf')
            for l in clause:
                var = abs(l) - 1
                score = unsat_vars_count.get(var, 0)
                if score < best_score:
                    best_score = score
                    best_lit = l
            
            lit = best_lit if best_lit is not None else rng.choice(clause)
            assignment[abs(lit) - 1] = not assignment[abs(lit) - 1]

            # Recompute unsat count
            new_unsat = count_unsat(assignment)
            if new_unsat < cur_unsat:
                cur_unsat = new_unsat
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment = assignment[:]
                if cur_unsat == 0:
                    return assignment
            elif new_unsat > cur_unsat:
                # With some probability, accept worse moves
                if rng.random() > 0.3:
                    # Revert
                    assignment[abs(lit) - 1] = not assignment[abs(lit) - 1]
                else:
                    cur_unsat = new_unsat
            else:
                cur_unsat = new_unsat  # no change

    return best_assignment if best_assignment else make_random_assignment(0)
# EVOLVE-BLOCK-END
