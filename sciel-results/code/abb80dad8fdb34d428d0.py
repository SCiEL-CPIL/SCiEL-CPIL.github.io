"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 320  # safety margin before 330s kill

    # Precompute variable occurrences: for each variable, list of (clause_idx, sign)
    # sign=1 means positive literal, sign=-1 means negative
    var_pos = [[] for _ in range(n)]  # clause indices where var appears positively
    var_neg = [[] for _ in range(n)]  # clause indices where var appears negatively
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            if lit > 0:
                var_pos[v].append(ci)
            else:
                var_neg[v].append(ci)

    def evaluate(assignment):
        """Return list of unsatisfied clause indices."""
        unsat = []
        for ci, clause in enumerate(clauses):
            sat = False
            for lit in clause:
                v = abs(lit) - 1
                if assignment[v] == (lit > 0):
                    sat = True
                    break
            if not sat:
                unsat.append(ci)
        return unsat

    def count_satisfied(assignment):
        """Count satisfied clauses."""
        count = 0
        for ci, clause in enumerate(clauses):
            sat = False
            for lit in clause:
                v = abs(lit) - 1
                if assignment[v] == (lit > 0):
                    sat = True
                    break
            if sat:
                count += 1
        return count

    def walksat(assignment, rng, max_steps, weight):
        """WalkSAT with adaptive weight. Returns solved assignment or None."""
        unsat = evaluate(assignment)
        if not unsat:
            return assignment

        for step in range(max_steps):
            if time.time() > deadline:
                return None
            clause = rng.choice(unsat)
            # Pick literal: with prob weight, pick from clause; else random var
            if rng.random() < weight:
                literal = rng.choice(clause)
            else:
                literal = rng.choice(clause)
            v = abs(literal) - 1
            assignment[v] = not assignment[v]
            # Recompute unsat - for 50 vars and moderate m, this is fine
            unsat = evaluate(assignment)
            if not unsat:
                return assignment
        return None

    def greedy(assignment, rng, max_steps):
        """Greedy: flip variable that maximizes newly satisfied clauses."""
        unsat = evaluate(assignment)
        if not unsat:
            return assignment

        for step in range(max_steps):
            if time.time() > deadline:
                return None
            # Find best flip
            best_gain = 0
            best_var = -1
            best_lit_sign = 1
            # For each variable, compute gain if we set it to True or False
            for v in range(n):
                # Current state
                cur = assignment[v]
                # Try flipping
                new = not cur
                # Count how many currently unsat clauses become sat
                gain = 0
                for ci in unsat:
                    clause = clauses[ci]
                    # Check if setting v=new satisfies this clause
                    for lit in clause:
                        if abs(lit) - 1 == v:
                            if (lit > 0) == new:
                                gain += 1
                            break
                    else:
                        # v not in this clause
                        pass
                if gain > best_gain:
                    best_gain = gain
                    best_var = v
                    best_lit_sign = 1 if new else -1
                elif gain == best_gain and gain > 0:
                    if rng.random() < 0.5:
                        best_gain = gain
                        best_var = v
                        best_lit_sign = 1 if new else -1

            if best_var < 0 or best_gain == 0:
                # No improving move; do a random flip
                v = rng.randrange(n)
                assignment[v] = not assignment[v]
            else:
                assignment[best_var] = not assignment[best_var]

            unsat = evaluate(assignment)
            if not unsat:
                return assignment
        return None

    # Try multiple strategies with restarts
    best_result = None
    best_score = -1

    for attempt in range(20):
        if time.time() > deadline:
            break
        rng = random.Random(attempt * 7919 + 13)
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Check if already satisfied
        unsat = evaluate(assignment)
        if not unsat:
            return assignment

        # Try WalkSAT with various parameters
        for weight in [0.5, 0.8, 0.95]:
            if time.time() > deadline:
                break
            result = walksat(assignment.copy(), rng, 5000, weight)
            if result is not None:
                return result

        # Try greedy
        if time.time() > deadline:
            break
        result = greedy(assignment.copy(), rng, 2000)
        if result is not None:
            return result

        # Track best partial solution
        score = count_satisfied(assignment)
        if score > best_score:
            best_score = score
            best_result = assignment.copy()

    # Return best found or last attempt
    if best_result is not None:
        return best_result

    # Final fallback: random assignment
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
