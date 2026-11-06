"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    # WalkSAT with a fixed flip budget, deadline guard, and clause-weighted flips.
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)

    # Pre-compute literal signs and variable indices for speed
    # clauses are lists of signed 1-based literals
    m = len(clauses)
    
    # Track which variable each literal maps to and its sign
    # We'll keep clauses as-is but precompute for fast access

    start_time = time.time()
    deadline = 290.0  # Safety margin before 300s limit

    # Random initial assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]

    # Precompute literal -> (var_index, is_positive)
    def lit_info(lit):
        var_idx = abs(lit) - 1
        is_pos = lit > 0
        return var_idx, is_pos

    # Precompute all literal infos
    clause_lits = []
    for c in clauses:
        clause_lits.append([(abs(l) - 1, l > 0) for l in c])

    def satisfied(clause_idx):
        for var_idx, is_pos in clause_lits[clause_idx]:
            if assignment[var_idx] == is_pos:
                return True
        return False

    def count_unsat():
        cnt = 0
        for ci in range(m):
            if not satisfied(ci):
                cnt += 1
        return cnt

    # Initial check
    unsat_count = count_unsat()
    if unsat_count == 0:
        return assignment

    # WalkSAT with clause-weighted random restarts
    best_assignment = assignment[:]
    best_unsat = unsat_count

    max_steps = 50000
    p_flip = 0.5  # Probability of flipping a literal in the clause that makes it true

    for step in range(max_steps):
        # Check deadline
        if time.time() - start_time > deadline:
            break

        # Find unsatisfied clauses
        unsat_clauses = []
        for ci in range(m):
            if not satisfied(ci):
                unsat_clauses.append(ci)

        if not unsat_clauses:
            best_assignment = assignment[:]
            best_unsat = 0
            break

        unsat_count = len(unsat_clauses)
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

        # Pick an unsatisfied clause
        ci = rng.choice(unsat_clauses)
        clause = clause_lits[ci]

        # Decide whether to flip a literal that satisfies the clause (greedy) or random
        if rng.random() < p_flip:
            # Flip a literal that would satisfy the clause
            var_idx, is_pos = rng.choice(clause)
            assignment[var_idx] = is_pos
        else:
            # Flip a random literal from the clause
            var_idx, _ = rng.choice(clause)
            assignment[var_idx] = not assignment[var_idx]

        # Random restart if stuck for too long
        if step > 0 and step % 5000 == 0 and unsat_count >= best_unsat:
            # Perturb: flip a few random variables
            for _ in range(3):
                vi = rng.randint(0, n - 1)
                assignment[vi] = not assignment[vi]

    # Final check
    unsat_count = count_unsat()
    if unsat_count == 0:
        return assignment
    if unsat_count < best_unsat:
        return assignment
    return best_assignment
# EVOLVE-BLOCK-END
