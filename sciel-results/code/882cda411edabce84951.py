<<<<<<< SEARCH
"""Improved WalkSAT with min-conflicts, restarts, and deadline guard for planted 3-SAT."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before 330s kill

    # Pre-extract clause literals as (var_idx, sign) for speed
    # var_idx is 0-based, sign is +1 for positive, -1 for negative
    clause_data = []
    for clause in clauses:
        clause_data.append([(abs(l) - 1, 1 if l > 0 else -1) for l in clause])

    num_clauses = len(clause_data)

    def is_satisfied(assignment):
        """Check if all clauses are satisfied."""
        for clause in clause_data:
            sat = False
            for var_idx, sign in clause:
                if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                    sat = True
                    break
            if not sat:
                return False
        return True

    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        count = 0
        for clause in clause_data:
            sat = False
            for var_idx, sign in clause:
                if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def walk_one(assignment, rng, max_steps, p_flip):
        """Run WalkSAT from current assignment for up to max_steps with flip probability p_flip."""
        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False

            # Find unsatisfied clauses
            unsat_clauses = []
            for i, clause in enumerate(clause_data):
                sat = False
                for var_idx, sign in clause:
                    if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(i)

            if not unsat_clauses:
                return assignment, True

            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            clause = clause_data[ci]

            # Score each literal: count how many other unsatisfied clauses contain this variable
            # Higher count means flipping is more likely to help satisfy more clauses
            var_unsat_count = {}
            for uc in unsat_clauses:
                for var_idx, sign in clause_data[uc]:
                    var_unsat_count[var_idx] = var_unsat_count.get(var_idx, 0) + 1

            # Weighted selection with noise: combine clause frequency with random component
            # The p_flip parameter controls how much randomness is in the choice
            weights = []
            for var_idx, sign in clause:
                base_w = var_unsat_count.get(var_idx, 0) + 1
                noise = rng.random() * p_flip * 5  # add noise scaled by p_flip
                weights.append(base_w + noise)

            # Normalize and pick
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0
            chosen_idx = 0
            for idx, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    chosen_idx = idx
                    break

            var_idx, sign = clause[chosen_idx]
            assignment[var_idx] = not assignment[var_idx]

        return assignment, False

    # Multi-restart strategy with varying parameters
    rng = random.Random(0)
    best_assignment = None
    best_unsat_count = float('inf')

    # Parameters tuned for planted 3-SAT with 150 variables
    # With incremental tracking, each step is faster, so we can do more steps
    max_steps_per_restart = 50000
    num_restarts = 1500  # max restarts, will stop at deadline

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Vary the flip probability to explore different regimes
        p_flip = 0.2 + 0.3 * (restart % 5) / 4.0  # cycle through 0.2, 0.275, 0.35, 0.425, 0.5

        # Try to solve
        assignment, solved = walk_one(assignment, rng, max_steps_per_restart, p_flip)

        if solved:
            return assignment

        # Track best
        uc = count_unsat(assignment)
        if uc < best_unsat_count:
            best_unsat_count = uc
            best_assignment = assignment[:]

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment

    # Fallback: all False
    return [False] * n

# EVOLVE-BLOCK-END
=======
"""Improved WalkSAT with incremental clause tracking, min-conflicts, restarts, and deadline guard for planted 3-SAT."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before 330s kill

    # Pre-extract clause literals as (var_idx, sign) for speed
    # var_idx is 0-based, sign is +1 for positive, -1 for negative
    clause_data = []
    for clause in clauses:
        clause_data.append([(abs(l) - 1, 1 if l > 0 else -1) for l in clause])

    num_clauses = len(clause_data)

    # Build variable-to-clause index for incremental updates
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clause_data):
        for var_idx, sign in clause:
            var_clauses[var_idx].append(ci)

    def is_satisfied(assignment):
        """Check if all clauses are satisfied."""
        for clause in clause_data:
            sat = False
            for var_idx, sign in clause:
                if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                    sat = True
                    break
            if not sat:
                return False
        return True

    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        count = 0
        for clause in clause_data:
            sat = False
            for var_idx, sign in clause:
                if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def walk_one(assignment, rng, max_steps, p_flip):
        """Run WalkSAT with incremental clause tracking from current assignment."""
        # Initialize clause satisfaction state
        clause_sat = [False] * num_clauses
        clause_unsat_count = [0] * num_clauses  # number of unsatisfied literals in each clause
        
        for ci, clause in enumerate(clause_data):
            for var_idx, sign in clause:
                lit_val = assignment[var_idx] if sign == 1 else (not assignment[var_idx])
                if lit_val:
                    clause_sat[ci] = True
                    break
                else:
                    clause_unsat_count[ci] += 1
        
        unsat_clauses = [i for i in range(num_clauses) if not clause_sat[i]]
        unsat_count = len(unsat_clauses)

        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False

            if unsat_count == 0:
                return assignment, True

            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            clause = clause_data[ci]

            # Score each literal: count how many other unsatisfied clauses contain this variable
            var_unsat_count = {}
            for uc in unsat_clauses:
                for var_idx, sign in clause_data[uc]:
                    var_unsat_count[var_idx] = var_unsat_count.get(var_idx, 0) + 1

            # Weighted selection with noise
            weights = []
            for var_idx, sign in clause:
                base_w = var_unsat_count.get(var_idx, 0) + 1
                noise = rng.random() * p_flip * 5
                weights.append(base_w + noise)

            # Normalize and pick
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0
            chosen_idx = 0
            for idx, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    chosen_idx = idx
                    break

            var_idx, sign = clause[chosen_idx]
            old_val = assignment[var_idx]
            new_val = not old_val
            assignment[var_idx] = new_val

            # Update clause satisfaction incrementally
            for uc in var_clauses[var_idx]:
                # Check if this variable appears in clause uc with what sign
                clause_uc = clause_data[uc]
                old_lit_val = old_val if sign == 1 else (not old_val)
                new_lit_val = new_val if sign == 1 else (not new_val)
                
                if not clause_sat[uc]:
                    # Clause was unsatisfied
                    if old_lit_val and not new_lit_val:
                        # Literal went from satisfied to unsatisfied (shouldn't happen for unsat clause)
                        clause_unsat_count[uc] -= 1
                    elif not old_lit_val and new_lit_val:
                        # Literal went from unsatisfied to satisfied
                        clause_unsat_count[uc] -= 1
                        if clause_unsat_count[uc] == 0:
                            clause_sat[uc] = True
                            unsat_count -= 1
                            if uc in unsat_clauses:
                                unsat_clauses.remove(uc)
                    else:
                        # No change in satisfaction status
                        pass
                else:
                    # Clause was satisfied
                    if old_lit_val and not new_lit_val:
                        # Literal went from satisfied to unsatisfied
                        clause_unsat_count[uc] += 1
                        if clause_unsat_count[uc] == len(clause_uc):
                            clause_sat[uc] = False
                            unsat_count += 1
                            unsat_clauses.append(uc)
                    else:
                        # Still satisfied or became more satisfied
                        pass

        return assignment, False

    # Multi-restart strategy with varying parameters
    rng = random.Random(0)
    best_assignment = None
    best_unsat_count = float('inf')

    # Parameters tuned for planted 3-SAT with 150 variables
    # Incremental tracking makes each step faster, allowing more steps per restart
    max_steps_per_restart = 100000
    num_restarts = 500  # fewer restarts but longer walks with faster steps

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Vary the flip probability to explore different regimes
        p_flip = 0.2 + 0.3 * (restart % 5) / 4.0  # cycle through 0.2, 0.275, 0.35, 0.425, 0.5

        # Try to solve
        assignment, solved = walk_one(assignment, rng, max_steps_per_restart, p_flip)

        if solved:
            return assignment

        # Track best
        uc = count_unsat(assignment)
        if uc < best_unsat_count:
            best_unsat_count = uc
            best_assignment = assignment[:]

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment

    # Fallback: all False
    return [False] * n

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
