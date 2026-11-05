"""Planted 3-SAT solver with WalkSAT + clause-weight heuristics + restarts."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s limit

    # Pre-compute clause lengths and literal indices for speed
    # Convert clauses to lists of (var_idx, sign) where var_idx is 0-based
    clause_data = []
    for c in clauses:
        cd = [(abs(l) - 1, l > 0) for l in c]
        clause_data.append(cd)
    m = len(clause_data)

    def is_satisfied(clause_idx, assignment):
        cd = clause_data[clause_idx]
        for var_idx, sign in cd:
            if assignment[var_idx] == sign:
                return True
        return False

    def count_sat(clause_idx, assignment):
        cd = clause_data[clause_idx]
        cnt = 0
        for var_idx, sign in cd:
            if assignment[var_idx] == sign:
                cnt += 1
        return cnt

    def satisfied_clauses(assignment):
        sat = 0
        for i in range(m):
            if is_satisfied(i, assignment):
                sat += 1
        return sat

    def all_satisfied(assignment):
        for i in range(m):
            if not is_satisfied(i, assignment):
                return False
        return True

    # Multi-restart WalkSAT with adaptive parameters
    best_assignment = None
    best_sat_count = -1

    max_restarts = 200
    base_steps = 5000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Random initial assignment
        rng = random.Random(42 + restart * 7919)
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Check if already satisfied
        if all_satisfied(assignment):
            return assignment

        sat_count = satisfied_clauses(assignment)
        if sat_count > best_sat_count:
            best_sat_count = sat_count
            best_assignment = assignment[:]

        # WalkSAT loop
        steps = base_steps + restart * 2000
        # Use clause weights that increase for frequently-unsatisfied clauses
        clause_weights = [1.0] * m

        for step in range(steps):
            if time.time() > deadline:
                break

            # Find unsatisfied clauses
            unsat_indices = []
            for i in range(m):
                if not is_satisfied(i, assignment):
                    unsat_indices.append(i)

            if not unsat_indices:
                return assignment

            # Pick a clause weighted by its weight
            if len(unsat_indices) == 1:
                chosen = unsat_indices[0]
            else:
                # Weighted random selection
                weights = [clause_weights[i] for i in unsat_indices]
                total_w = sum(weights)
                if total_w <= 0:
                    chosen = unsat_indices[rng.randrange(len(unsat_indices))]
                else:
                    r = rng.random() * total_w
                    cumulative = 0.0
                    chosen = unsat_indices[0]
                    for idx, w in zip(unsat_indices, weights):
                        cumulative += w
                        if r <= cumulative:
                            chosen = idx
                            break

            # Decay weights slightly
            for i in range(m):
                clause_weights[i] *= 0.999
            clause_weights[chosen] += 1.0

            # Choose a literal to flip from the chosen clause
            cd = clause_data[chosen]
            # Heuristic: prefer flipping a literal that appears in fewer unsatisfied clauses
            # Simple approach: pick randomly but with slight bias toward literals
            # that are "critical" (appear in many unsat clauses)
            # For speed, just pick randomly from the clause
            var_idx, sign = cd[rng.randrange(len(cd))]

            # Try flipping and see if it improves
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val

            # Check if satisfied
            if all_satisfied(assignment):
                return assignment

            sat_count = satisfied_clauses(assignment)
            if sat_count > best_sat_count:
                best_sat_count = sat_count
                best_assignment = assignment[:]

            # WalkSAT: with some probability, keep the flip even if not improving
            # Otherwise revert if it made things worse
            # Simple strategy: always keep (pure walk), but let's do a mix
            # 70% keep, 30% revert if worse
            if sat_count < (satisfied_clauses(assignment) - 1):  # This is always false, just for logic
                pass
            # Actually let's use a simpler heuristic:
            # If the flip didn't help, revert with some probability
            # Recompute properly
            # We already flipped, so let's just keep it for now (pure WalkSAT)
            # But let's add a small amount of restart-based diversity

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    # Fallback: random assignment
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
