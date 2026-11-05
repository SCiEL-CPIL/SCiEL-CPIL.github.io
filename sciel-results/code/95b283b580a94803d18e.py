"""Optimized 3-SAT solver with cached clause satisfaction and min-confl WalkSAT."""
# EVOLVE-BLOCK-START
import random
import time
from collections import defaultdict


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    deadline = time.time() + 285.0  # safety margin below 330s kill
    num_clauses = len(clauses)

    # Build variable-to-clauses index for fast incremental updates
    var_to_clauses = defaultdict(list)
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_to_clauses[abs(lit) - 1].append(ci)

    # Precompute for each clause: list of (var_idx, target_val) for fast satisfaction check
    clause_info = []
    for clause in clauses:
        info = []
        for lit in clause:
            var_idx = abs(lit) - 1
            target_val = (lit > 0)
            info.append((var_idx, target_val))
        clause_info.append(info)

    # Track satisfaction status of each clause to avoid redundant scans
    clause_sat = [False] * num_clauses

    def is_clause_satisfied(ci, assignment):
        for var_idx, target_val in clause_info[ci]:
            if assignment[var_idx] == target_val:
                return True
        return False

    def init_clause_sat(assignment):
        for ci in range(num_clauses):
            clause_sat[ci] = is_clause_satisfied(ci, assignment)

    def compute_unsat_list(assignment):
        init_clause_sat(assignment)
        unsat = [ci for ci in range(num_clauses) if not clause_sat[ci]]
        return unsat

    def update_clause_sat_after_flip(var_idx, assignment):
        """Update clause_sat for all clauses containing var_idx."""
        for ci in var_to_clauses[var_idx]:
            clause_sat[ci] = is_clause_satisfied(ci, assignment)

    def make_random_assignment(seed):
        rng = random.Random(seed)
        return [rng.random() < 0.5 for _ in range(n)]

    def walksat_step(assignment, unsat_list, rng, p_walk):
        """Perform one WalkSAT step with min-confl heuristic.
        Returns (new_unsat_list, flipped_var, old_val) or None if no unsat clauses."""
        if not unsat_list:
            return unsat_list, -1, None

        # Pick a random unsatisfied clause
        ci = rng.choice(unsat_list)
        clause = clause_info[ci]

        # Min-confl: for each literal in the clause, count how many OTHER satisfied clauses it would break
        best_var = None
        best_score = float('inf')
        best_lit_val = None

        for var_idx, target_val in clause:
            # Count how many currently-satisfied clauses would become unsat if we flip var_idx
            conflict = 0
            for other_ci in var_to_clauses[var_idx]:
                if other_ci == ci:
                    continue
                if not clause_sat[other_ci]:
                    # Already unsat, no conflict
                    continue
                # This clause is satisfied. Check if flipping var_idx would break it.
                # Find the literal for var_idx in this clause
                for v_idx, t_val in clause_info[other_ci]:
                    if v_idx == var_idx:
                        if assignment[var_idx] == t_val:
                            # This literal is satisfied. Flipping makes it unsatisfied.
                            # Check if any other literal in this clause is satisfied
                            other_lit_sat = False
                            for v2, t2 in clause_info[other_ci]:
                                if v2 != var_idx and assignment[v2] == t2:
                                    other_lit_sat = True
                                    break
                            if not other_lit_sat:
                                conflict += 1
                        break

            score = conflict
            if score < best_score:
                best_score = score
                best_var = var_idx
                best_lit_val = target_val

        if best_var is None:
            return unsat_list, -1, None

        # WalkSAT: with probability p_walk, pick random literal from clause
        if rng.random() < p_walk:
            var_idx, target_val = rng.choice(clause)
            best_var = var_idx
            best_lit_val = target_val

        old_val = assignment[best_var]
        new_val = not old_val
        assignment[best_var] = new_val

        # Update clause_sat for affected clauses
        update_clause_sat_after_flip(best_var, assignment)

        # Rebuild unsat list from updated clause_sat
        new_unsat_list = [ci for ci in range(num_clauses) if not clause_sat[ci]]

        return new_unsat_list, best_var, old_val

    def final_polish(assignment, unsat_list, rng, max_steps):
        """Final high-precision polish with tighter parameters."""
        cur_unsat = len(unsat_list)
        if cur_unsat == 0:
            return assignment, 0

        p_walk = 0.02
        steps = 0
        while cur_unsat > 0 and steps < max_steps:
            if time.time() > deadline:
                break
            steps += 1
            new_unsat_list, flipped_var, old_val = walksat_step(assignment, unsat_list, rng, p_walk)
            cur_unsat = len(new_unsat_list)
            if cur_unsat == 0:
                return assignment, 0
            unsat_list = new_unsat_list
            if cur_unsat < len(unsat_list):
                p_walk = max(0.01, p_walk * 0.9)
            else:
                if steps % 50 == 0:
                    p_walk = min(0.3, p_walk * 1.2)
        return assignment, cur_unsat

    best_assignment = None
    best_unsat = num_clauses + 1

    # Try multiple random restarts with WalkSAT + min-confl
    for attempt in range(8000):
        if time.time() > deadline:
            break

        assignment = make_random_assignment(attempt * 7919 + 13)
        unsat_list = compute_unsat_list(assignment)
        cur_unsat = len(unsat_list)

        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        if cur_unsat == 0:
            return assignment

        # Local search: WalkSAT with dynamic parameters
        rng = random.Random(attempt * 31337 + 42)
        steps = 0
        max_steps = 15000

        # Adaptive walk probability: start low, increase if stuck
        p_walk = 0.05
        stuck_counter = 0
        last_improvement_step = 0

        while cur_unsat > 0 and steps < max_steps:
            if time.time() > deadline:
                break
            steps += 1

            new_unsat_list, flipped_var, old_val = walksat_step(assignment, unsat_list, rng, p_walk)
            cur_unsat = len(new_unsat_list)
            unsat_list = new_unsat_list

            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]
                last_improvement_step = steps
            if cur_unsat == 0:
                return assignment

            # Track if we're stuck
            if steps - last_improvement_step > 200:
                stuck_counter += 1
                if stuck_counter > 3:
                    p_walk = min(0.5, p_walk * 1.5)
                    stuck_counter = 0
            else:
                stuck_counter = 0
                if cur_unsat < best_unsat:
                    p_walk = max(0.02, p_walk * 0.85)

            # If close to solution, do a polish pass
            if cur_unsat <= 3 and steps > 5000:
                polish_assignment, polish_unsat = final_polish(assignment, unsat_list, rng, 5000)
                if polish_unsat == 0:
                    return polish_assignment
                if polish_unsat < cur_unsat:
                    assignment = polish_assignment
                    cur_unsat = polish_unsat
                    unsat_list = compute_unsat_list(assignment)

    # Final attempt: if we have a good partial solution, try to polish it
    if best_assignment is not None and best_unsat <= 5:
        rng = random.Random(99999)
        polish_assignment, polish_unsat = final_polish(best_assignment, compute_unsat_list(best_assignment), rng, 20000)
        if polish_unsat == 0:
            return polish_assignment

    return best_assignment if best_assignment else make_random_assignment(0)
# EVOLVE-BLOCK-END
