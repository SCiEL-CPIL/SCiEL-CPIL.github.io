"""Improved 3-SAT solver using optimized WalkSAT with literal-count tracking."""

import random
import time

# Module-level state for time management
_start_time = time.time()
# Assume 300s total limit, leave 10s buffer for safety/overhead
_DEADLINE = _start_time + 290.0


def solve(instance):
    """
    Solves a 3-SAT instance.
    Returns a list of booleans representing the assignment.
    """
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    # Pre-process: Map variable indices to clauses containing them
    var_to_clauses = [[] for _ in range(n)]
    # For each clause, store the position of each variable's literal
    # clause_var_pos[c_idx][var_idx] = position in clause (0,1,2) or -1 if not present
    clause_var_pos = []
    # clause_literals[c_idx] = list of (var_idx, is_pos)
    clause_literals = []

    for idx, clause in enumerate(clauses):
        var_pos = {}
        lits = []
        for pos, lit in enumerate(clause):
            var_idx = abs(lit) - 1
            is_pos = lit > 0
            var_pos[var_idx] = pos
            lits.append((var_idx, is_pos))
        clause_var_pos.append(var_pos)
        clause_literals.append(lits)
        for lit in clause:
            var_idx = abs(lit) - 1
            var_to_clauses[var_idx].append(idx)

    def compute_state(assignment):
        """Compute clause_satisfied and literal_true_counts for a given assignment."""
        clause_satisfied = [False] * m
        lit_true_counts = [0] * m  # number of true literals per clause
        for i, lits in enumerate(clause_literals):
            cnt = 0
            for var_idx, is_pos in lits:
                if assignment[var_idx] == is_pos:
                    cnt += 1
            lit_true_counts[i] = cnt
            clause_satisfied[i] = (cnt > 0)
        unsat_count = sum(1 for s in clause_satisfied if not s)
        return clause_satisfied, lit_true_counts, unsat_count

    # Initialize random assignment
    assignment = [random.random() < 0.5 for _ in range(n)]
    clause_satisfied, lit_true_counts, unsat_count = compute_state(assignment)

    # Best solution tracking
    best_assignment = assignment.copy()
    best_unsat_count = unsat_count

    if unsat_count == 0:
        return assignment

    # WalkSAT with restarts and adaptive parameters
    max_restarts = 200
    base_steps = 3000
    flip_threshold = 0.35  # 35% random, 65% best (more exploitation)

    # Maintain a small pool of unsatisfied clause indices to avoid O(m) scans.
    def rebuild_unsat_pool():
        pool = [i for i in range(m) if not clause_satisfied[i]]
        return pool

    unsat_pool = rebuild_unsat_pool()
    # Keep the pool reasonably fresh; rebuild if it is empty or if it looks stale.
    pool_rebuild_counter = 0

    # Pre-compute for each variable, the list of (clause_idx, position_in_clause)
    var_clause_pos = [[] for _ in range(n)]
    for c_idx in range(m):
        for var_idx, pos in clause_var_pos[c_idx].items():
            var_clause_pos[var_idx].append((c_idx, pos))

    for restart in range(max_restarts):
        # Check time
        if time.time() > _DEADLINE - 2:
            return best_assignment

        # Reset to random assignment for each restart
        assignment = [random.random() < 0.5 for _ in range(n)]
        clause_satisfied, lit_true_counts, unsat_count = compute_state(assignment)

        if unsat_count == 0:
            return assignment

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()

        # Adaptive steps: more steps early, fewer later
        steps = base_steps

        for step in range(steps):
            # Check time every 100 steps for efficiency
            if step % 100 == 0 and time.time() > _DEADLINE - 2:
                return best_assignment

            # Find unsatisfied clauses - maintain a small pool for O(1) random access
            if unsat_count == 0:
                return assignment

            # Rebuild the pool occasionally or when it becomes unusable.
            pool_rebuild_counter += 1
            if not unsat_pool or pool_rebuild_counter >= 64:
                unsat_pool = rebuild_unsat_pool()
                pool_rebuild_counter = 0
                if not unsat_pool:
                    return assignment

            # Pick random unsatisfied clause from the pool
            clause_idx = random.choice(unsat_pool)
            clause = clause_literals[clause_idx]

            # Decide: random flip or best flip
            if random.random() < flip_threshold:
                # Random flip: pick random variable from clause
                var_idx, is_pos = random.choice(clause)
            else:
                # Best flip: pick variable that minimizes increase in unsat count
                # For 3-CNF, delta computation is fast with lit_true_counts
                best_var = clause[0][0]
                best_delta = float('inf')

                for var_idx, is_pos in clause:
                    # Calculate delta if we flip var_idx
                    delta = 0
                    for c_idx, pos in var_clause_pos[var_idx]:
                        # Current state of this clause
                        old_cnt = lit_true_counts[c_idx]
                        old_sat = old_cnt > 0

                        # Was the literal for var_idx in this clause true?
                        v, p = clause_literals[c_idx][pos]
                        lit_was_true = (assignment[v] == p)

                        # New count after flipping
                        if lit_was_true:
                            new_cnt = old_cnt - 1
                        else:
                            new_cnt = old_cnt + 1

                        new_sat = new_cnt > 0

                        if old_sat and not new_sat:
                            delta += 1
                        elif not old_sat and new_sat:
                            delta -= 1

                    if delta < best_delta:
                        best_delta = delta
                        best_var = var_idx
                    elif delta == best_delta and random.random() < 0.5:
                        best_var = var_idx

                var_idx = best_var

            # Flip the variable
            assignment[var_idx] = not assignment[var_idx]

            # Update clause satisfaction efficiently using lit_true_counts
            for c_idx, pos in var_clause_pos[var_idx]:
                v, p = clause_literals[c_idx][pos]
                # The literal truth value just flipped
                if assignment[v] == p:
                    # Literal is now true
                    lit_true_counts[c_idx] += 1
                    if not clause_satisfied[c_idx]:
                        clause_satisfied[c_idx] = True
                        unsat_count -= 1
                else:
                    # Literal is now false
                    lit_true_counts[c_idx] -= 1
                    if lit_true_counts[c_idx] == 0:
                        clause_satisfied[c_idx] = False
                        unsat_count += 1

            if unsat_count == 0:
                return assignment

            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment.copy()

            # Keep the unsat pool roughly in sync with the current state.
            # We only need a pool of *some* unsat clauses, so we can rebuild
            # lazily; but if the chosen clause is now satisfied, drop it.
            if clause_satisfied[clause_idx]:
                # Remove this index from the pool (small list; linear is fine).
                try:
                    unsat_pool.remove(clause_idx)
                except ValueError:
                    pass

    return best_assignment
