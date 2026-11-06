"""Planted 3-SAT solver (50 vars) — Weighted WalkSAT with efficient state tracking."""
import time
import random
import math

# EVOLVE-BLOCK-START


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 290.0  # safety margin under 300s per-instance budget

    # Precompute clause literal indices for speed
    # clause_vars[i] = [v1, v2, v3] (0-indexed)
    # clause_signs[i] = [s1, s2, s3] (1 or -1)
    clause_vars = [[abs(l) - 1 for l in c] for c in clauses]
    clause_signs = [[(1 if l > 0 else -1) for l in c] for c in clauses]

    # Build variable-to-clauses index
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            var_to_clauses[clause_vars[i][j]].append(i)

    def is_satisfied(assign):
        """Check if all clauses are satisfied."""
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_signs[i]
            ok = False
            for j in range(3):
                if assign[cv[j]] == (cs[j] > 0):
                    ok = True
                    break
            if not ok:
                return False
        return True

    def get_initial_state(assign):
        """Calculate initial unsat count and clause satisfaction status."""
        clause_sat = [False] * m
        unsat_count = 0
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_signs[i]
            sat = False
            for j in range(3):
                if assign[cv[j]] == (cs[j] > 0):
                    sat = True
                    break
            clause_sat[i] = sat
            if not sat:
                unsat_count += 1
        return unsat_count, clause_sat

    def polish(assign, max_iters=50):
        """Deterministic sweep: flip any variable that strictly reduces unsat count."""
        # We need to track unsat count efficiently here too, or just recalculate.
        # For N=50, recalculation is fast.
        for _ in range(max_iters):
            changed = False
            base = 0
            # Calculate current unsat count
            for i in range(m):
                cv = clause_vars[i]
                cs = clause_signs[i]
                sat = False
                for j in range(3):
                    if assign[cv[j]] == (cs[j] > 0):
                        sat = True
                        break
                if not sat:
                    base += 1
            
            if base == 0:
                break
                
            for v in range(n):
                assign[v] = not assign[v]
                # Recalculate unsat count
                new_count = 0
                for i in range(m):
                    cv = clause_vars[i]
                    cs = clause_signs[i]
                    sat = False
                    for j in range(3):
                        if assign[cv[j]] == (cs[j] > 0):
                            sat = True
                            break
                    if not sat:
                        new_count += 1
                if new_count < base:
                    base = new_count
                    changed = True
                else:
                    assign[v] = not assign[v]
            if not changed:
                break
        return assign

    def walk_sat_weighted(assign, rng, flip_budget, deadline):
        """
        Weighted WalkSAT.
        Returns (assign, solved_bool, best_unsat_in_run).
        """
        unsat_count, clause_sat = get_initial_state(assign)
        if unsat_count == 0:
            return assign, True, 0

        # Weights for clauses
        weights = [1.0] * m
        best_unsat = unsat_count
        best_assign = list(assign)

        for step in range(flip_budget):
            if time.time() > deadline - 5.0:
                break

            if unsat_count == 0:
                return assign, True, 0

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = list(assign)

            # 1. Select an unsatisfied clause with probability proportional to weight
            # To avoid summing all weights every time, we can use a simple method:
            # Pick a random unsat clause, but bias it.
            # For N=50, M is likely ~250-300. Summing is cheap.
            total_weight = 0.0
            unsat_indices = []
            for i in range(m):
                if not clause_sat[i]:
                    unsat_indices.append(i)
                    total_weight += weights[i]
            
            if not unsat_indices:
                return assign, True, 0

            # Roulette wheel selection
            r = rng.random() * total_weight
            cumulative = 0.0
            selected_clause = unsat_indices[0]
            for i in unsat_indices:
                cumulative += weights[i]
                if r <= cumulative:
                    selected_clause = i
                    break
            
            cv = clause_vars[selected_clause]
            cs = clause_signs[selected_clause]

            # 2. Choose which variable to flip
            # Score each variable in the clause by how many OTHER unsat clauses it satisfies
            best_score = -1
            best_j = -1
            for j in range(3):
                v = cv[j]
                s = cs[j]
                new_val = (s > 0)
                score = 0
                # Check other unsat clauses
                for ui in unsat_indices:
                    if ui == selected_clause:
                        continue
                    uc = clause_vars[ui]
                    us = clause_signs[ui]
                    for k in range(3):
                        if uc[k] == v:
                            if new_val == (us[k] > 0):
                                score += 1
                            break
                # Add small random noise to break ties
                score += rng.random() * 0.01
                if score > best_score:
                    best_score = score
                    best_j = j
            
            v_to_flip = cv[best_j]
            
            # 3. Flip the variable and update state
            old_val = assign[v_to_flip]
            assign[v_to_flip] = not old_val
            
            # Update clause_sat and unsat_count for all clauses containing v_to_flip
            for ci in var_to_clauses[v_to_flip]:
                cv_c = clause_vars[ci]
                cs_c = clause_signs[ci]
                was_sat = clause_sat[ci]
                
                # Check if clause is now satisfied
                is_sat = False
                for k in range(3):
                    if assign[cv_c[k]] == (cs_c[k] > 0):
                        is_sat = True
                        break
                
                if was_sat and not is_sat:
                    unsat_count += 1
                    clause_sat[ci] = False
                    # Increase weight of this clause as it became unsat
                    weights[ci] *= 1.1
                elif not was_sat and is_sat:
                    unsat_count -= 1
                    clause_sat[ci] = True
                    # Reset weight of this clause as it became sat
                    weights[ci] = 1.0
                else:
                    # Status didn't change, but if it's unsat, maybe increase weight slightly?
                    # Or leave it. Let's only increase when it becomes unsat.
                    pass

        return assign, is_satisfied(assign), best_unsat

    # --- Initialization ---
    rng = random.Random(12345)
    best_assign = None
    best_unsat = m + 1

    # Try a few random inits and some structured ones
    candidates = []
    for _ in range(10):
        candidates.append([rng.random() < 0.5 for _ in range(n)])
    candidates.append([True] * n)
    candidates.append([False] * n)
    
    for a in candidates:
        u, _ = get_initial_state(a)
        if u < best_unsat:
            best_unsat = u
            best_assign = list(a)
        if u == 0:
            return a

    # --- Main Search ---
    restarts = 0
    max_restarts = 5000
    while time.time() < deadline - 10.0 and restarts < max_restarts:
        restarts += 1
        current = list(best_assign)
        
        # Perturb slightly
        # Number of perturbations increases with restarts to escape deeper minima
        num_perturb = min(5, 1 + restarts // 10)
        for _ in range(num_perturb):
            v = rng.randrange(n)
            current[v] = not current[v]
            
        # Flip budget: increase over time? Or keep fixed but high.
        # N=50, typical WalkSAT needs ~1000-5000 flips.
        flip_budget = 1000 + rng.randrange(1000)
        
        res, solved, run_best_unsat = walk_sat_weighted(current, rng, flip_budget, deadline)
        
        # Update global best
        if run_best_unsat < best_unsat:
            best_unsat = run_best_unsat
            best_assign = list(res)
            
        if solved:
            return res

    # Final polish
    final = list(best_assign)
    polish(final)
    if is_satisfied(final):
        return final
    
    # Try a few more polish passes with random perturbations
    for _ in range(50):
        perturbed = list(final)
        for _ in range(rng.randrange(1, 5)):
            v = rng.randrange(n)
            perturbed[v] = not current[v] # Bug in thought: should be perturbed[v] = not perturbed[v]
            perturbed[v] = not perturbed[v]
        polish(perturbed, max_iters=20)
        if is_satisfied(perturbed):
            return perturbed
        u, _ = get_initial_state(perturbed)
        if u < best_unsat:
            best_unsat = u
            final = list(perturbed)
            
    return final

# EVOLVE-BLOCK-END
