"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    """Satisfying assignment for a locally generated planted satisfiable 3-CNF.

    instance: dict with 'n' (int) and 'clauses' (list of lists of signed 1-based ints).
    Returns: list of booleans (length n) such that every clause is satisfied.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Deadline guard: evaluator kills at 330s; keep a safety margin.
    start = time.time()
    deadline = start + 295.0  # 35s margin

    # Precompute literal arrays for vectorized evaluation.
    # For each clause, store (var_index, sign) where sign is +1 for positive, -1 for negative
    # lit_vars[c][k] = 0-based variable index
    # lit_signs[c][k] = +1 or -1
    lit_vars = np.empty((m, 3), dtype=np.int32)
    lit_signs = np.empty((m, 3), dtype=np.int8)
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            lit_vars[i, j] = abs(lit) - 1
            lit_signs[i, j] = 1 if lit > 0 else -1

    # Assignment as numpy array of 0/1
    def make_random_assign(rng):
        return np.array([rng.random() < 0.5 for _ in range(n)], dtype=np.int8)

    def is_satisfied(assign):
        """Return boolean array of length m: True if clause i is satisfied."""
        # For each literal: assign[var] * sign == 1 means literal is true
        # A clause is satisfied if any of its 3 literals is true
        # lit_val[c, k] = assign[lit_vars[c,k]] * lit_signs[c,k]
        lit_vals = assign[lit_vars] * lit_signs  # shape (m, 3)
        return np.any(lit_vals == 1, axis=1)

    def count_unsat(assign):
        sat = is_satisfied(assign)
        return np.sum(~sat)

    # WalkSAT with clause weight tracking for better variable selection
    def walksat(assign, rng, max_steps, clause_weights):
        """Run WalkSAT. Returns (assignment, solved, unsat_count_at_end)."""
        n_vars = n
        current_unsat = count_unsat(assign)
        if current_unsat == 0:
            return assign, True, 0

        # Track which clauses are unsatisfied
        sat_mask = is_satisfied(assign)
        unsat_indices = np.where(~sat_mask)[0]

        for step in range(max_steps):
            if current_unsat == 0:
                return assign, True, 0

            # Pick a random unsatisfied clause
            if len(unsat_indices) == 0:
                return assign, True, 0
            ci = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]

            # Get the three literals in this clause
            vars_in_clause = lit_vars[ci]
            signs_in_clause = lit_signs[ci]

            # For each literal, check if flipping the variable would satisfy the clause
            # and count how many currently unsatisfied clauses would be broken
            best_var = -1
            best_score = float('inf')

            for k in range(3):
                v = vars_in_clause[k]
                s = signs_in_clause[k]
                # Flipping v: new assignment at v = 1 - assign[v]
                # The literal (v, s) is satisfied after flip if (1-assign[v]) * s == 1
                # i.e., assign[v] == 0 if s==1, or assign[v] == 1 if s==-1
                # i.e., assign[v] != s (since s is 1 or -1, and assign is 0 or 1)
                # Wait: if s=1, literal true when assign[v]=1. After flip, assign[v]=1-assign[v].
                # We want 1-assign[v] == 1 => assign[v] == 0.
                # If s=-1, literal true when assign[v]=0. After flip, assign[v]=1-assign[v].
                # We want 1-assign[v] == 0 => assign[v] == 1.
                # So literal is satisfied after flip iff assign[v] == (1 if s==1 else 0) == (s+1)//2? 
                # Actually: assign[v] should be 0 if s=1, 1 if s=-1. That's assign[v] == 1-s... no.
                # s=1: want assign[v]=0. 1-s = 0. Yes.
                # s=-1: want assign[v]=1. 1-s = 2. No.
                # Let me just check: after flip, assign[v] becomes 1-assign[v].
                # Literal value = (1-assign[v]) * s. We want this == 1.
                # (1-assign[v]) * s == 1 => 1-assign[v] == 1/s == s (since s=±1)
                # => assign[v] == 1 - s
                # s=1: assign[v] == 0. Correct.
                # s=-1: assign[v] == 2. But assign is 0/1, so this means it's always satisfied after flip?
                # Wait, that can't be right. Let me reconsider.
                # If s = -1, literal is ¬v. It's true when assign[v] = 0.
                # After flip, assign[v] = 1 - assign[v].
                # We want 1 - assign[v] = 0 => assign[v] = 1.
                # So for s=-1, we need current assign[v] = 1.
                # For s=1, we need current assign[v] = 0.
                # In both cases: assign[v] == 1 - (1 if s==1 else 0)... 
                # s=1: 1-1=0. s=-1: 1-0=1. So assign[v] should be (1 if s==-1 else 0) = (1-s)/2? 
                # (1-1)/2 = 0 for s=1. (1-(-1))/2 = 1 for s=-1. Yes!
                # So literal is satisfied after flip iff assign[v] == (1-s)//2... but (1-s)/2 for s=-1 is 1.
                # Actually let's just compute: target_val = 1 if s == 1 else 0
                # satisfied_after_flip = (assign[v] != target_val) ... no wait.
                # We flip assign[v] to 1-assign[v]. The literal value becomes (1-assign[v])*s.
                # This equals 1 iff (1-assign[v]) == s (since s=±1 and 1/s=s).
                # So 1-assign[v] == s => assign[v] == 1-s.
                # s=1: assign[v]==0. s=-1: assign[v]==2 (impossible, so always true?).
                # Hmm, for s=-1: 1-assign[v] == -1 => assign[v] == 2. Since assign is 0/1, this is never true.
                # That means for s=-1, flipping NEVER satisfies the literal? That's wrong.
                # Let me re-examine. Literal ¬v (s=-1) is true when v=false (assign=0).
                # If currently assign[v]=1 (v=true), flipping makes assign[v]=0 (v=false), literal becomes true.
                # So flipping DOES satisfy ¬v when currently assign[v]=1.
                # My formula: 1-assign[v] == s => 1-1 == -1 => 0 == -1. False. But it should be true!
                # The issue: literal value = assign[v] * s? No.
                # If s=1, literal is v, value = assign[v].
                # If s=-1, literal is ¬v, value = 1 - assign[v].
                # So literal value = assign[v] if s=1, else 1-assign[v].
                # = assign[v] * (1 if s=1 else 0) + (1-assign[v]) * (1 if s=-1 else 0)
                # = assign[v] * (s+1)/2 + (1-assign[v]) * (1-s)/2
                # For s=1: assign[v] * 1 + (1-assign[v]) * 0 = assign[v]. Correct.
                # For s=-1: assign[v] * 0 + (1-assign[v]) * 1 = 1-assign[v]. Correct.
                # After flip, assign[v] -> 1-assign[v]:
                # New value = (1-assign[v]) * (s+1)/2 + assign[v] * (1-s)/2
                # For s=1: (1-assign[v]) * 1 + assign[v] * 0 = 1-assign[v]. We want this = 1 => assign[v]=0.
                # For s=-1: (1-assign[v]) * 0 + assign[v] * 1 = assign[v]. We want this = 1 => assign[v]=1.
                # So satisfied after flip iff:
                #   s=1: assign[v] == 0
                #   s=-1: assign[v] == 1
                # In both cases: assign[v] == (1 if s==-1 else 0) = max(0, -s) ... 
                # s=1: 0. s=-1: 1. So assign[v] == (1 - s) // 2? (1-1)//2=0, (1-(-1))//2=1. Yes!
                # But (1-s)//2 for integers: s=1 -> 0, s=-1 -> 1. Correct.

                # Simpler: just check if flipping v makes this literal true
                new_assign_v = 1 - assign[v]
                if s == 1:
                    lit_true_after_flip = (new_assign_v == 1)
                else:
                    lit_true_after_flip = (new_assign_v == 0)

                if not lit_true_after_flip:
                    # Flipping v does not satisfy this literal, so it won't help this clause directly
                    # But it might still be the best choice if it breaks few other clauses
                    pass

                # Count how many currently unsatisfied clauses would be broken by flipping v
                # A currently unsatisfied clause c is "broken" (becomes more unsatisfied) if:
                #   - It contains v, and the literal of v in c was the only thing preventing... 
                #   Actually, for an unsatisfied clause, all literals are false.
                #   Flipping v changes the value of the literal involving v.
                #   If the new value is true, the clause becomes satisfied (good).
                #   If the new value is still false, no change.
                #   We can't make an already-unsatisfied clause "more unsatisfied" in 3-SAT with 3 literals,
                #   because if all 3 were false and one becomes true, it's satisfied. If it stays false, still unsat.
                #   So flipping can only turn unsat->sat, never sat->unsat for the clause containing the flipped var.
                #   But for OTHER clauses that contain v, flipping v might turn a satisfied clause to unsatisfied.
                
                # Count satisfied clauses that contain v and would become unsatisfied
                # This is expensive to do exactly. Use a heuristic: count based on clause weights.
                # For simplicity, use the standard WalkSAT heuristic:
                #   score = (number of unsat clauses in this clause that would be fixed) 
                #           - (number of sat clauses that would be broken)
                # Since we can't easily count the second part, use a simpler approach:
                # Pick randomly with some probability, otherwise pick the variable that appears in most unsat clauses.

            # Simpler WalkSAT: with probability p, flip random variable in clause; else flip variable that minimizes "bad" flips
            p = 0.5
            if rng.random() < p:
                k = rng.randint(0, 2)
                v = vars_in_clause[k]
            else:
                # Pick variable that appears in the most unsatisfied clauses (heuristic)
                # Use clause weights as proxy
                scores = []
                for k in range(3):
                    v = vars_in_clause[k]
                    # Heuristic: prefer variables with higher weight
                    scores.append(clause_weights[v] if v < len(clause_weights) else 0)
                # Add some randomness
                max_score = max(scores)
                candidates = [k for k in range(3) if scores[k] == max_score]
                k = candidates[rng.randint(0, len(candidates) - 1)]
                v = vars_in_clause[k]

            # Flip variable v
            assign[v] = 1 - assign[v]

            # Update unsat count (recompute periodically for simplicity)
            if step % 10 == 0:
                current_unsat = count_unsat(assign)
                sat_mask = is_satisfied(assign)
                unsat_indices = np.where(~sat_mask)[0]
                # Update clause weights: increase weight of variables in unsat clauses
                if len(unsat_indices) > 0:
                    for ci in unsat_indices:
                        for k in range(3):
                            v = lit_vars[ci, k]
                            clause_weights[v] += 1
                # Occasionally decay weights
                if step % 100 == 0:
                    clause_weights *= 0.9

        return assign, False, current_unsat

    # Multiple restarts with different seeds
    best_assign = None
    best_unsat = float('inf')

    # Initial random assignment
    rng = random.Random(0)
    assign = make_random_assign(rng)
    unsat_count = count_unsat(assign)
    if unsat_count == 0:
        return assign.tolist()
    best_assign = assign.copy()
    best_unsat = unsat_count

    # Restart loop
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        # Create new random assignment
        seed = rng.randint(0, 2**31 - 1)
        rng_restart = random.Random(seed)
        assign = make_random_assign(rng_restart)
        unsat_count = count_unsat(assign)
        if unsat_count == 0:
            return assign.tolist()
        
        if unsat_count < best_unsat:
            best_assign = assign.copy()
            best_unsat = unsat_count

        # Run WalkSAT with a step budget
        # Budget scales with problem size
        max_steps = 50000
        clause_weights = np.zeros(n, dtype=np.float64)
        
        assign, solved, final_unsat = walksat(assign, rng_restart, max_steps, clause_weights)
        
        if solved:
            return assign.tolist()
        
        if final_unsat < best_unsat:
            best_assign = assign.copy()
            best_unsat = final_unsat

        # Check deadline
        if time.time() > deadline - 5.0:
            break

    # Return best assignment found
    if best_assign is not None:
        return best_assign.tolist()
    return [False] * n
# EVOLVE-BLOCK-END
