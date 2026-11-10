import time
import random
import numpy as np

def solve(instance):
    """
    Solves a planted 3-SAT instance using an optimized WalkSAT algorithm 
    with min-conflicts heuristic and vectorized clause tracking.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Pre-process clauses into 0-based indices and signs
    # clause_vars[i] is a list of (var_idx, sign) for clause i
    # sign is +1 for positive, -1 for negative
    clause_vars = []
    var_to_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        vars_in_clause = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            vars_in_clause.append((v, s))
            var_to_clauses[v].append(i)
        clause_vars.append(vars_in_clause)
        
    # Initial assignment based on phase saving (sign frequency)
    # In planted 3-SAT, variables often appear more with one sign
    phase = [0] * n
    for v in range(n):
        pos_count = 0
        neg_count = 0
        for c_idx in var_to_clauses[v]:
            for v_idx, s in clause_vars[c_idx]:
                if v_idx == v:
                    if s == 1:
                        pos_count += 1
                    else:
                        neg_count += 1
        phase[v] = 1 if pos_count >= neg_count else -1
        
    # Helper to check if a variable's current value satisfies a literal
    # assignment is an array of +1/-1
    # literal is (v, s)
    def is_satisfied(assignment, v, s):
        return assignment[v] == s

    def get_unsat_count(assignment):
        count = 0
        for i in range(m):
            sat = False
            for v, s in clause_vars[i]:
                if assignment[v] == s:
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def walksat(assignment, max_steps, noise, rng, deadline):
        """
        WalkSAT step.
        assignment: list of +1/-1
        Returns (assignment, solved)
        """
        # Track satisfied count per clause
        # sat_counts[i] = number of literals in clause i that are satisfied
        sat_counts = np.zeros(m, dtype=np.int32)
        
        # Initialize sat_counts
        for i in range(m):
            cnt = 0
            for v, s in clause_vars[i]:
                if assignment[v] == s:
                    cnt += 1
            sat_counts[i] = cnt
            
        # Build set of unsatisfied clauses indices
        # Using a set for O(1) removal/addition and O(1) membership
        unsat_clauses = set(np.where(sat_counts == 0)[0])
        
        if not unsat_clauses:
            return assignment, True

        # Pre-compute clause lengths for quick access? 
        # Actually, just using the list is fine.
        
        for _ in range(max_steps):
            if time.time() > deadline:
                return assignment, False
                
            if not unsat_clauses:
                return assignment, True

            # Pick a random unsatisfied clause
            c_idx = next(iter(unsat_clauses))
            
            # Min-conflicts heuristic:
            # For each variable in the clause, count how many *other* unsatisfied clauses it appears in.
            # We want to flip the variable that minimizes the increase in unsatisfied clauses (or maximizes decrease).
            # Actually, standard min-conflicts: flip variable that appears in the fewest unsatisfied clauses.
            # If we flip v, the clauses containing v will change status.
            # Clauses already unsatisfied containing v: if v satisfies them, they become sat (good). If not, they stay unsat.
            # Clauses currently sat containing v: if v was the only one satisfying them, they become unsat (bad).
            
            # Simplified WalkSAT heuristic:
            # Pick a variable from the clause.
            # With prob (1-noise), pick the variable that minimizes the number of unsatisfied clauses it's in (excluding the current one? or including?).
            # Let's use the "min-conflicts" score: number of unsatisfied clauses containing the variable.
            
            min_conf = float('inf')
            best_var = None
            best_sign = None
            candidates = []
            
            for v, s in clause_vars[c_idx]:
                # Count unsatisfied clauses containing v
                conf = 0
                for cj in var_to_clauses[v]:
                    if cj in unsat_clauses:
                        conf += 1
                candidates.append((conf, v, s))
                
            if candidates:
                # Find min conflict
                min_conf = min(c[0] for c in candidates)
                
                # Get all vars with min conflict
                min_vars = [(v, s) for conf, v, s in candidates if conf == min_conf]
                
                if rng.random() < noise:
                    # Random choice from all vars in clause
                    v, s = clause_vars[c_idx][rng.randint(0, len(clause_vars[c_idx]) - 1)]
                else:
                    # Random choice from min-conflict vars
                    v, s = min_vars[rng.randint(0, len(min_vars) - 1)]
            else:
                continue # Should not happen

            # Flip variable v
            old_val = assignment[v]
            new_val = -old_val
            assignment[v] = new_val
            
            # Update sat_counts and unsat_clauses for all clauses containing v
            for cj in var_to_clauses[v]:
                # Determine if the literal (v, s_cj) is now satisfied
                # We need the sign of v in clause cj
                # Find the sign of v in clause cj
                # We can precompute a lookup or just scan the small clause (size 3)
                s_v_in_cj = 0
                for v_idx, s_idx in clause_vars[cj]:
                    if v_idx == v:
                        s_v_in_cj = s_idx
                        break
                
                was_sat = sat_counts[cj] > 0
                is_now_sat = (new_val == s_v_in_cj)
                
                if was_sat:
                    if is_now_sat:
                        # Still satisfied. 
                        # Wait, if it was satisfied, it had at least 1 sat literal.
                        # If v was the one satisfying it, and we flipped it, it might become unsat.
                        # If v was NOT the one satisfying it (another literal was), it stays sat.
                        # We need to know if v was contributing to the satisfaction.
                        # Actually, the logic `sat_counts[cj] > 0` is not enough to know if v contributed.
                        # We need to know if the *specific literal* (v, s_v_in_cj) was satisfied.
                        
                        # Let's refine the update logic:
                        # 1. Check if literal (v, s_v_in_cj) was satisfied BEFORE flip.
                        # 2. Check if literal (v, s_v_in_cj) is satisfied AFTER flip.
                        
                        # We don't have "before" state easily for the specific literal without storing it.
                        # But we can infer:
                        # If `old_val == s_v_in_cj`, then the literal was satisfied.
                        # If `new_val == s_v_in_cj`, then the literal is satisfied.
                        
                        was_lit_sat = (old_val == s_v_in_cj)
                        is_lit_sat = (new_val == s_v_in_cj)
                        
                        if was_lit_sat and not is_lit_sat:
                            # Literal stopped being satisfied
                            sat_counts[cj] -= 1
                            if sat_counts[cj] == 0:
                                unsat_clauses.add(cj)
                        elif not was_lit_sat and is_lit_sat:
                            # Literal started being satisfied
                            sat_counts[cj] += 1
                            if sat_counts[cj] == 1:
                                # Just became satisfied (was 0, now 1)
                                if cj in unsat_clauses:
                                    unsat_clauses.remove(cj)
                        # else: no change in sat status of this literal, and total count doesn't change? 
                        # Wait. If was_lit_sat and is_lit_sat, count unchanged.
                        # If not was_lit_sat and not is_lit_sat, count unchanged.
                        
                    else:
                        # Not satisfied after flip
                        was_lit_sat = (old_val == s_v_in_cj)
                        is_lit_sat = False # new_val != s_v_in_cj
                        if was_lit_sat:
                            sat_counts[cj] -= 1
                            if sat_counts[cj] == 0:
                                unsat_clauses.add(cj)
                else:
                    # Was unsatisfied (sat_counts[cj] == 0)
                    # Only one way to change: literal becomes satisfied
                    is_lit_sat = (new_val == s_v_in_cj)
                    if is_lit_sat:
                        sat_counts[cj] += 1
                        if sat_counts[cj] == 1:
                            if cj in unsat_clauses:
                                unsat_clauses.remove(cj)
            
            if not unsat_clauses:
                return assignment, True

        return assignment, False

    # Main Solver Loop
    start_time = time.time()
    deadline = start_time + 280.0 # Safety margin
    
    rng = random.Random(42)
    best_assignment = None
    best_unsat = float('inf')
    
    # Initial attempt with phase-saved assignment
    assignment = phase[:]
    
    # Adaptive restarts
    # Start with higher noise, lower steps? Or lower noise, higher steps?
    # Standard WalkSAT: noise ~0.5, steps ~ 100 * n or similar.
    # With 150 vars, 100*150 = 15000 steps.
    # Let's try a schedule.
    
    restart = 0
    while time.time() < deadline:
        # Determine parameters for this restart
        # Noise decreases slightly, steps increase
        noise = max(0.1, 0.5 - restart * 0.01)
        steps = min(50000, 5000 + restart * 1000)
        
        # Initialize assignment
        if restart == 0:
            assignment = phase[:]
        else:
            # Mix of phase and random
            assignment = []
            for i in range(n):
                if rng.random() < 0.2:
                    assignment.append(phase[i])
                else:
                    assignment.append(1 if rng.random() < 0.5 else -1)
            
        assignment, solved = walksat(assignment, steps, noise, rng, deadline)
        
        if solved:
            return assignment
            
        # Update best
        unsat_cnt = get_unsat_count(assignment)
        if unsat_cnt < best_unsat:
            best_unsat = unsat_cnt
            best_assignment = assignment[:]
            
        restart += 1
        
        # Early exit if stuck?
        if restart > 50 and best_unsat > 5:
             # Give it a few more chances
             if restart > 100:
                 break

    if best_assignment is not None:
        return best_assignment
    return phase

# EVOLVE-BLOCK-END
