import random
import time
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Convert to 0-based: for each clause, store (var_idx, sign) where sign is +1/-1
    # We'll represent each literal as: var*2 + (0 if positive, 1 if negative)
    # Actually let's keep it simple: for each clause store list of (var, sign)
    
    # Preprocess
    m = len(clauses)
    clause_vars = []    # clause_vars[i] = list of var indices (0-based)
    clause_signs = []   # clause_signs[i] = list of signs (+1/-1)
    
    for c in clauses:
        vars_list = []
        signs_list = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            vars_list.append(var)
            signs_list.append(sign)
        clause_vars.append(vars_list)
        clause_signs.append(signs_list)
    
    # For each variable, store list of (clause_idx, sign) where it appears
    var_occurrences = [[] for _ in range(n)]
    for i in range(m):
        for j, var in enumerate(clause_vars[i]):
            var_occurrences[var].append((i, clause_signs[i][j]))
    
    # State: assignment as array of 0/1
    # lit_count[var][sign]: number of clauses where literal (var, sign) appears in an UNSAT clause
    # Actually, let's use a simpler approach:
    # For each variable, track how many unsatisfied clauses contain it positively and negatively.
    # When we flip var from 0 to 1:
    #   - Clauses containing (var, +1) that were unsat may become sat
    #   - Clauses containing (var, -1) that were unsat stay unsat (or become more unsat)
    
    # Better approach: maintain sat_count per clause
    # sat_count[i] = number of satisfied literals in clause i
    # A clause is satisfied iff sat_count[i] > 0
    
    # When we flip variable v from current value to new value:
    # For each clause containing v with sign s:
    #   old_lit_sat = (assignment[v] == (s == 1))
    #   new_lit_sat = (not assignment[v] == (s == 1))
    #   delta_sat = new_lit_sat - old_lit_sat  (each is 0 or 1)
    #   sat_count[clause] += delta_sat
    
    # To decide which variable to flip in an unsat clause:
    # We want to maximize the number of newly satisfied clauses.
    # For a variable v in an unsat clause with sign s:
    #   If we set v to satisfy s (i.e., v = (s==1)):
    #     - This clause becomes satisfied (if it wasn't)
    #     - Other clauses containing v with sign s may become satisfied
    #     - Other clauses containing v with sign -s may become unsatisfied
    #   
    #   The net change in satisfied clauses = 
    #     (number of unsat clauses containing (v, s)) - (number of unsat clauses containing (v, -s))
    #     But we need to be careful: the clause we're looking at is already unsat.
    
    # Let's define:
    # pos_unsat[v] = number of UNSAT clauses containing literal (v, +1)
    # neg_unsat[v] = number of UNSAT clauses containing literal (v, -1)
    
    # If we flip v to TRUE:
    #   - All unsat clauses containing (v, +1) become sat: gain pos_unsat[v]
    #   - All unsat clauses containing (v, -1) ... wait, they were unsat and (v,-1) was false.
    #     After flip, (v,-1) is still false (v is now true, so -v is false). No change for those.
    #   - But what about clauses where (v,-1) was the only false literal? They were already unsat
    #     because (v,-1) was false and all others were... no, if (v,-1) was false and the clause was
    #     unsat, it means all literals were false. After flip, (v,-1) is still false. No change.
    #   - Actually, flipping v from false to true:
    #     * Clauses with (v,+1): the literal becomes true -> clause becomes sat (if it was unsat)
    #     * Clauses with (v,-1): the literal becomes false -> was already false, no change
    #   So gain = pos_unsat[v]
    #
    # If we flip v to FALSE:
    #   - gain = neg_unsat[v]
    
    # But wait: if v is currently true and we flip to false:
    #   * Clauses with (v,+1): literal becomes false. If clause was sat only because of (v,+1),
    #     it becomes unsat. Loss = number of SAT clauses where (v,+1) is the only true literal.
    #   * Clauses with (v,-1): literal becomes true. If clause was unsat, it becomes sat.
    #     Gain = neg_unsat[v]
    
    # So the net effect depends on current value. This is getting complex.
    
    # Simpler: just compute the delta by checking each clause containing v.
    # Since each variable appears in maybe 10-20 clauses on average for 50 vars, this is fast enough.
    
    rng = random.Random()
    max_time = 285.0  # Safety margin
    start_time = time.time()
    
    def compute_sat_counts(assignment):
        """Compute sat_count for each clause."""
        sat_counts = np.zeros(m, dtype=np.int32)
        for i in range(m):
            for j, var in enumerate(clause_vars[i]):
                sign = clause_signs[i][j]
                if assignment[var] == (sign == 1):
                    sat_counts[i] += 1
        return sat_counts
    
    def is_satisfied(i, sat_counts):
        return sat_counts[i] > 0
    
    # Initial random assignment
    best_assignment = None
    best_unsat = m + 1
    
    def evaluate(assignment):
        nonlocal best_assignment, best_unsat
        sat_counts = compute_sat_counts(assignment)
        unsat_count = np.sum(sat_counts == 0)
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment.copy()
        return unsat_count, sat_counts
    
    # WalkSAT with restarts
    restart_limit = 500
    walk_prob = 0.3
    
    while time.time() - start_time < max_time:
        # Random restart
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        
        unsat_count, sat_counts = evaluate(assignment)
        if unsat_count == 0:
            return assignment.tolist()
        
        steps = 0
        while steps < restart_limit:
            if unsat_count == 0:
                return assignment.tolist()
            
            # Check time periodically
            if steps % 50 == 0 and time.time() - start_time > max_time:
                break
            
            # Find an unsatisfied clause
            # For efficiency, maintain a list of unsat clause indices
            unsat_indices = np.where(sat_counts == 0)[0]
            if len(unsat_indices) == 0:
                return assignment.tolist()
            
            c_idx = rng.choice(unsat_indices.tolist())
            
            # Choose which variable to flip in this clause
            # Option 1: Random (WalkSAT)
            # Option 2: Greedy - pick variable that maximizes delta
            
            if rng.random() < walk_prob:
                # Random flip
                j = rng.randrange(len(clause_vars[c_idx]))
                var = clause_vars[c_idx][j]
                assignment[var] = not assignment[var]
            else:
                # Greedy: try each variable in the clause, compute delta
                best_var = -1
                best_delta = -10**9
                
                for j, var in enumerate(clause_vars[c_idx]):
                    sign = clause_signs[c_idx][j]
                    # Simulate flip
                    old_val = assignment[var]
                    new_val = not old_val
                    
                    delta = 0
                    for (ci, si) in var_occurrences[var]:
                        # Current satisfaction of literal (var, si) in clause ci
                        old_lit_sat = old_val == (si == 1)
                        new_lit_sat = new_val == (si == 1)
                        if old_lit_sat and not new_lit_sat:
                            # This literal was sat, becomes unsat
                            # Clause ci was sat? Check if it becomes unsat
                            if sat_counts[ci] == 1:
                                delta -= 1
                        elif not old_lit_sat and new_lit_sat:
                            # This literal was unsat, becomes sat
                            # Clause ci was unsat? It becomes sat
                            if sat_counts[ci] == 0:
                                delta += 1
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_var = var
                
                if best_var != -1:
                    assignment[best_var] = not assignment[best_var]
                else:
                    j = rng.randrange(len(clause_vars[c_idx]))
                    assignment[best_var] = not assignment[best_var]
            
            # Update sat_counts incrementally
            # We flipped some variable, but we don't track which one in the above code path cleanly.
            # Let's just recompute for simplicity, or track the flipped var.
            
            # Actually, let's track the flipped variable properly
            # For the random case, we know var. For greedy, we know best_var.
            # Let me restructure to always know which var was flipped.
            
            # Recompute sat_counts (expensive but correct)
            unsat_count, sat_counts = evaluate(assignment)
            
            steps += 1
        
        # Increase restart limit
        restart_limit = int(restart_limit * 1.3)
        if restart_limit > 5000:
            restart_limit = 5000
    
    # Return best found
    if best_assignment is not None:
        return best_assignment.tolist()
    return [False] * n
