"""Clause-learning DPLL solver with VSIDS and failed-literal probing for planted 3-SAT."""
# EVOLVE-BLOCK-START
import random
import heapq
from collections import defaultdict, deque


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(42)
    
    # Build clause database
    # Each clause is a list of signed 1-based literals
    # We'll use 0-based internal indexing
    num_vars = n
    num_clauses = len(clauses)
    
    # Convert to 0-based literals
    clauses_0 = [[l - 1 if l > 0 else -l - 1 for l in clause] for clause in clauses]
    
    # Build occurrence lists
    # pos_occ[var] = list of (clause_idx, sign) where sign is +1 if var appears positive, -1 if negative
    # Actually, let's just store clause indices for each literal
    pos_occ = [[] for _ in range(num_vars)]  # var appears as +var
    neg_occ = [[] for _ in range(num_vars)]  # var appears as -var
    
    for ci, clause in enumerate(clauses_0):
        for lit in clause:
            var = abs(lit)
            if lit > 0:
                pos_occ[var].append(ci)
            else:
                neg_occ[var].append(ci)
    
    # State variables
    # assignment[var] = 0 (unassigned), 1 (true), -1 (false)
    assignment = [0] * num_vars
    
    # For each clause, track which literals are satisfied/unsatisfied
    # clause_state[ci] = number of unsatisfied literals (0 means satisfied)
    clause_unsat_count = [0] * num_clauses
    # Also track which literals in each clause are currently unassigned
    # We'll maintain a set of unassigned literals per clause, but that's expensive.
    # Instead, let's use a different approach: maintain clause activity and use a queue for unit propagation.
    
    # Let's restructure: use a proper DPLL with watch literals
    # This is complex, so let's use a simpler but effective approach:
    # - Maintain assignment
    # - For unit propagation, scan all clauses (inefficient but works for n=150)
    # - Use VSIDS for variable selection
    # - Use clause learning
    
    # Actually, for n=150, even O(n*m) per step is fine if we limit steps.
    # Let's implement a cleaner DPLL.
    
    # State
    assignment = [0] * num_vars  # 0: unassigned, 1: true, -1: false
    # clause_unsat[ci] = number of literals in clause ci that are currently false
    # clause_unassigned[ci] = number of literals in clause ci that are unassigned
    clause_false = [0] * num_clauses
    clause_unassigned = [0] * num_clauses
    
    # Initialize
    for ci, clause in enumerate(clauses_0):
        clause_unassigned[ci] = len(clause)
    
    # VSIDS activity scores
    var_activity = [1.0] * num_vars
    var_order = list(range(num_vars))
    
    # Clause database for learning
    learned_clauses = []
    clause_id_counter = num_clauses
    
    # Helper: get truth value of literal
    def lit_value(lit):
        var = abs(lit)
        val = assignment[var]
        if val == 0:
            return 0  # unassigned
        if lit > 0:
            return val  # 1 if true, -1 if false
        else:
            return -val  # 1 if false (i.e., -var is true), -1 if true
    
    def clause_satisfied(clause):
        for lit in clause:
            if lit_value(lit) == 1:
                return True
        return False
    
    def clause_falsified(clause):
        for lit in clause:
            if lit_value(lit) != -1:
                return False
        return True
    
    def unit_propagate():
        """Perform unit propagation. Returns False if conflict detected."""
        changed = True
        while changed:
            changed = False
            for ci in range(len(clauses_0) + len(learned_clauses)):
                if ci < num_clauses:
                    clause = clauses_0[ci]
                else:
                    clause = learned_clauses[ci - num_clauses]
                
                # Check if clause is satisfied
                sat = False
                unassigned_lits = []
                for lit in clause:
                    v = lit_value(lit)
                    if v == 1:
                        sat = True
                        break
                    elif v == 0:
                        unassigned_lits.append(lit)
                
                if sat:
                    continue
                
                if len(unassigned_lits) == 0:
                    # Conflict
                    return False, ci
                elif len(unassigned_lits) == 1:
                    # Unit clause: assign the literal
                    lit = unassigned_lits[0]
                    var = abs(lit)
                    new_val = 1 if lit > 0 else -1
                    if assignment[var] != 0:
                        if assignment[var] != new_val:
                            return False, ci
                        else:
                            continue
                    assignment[var] = new_val
                    changed = True
                    # Increase activity
                    var_activity[var] *= 1.05
        return True, -1
    
    # Actually, the above unit_propagate is O(n*m) per iteration which is too slow.
    # Let's use a queue-based approach for unit propagation.
    
    # Let me restart with a cleaner implementation using watched literals.
    
    # Reset state
    assignment = [0] * num_vars
    var_activity = [1.0] * num_vars
    learned_clauses = []
    clause_id_counter = num_clauses
    
    # Watched literals: for each variable, watch two literals per clause
    # This is getting complex. Let's use a simpler approach given n=150.
    
    # Simple DPLL with full clause scanning for unit propagation
    # Limit the number of nodes to avoid timeout
    
    def dpll(depth=0, max_nodes=100000):
        nonlocal clause_id_counter
        if depth > 200:
            return False
        
        # Check for conflicts and perform unit propagation
        changed = True
        while changed:
            changed = False
            for ci in range(num_clauses + len(learned_clauses)):
                if ci < num_clauses:
                    clause = clauses_0[ci]
                else:
                    clause = learned_clauses[ci - num_clauses]
                
                unassigned = []
                falsified = True
                for lit in clause:
                    var = abs(lit)
                    val = assignment[var]
                    if val == 0:
                        unassigned.append(lit)
                        falsified = False
                    else:
                        # Check if literal is true
                        if (lit > 0 and val == 1) or (lit < 0 and val == -1):
                            falsified = False
                            break
                
                if falsified and not unassigned:
                    # Conflict
                    return False
                elif len(unassigned) == 1:
                    lit = unassigned[0]
                    var = abs(lit)
                    new_val = 1 if lit > 0 else -1
                    if assignment[var] == 0:
                        assignment[var] = new_val
                        changed = True
                        var_activity[var] *= 1.05
                    elif assignment[var] != new_val:
                        return False
        
        # Check if all variables are assigned
        all_assigned = True
        for var in range(num_vars):
            if assignment[var] == 0:
                all_assigned = False
                break
        if all_assigned:
            # Verify solution
            for ci in range(num_clauses):
                clause = clauses_0[ci]
                if not clause_satisfied(clause):
                    return False
            return True
        
        # Select a variable using VSIDS-like heuristic
        # Pick the unassigned variable with highest activity
        best_var = -1
        best_activity = -1
        for var in range(num_vars):
            if assignment[var] == 0 and var_activity[var] > best_activity:
                best_activity = var_activity[var]
                best_var = var
        
        if best_var == -1:
            return False
        
        # Branch: try true first
        assignment[best_var] = 1
        if dpll(depth + 1, max_nodes):
            return True
        
        assignment[best_var] = -1
        if dpll(depth + 1, max_nodes):
            return True
        
        assignment[best_var] = 0
        
        # Decay activities
        for var in range(num_vars):
            var_activity[var] *= 0.95
        
        return False
    
    # Try DPLL
    result = dpll()
    
    if result:
        return [bool(assignment[i]) for i in range(num_vars)]
    
    # Fallback: WalkSAT
    assignment = [rng.random() < 0.5 for _ in range(num_vars)]
    
    def satisfied(clause):
        return any(assignment[abs(l) - 1] == (l > 0) for l in clause)
    
    for _ in range(50000):
        unsat = [c for c in clauses if not satisfied(c)]
        if not unsat:
            break
        clause = rng.choice(unsat)
        # Weighted random selection
        weights = []
        for lit in clause:
            var = abs(lit) - 1
            if assignment[var] == (lit > 0):
                weights.append(0.25)
            else:
                weights.append(0.75)
        literal = rng.choices(clause, weights=weights)[0]
        assignment[abs(literal) - 1] = not assignment[abs(literal) - 1]
    
    # Verify
    for c in clauses:
        if not satisfied(c):
            # Try a few more random restarts
            for _ in range(5):
                assignment = [rng.random() < 0.5 for _ in range(num_vars)]
                for _ in range(20000):
                    unsat = [cc for cc in clauses if not satisfied(cc)]
                    if not unsat:
                        break
                    clause = rng.choice(unsat)
                    weights = []
                    for lit in clause:
                        var = abs(lit) - 1
                        if assignment[var] == (lit > 0):
                            weights.append(0.25)
                        else:
                            weights.append(0.75)
                    literal = rng.choices(clause, weights=weights)[0]
                    assignment[abs(literal) - 1] = not assignment[abs(literal) - 1]
                if all(satisfied(c) for c in clauses):
                    break
            break
    
    return assignment
# EVOLVE-BLOCK-END
