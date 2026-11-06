import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Preprocessing
    # Convert clauses to 0-indexed literals for internal use, but keep 1-based for logic?
    # Let's use 0-based indices for vars: 0..n-1
    # Literal l: if l > 0, var is l-1, sign is +. If l < 0, var is -l-1, sign is -.
    
    # Build variable to clause mapping
    # var_to_clauses[v] = list of (clause_idx, literal_value_in_clause)
    var_to_clauses = [[] for _ in range(n)]
    processed_clauses = []
    
    for i, clause in enumerate(clauses):
        processed_clauses.append(clause)
        for lit in clause:
            v = abs(lit) - 1
            var_to_clauses[v].append((i, lit))

    m = len(clauses)
    
    # State
    assignment = [False] * n
    clause_sat = [False] * m
    
    # unsat_count[v] = number of unsatisfied clauses containing v
    unsat_count = [0] * n
    
    def is_clause_satisfied(c_idx, assign):
        for lit in processed_clauses[c_idx]:
            v = abs(lit) - 1
            if assign[v] == (lit > 0):
                return True
        return False

    def init_state(assign):
        for c in range(m):
            if is_clause_satisfied(c, assign):
                clause_sat[c] = True
            else:
                clause_sat[c] = False
        
        # Initialize unsat_count
        for v in range(n):
            count = 0
            for c_idx, lit in var_to_clauses[v]:
                if not clause_sat[c_idx]:
                    count += 1
            unsat_count[v] = count

    def flip(v, assign):
        # Flip variable v
        assign[v] = not assign[v]
        
        # Update clause_sat and unsat_count for all clauses containing v
        for c_idx, lit in var_to_clauses[v]:
            was_sat = clause_sat[c_idx]
            # Check if clause is now satisfied
            # Since only v changed, we can check if v satisfies it, or if it was already sat by others
            # If v satisfies it, it's sat.
            # If v doesn't satisfy it, it's sat only if it was sat before AND v wasn't the only satisfying literal?
            # Actually, simpler: just re-evaluate the clause? O(3).
            # Or: 
            # If lit matches assign[v], clause is sat.
            # Else, clause is sat only if it was sat before? No, if v was the only one, it becomes unsat.
            
            # Let's just re-evaluate for correctness, it's cheap (3 lits)
            new_sat = is_clause_satisfied(c_idx, assign)
            
            if was_sat != new_sat:
                clause_sat[c_idx] = new_sat
                # Update unsat_count for all vars in this clause
                for lit2 in processed_clauses[c_idx]:
                    v2 = abs(lit2) - 1
                    if new_sat:
                        unsat_count[v2] -= 1
                    else:
                        unsat_count[v2] += 1

    def get_gain(v, assign):
        # Calculate gain of flipping v
        # Gain = (unsat clauses containing v that become sat) - (sat clauses containing v that become unsat)
        gain = 0
        for c_idx, lit in var_to_clauses[v]:
            # If clause is currently unsat:
            #   If flipping v satisfies it, gain +1
            #   Else gain 0
            # If clause is currently sat:
            #   If flipping v makes it unsat, gain -1
            #   Else gain 0
            
            if not clause_sat[c_idx]:
                # Clause is unsat. Does flipping v satisfy it?
                # v is flipped. New value is not assign[v].
                # Literal lit is satisfied if (not assign[v]) == (lit > 0)
                if (not assign[v]) == (lit > 0):
                    gain += 1
            else:
                # Clause is sat. Does flipping v make it unsat?
                # It is currently sat. If v is currently satisfying it, and no other literal satisfies it, it becomes unsat.
                # Check if v currently satisfies it
                if assign[v] == (lit > 0):
                    # Check if any other literal satisfies it
                    others_sat = False
                    for lit2 in processed_clauses[c_idx]:
                        if lit2 == lit:
                            continue
                        v2 = abs(lit2) - 1
                        if assign[v2] == (lit2 > 0):
                            others_sat = True
                            break
                    if not others_sat:
                        gain -= 1
        return gain

    def solve_walksat(assign, max_steps=5000, rng=None):
        if rng is None:
            rng = random.Random()
            
        init_state(assign)
        
        # Check if already solved
        unsat_clauses = [i for i in range(m) if not clause_sat[i]]
        if not unsat_clauses:
            return assign
        
        steps = 0
        last_improvement = 0
        min_unsat = len(unsat_clauses)
        
        while steps < max_steps:
            unsat_clauses = [i for i in range(m) if not clause_sat[i]]
            if not unsat_clauses:
                return assign
            
            if len(unsat_clauses) < min_unsat:
                min_unsat = len(unsat_clauses)
                last_improvement = steps
            
            # Restart if stuck
            if steps - last_improvement > 200:
                break
            
            # Pick a random unsat clause
            c_idx = rng.choice(unsat_clauses)
            clause = processed_clauses[c_idx]
            
            # Pick variable to flip
            # With prob 0.2, pick random var from clause
            # Else, pick var with max gain
            if rng.random() < 0.2:
                v = abs(rng.choice(clause)) - 1
            else:
                best_v = -1
                best_gain = -100
                for lit in clause:
                    v = abs(lit) - 1
                    g = get_gain(v, assign)
                    if g > best_gain:
                        best_gain = g
                        best_v = v
                v = best_v
            
            flip(v, assign)
            steps += 1
            
        return None # Failed

    # Main loop with restarts
    deadline = time.time() + 280 # 20s margin
    
    best_assign = None
    best_unsat = float('inf')
    
    seed = 0
    while time.time() < deadline:
        rng = random.Random(seed)
        assign = [rng.random() < 0.5 for _ in range(n)]
        
        res = solve_walksat(assign, max_steps=10000, rng=rng)
        
        if res:
            # Verify
            for c in clauses:
                if not any(res[abs(l)-1] == (l>0) for l in c):
                    break
            else:
                return res
        
        seed += 1

    # If all fails, return last best or random
    return [False] * n
