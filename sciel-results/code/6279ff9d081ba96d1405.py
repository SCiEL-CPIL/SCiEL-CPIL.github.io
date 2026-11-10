# EVOLVE-BLOCK-START
import time
import random
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Pre-compute clause structure for speed
    # clause_lits[i] = list of (var_index, sign) where sign is +1 or -1
    # var_clauses[v] = list of clause indices containing variable v
    clause_lits = []
    var_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        lits = []
        for l in c:
            v = abs(l) - 1
            s = 1 if l > 0 else -1
            lits.append((v, s))
            var_clauses[v].append(i)
        clause_lits.append(lits)
    
    # Count occurrences of each literal in each clause? No, just variable frequencies for weighting
    var_freq = [0] * n
    for c in clauses:
        seen_vars = set()
        for l in c:
            v = abs(l) - 1
            if v not in seen_vars:
                var_freq[v] += 1
                seen_vars.add(v)
    
    # Normalize weights
    max_freq = max(var_freq) if var_freq else 1
    var_weights = [1.0 + 0.5 * (f / max_freq) for f in var_freq]
    
    def check_satisfied(assignment):
        for c in clause_lits:
            sat = False
            for v, s in c:
                if assignment[v] == (s == 1):
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    def count_unsat(assignment):
        count = 0
        for c in clause_lits:
            sat = False
            for v, s in c:
                if assignment[v] == (s == 1):
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    # State tracking
    # clause_sat_count[i] = number of literals satisfied in clause i
    # assignment[v] = True/False
    # We maintain clause_sat_count incrementally
    
    def init_state(assignment):
        clause_sat = [0] * m
        for i, c in enumerate(clause_lits):
            for v, s in c:
                if assignment[v] == (s == 1):
                    clause_sat[i] += 1
        return clause_sat
    
    def get_unsat_clauses(clause_sat):
        return [i for i in range(m) if clause_sat[i] == 0]
    
    # Weighted clause selection
    # Weight of a clause = product of variable weights of its literals? Or sum?
    # Let's use sum of variable weights for the clause
    clause_weights = [0.0] * m
    for i, c in enumerate(clause_lits):
        w = 0.0
        for v, s in c:
            w += var_weights[v]
        clause_weights[i] = w
    
    def pick_clause(unsat_indices, rng):
        # Weighted random selection
        # To be fast, we can just pick randomly if list is small, else weighted
        if len(unsat_indices) == 1:
            return unsat_indices[0]
        # Weighted choice
        # Sum of weights
        total_w = 0.0
        weights = []
        for idx in unsat_indices:
            w = clause_weights[idx]
            weights.append(w)
            total_w += w
        r = rng.random() * total_w
        cum = 0.0
        for idx, w in zip(unsat_indices, weights):
            cum += w
            if r <= cum:
                return idx
        return unsat_indices[-1]
    
    def pick_variable(clause_idx, clause_sat, assignment, rng):
        # Min-conflicts style: pick variable that satisfies the clause and breaks fewest other unsat clauses?
        # Or WalkSAT style: random, or biased towards satisfying.
        # Let's use a hybrid: 
        # 1. If there's a literal in the clause that is currently false (and flipping it satisfies this clause), 
        #    pick among those.
        # 2. Among those, prefer variables that appear in fewer other unsat clauses? 
        #    Or just random.
        #    Actually, a common heuristic is to pick the variable that satisfies the most other clauses (greedy).
        #    But that's expensive.
        #    Let's try: pick a random literal from the clause that is currently false.
        #    If all are true, the clause is satisfied, so this shouldn't happen.
        
        # Get literals in this clause
        c = clause_lits[clause_idx]
        # Find false literals
        false_lits = []
        for v, s in c:
            if assignment[v] != (s == 1):
                false_lits.append((v, s))
        
        if not false_lits:
            # Clause is satisfied? Shouldn't be in unsat list.
            return None
        
        # Score each candidate variable
        # Score = number of other unsat clauses that would become satisfied by flipping this variable?
        # That requires checking all clauses containing v.
        # Let's do a lighter heuristic: 
        # Score = - (number of other unsat clauses containing v) ? 
        # i.e., prefer variables that are in fewer unsat clauses (less disruptive).
        # But we need to know which clauses are unsat.
        
        # Let's stick to a simpler, effective heuristic:
        # Pick a random false literal.
        # To add intelligence, we can bias towards variables with lower "degree" in unsat set.
        
        # For now, let's just pick randomly from false_lits.
        # Later we can enhance with "escape" moves.
        v, s = rng.choice(false_lits)
        return v
    
    def flip(assignment, clause_sat, v):
        # Flip variable v
        new_val = not assignment[v]
        assignment[v] = new_val
        
        # Update clause_sat for all clauses containing v
        for c_idx in var_clauses[v]:
            c = clause_lits[c_idx]
            # Determine old contribution and new contribution
            # Old: was it satisfied?
            old_sat = 0
            new_sat = 0
            for vv, ss in c:
                if vv == v:
                    # This is the variable we flipped
                    if new_val == (ss == 1):
                        new_sat = 1
                    # old_sat was 0 because we flipped it, so it was unsatisfied before?
                    # Wait, we need to know the state BEFORE flip.
                    # assignment[v] is already new_val.
                    # So old_val = not new_val.
                    old_val = not new_val
                    if old_val == (ss == 1):
                        old_sat = 1
                else:
                    if assignment[vv] == (ss == 1):
                        new_sat += 1
                        old_sat += 1 # Others didn't change
            
            # Actually, simpler:
            # old_sat_count = clause_sat[c_idx]
            # new_sat_count = ?
            # The only change is in variable v.
            # If v was satisfied before, and is not now, sat_count decreases by 1.
            # If v was not satisfied before, and is now, sat_count increases by 1.
            
            old_val = not new_val
            was_sat = (old_val == (s_for_v == 1)) # Need s_for_v
            # We need the sign of v in this clause.
            # Find it
            s_v = None
            for vv, ss in c:
                if vv == v:
                    s_v = ss
                    break
            
            was_sat = (old_val == (s_v == 1))
            is_sat = (new_val == (s_v == 1))
            
            if was_sat and not is_sat:
                clause_sat[c_idx] -= 1
            elif not was_sat and is_sat:
                clause_sat[c_idx] += 1
    
    # Main loop
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_assignment = None
    best_unsat = float('inf')
    
    # Restart strategy
    max_restarts = 50
    steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_sat = init_state(assignment)
        
        # Run local search
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            unsat_count = sum(1 for s in clause_sat if s == 0)
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
            
            # Get unsat clauses
            unsat_indices = [i for i in range(m) if clause_sat[i] == 0]
            
            # Pick clause
            c_idx = pick_clause(unsat_indices, rng)
            
            # Pick variable
            v = pick_variable(c_idx, clause_sat, assignment, rng)
            
            if v is not None:
                flip(assignment, clause_sat, v)
            
            # Simulated Annealing escape?
            # Occasionally, accept a worse move?
            # Or just restart more frequently.
            
    # If we didn't find a solution, return the best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        return assignment

# EVOLVE-BLOCK-END
