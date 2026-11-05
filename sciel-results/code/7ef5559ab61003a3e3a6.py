import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(42)
    
    # Pre-compute variable to clause indices mapping
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            var_to_clauses[abs(l)-1].append(i)
            
    # Initialize
    assignment = [rng.random() < 0.5 for _ in range(n)]
    phase = assignment[:]
    
    # Clause status: 0 for unsat, 1 for sat? Or count of true literals?
    # Count of true literals is better for quick checks.
    clause_true_count = [0] * len(clauses)
    unsat_clauses = []
    
    def update_clause_status(ci):
        count = 0
        for l in clauses[ci]:
            if assignment[abs(l)-1] == (l > 0):
                count += 1
        clause_true_count[ci] = count
        if count == 0:
            # Add to unsat if not already there
            # Using a set for O(1) lookup? Or list and linear search?
            # Set is better.
            pass # Handled by global set
            
    # Let's use a set for unsat clause indices
    unsat_set = set()
    
    # Initial setup
    for i, c in enumerate(clauses):
        count = 0
        for l in c:
            if assignment[abs(l)-1] == (l > 0):
                count += 1
        clause_true_count[i] = count
        if count == 0:
            unsat_set.add(i)
            
    # Weights
    weights = [1.0] * len(clauses)
    global_decay = 1.0
    decay_rate = 0.99
    inc_amount = 1.0
    
    def flip_var(v_idx, new_val):
        nonlocal global_decay
        if assignment[v_idx] == new_val:
            return
        assignment[v_idx] = new_val
        phase[v_idx] = new_val
        
        # Update clauses containing v_idx
        for ci in var_to_clauses[v_idx]:
            # Determine change in true count
            # Old value: assignment[v_idx] (before flip, but we already flipped it? No, check before)
            # Let's store old_val before flip
            pass # Need to restructure
            
    # Restructure flip logic
    def do_flip(v_idx, new_val):
        old_val = assignment[v_idx]
        if old_val == new_val:
            return
        
        assignment[v_idx] = new_val
        phase[v_idx] = new_val
        
        for ci in var_to_clauses[v_idx]:
            # Check if the literal in this clause for v_idx changed truth value
            literal_found = False
            for l in clauses[ci]:
                if abs(l) - 1 == v_idx:
                    literal_found = True
                    old_truth = (old_val == (l > 0))
                    new_truth = (new_val == (l > 0))
                    if old_truth and not new_truth:
                        # Lost a true literal
                        clause_true_count[ci] -= 1
                        if clause_true_count[ci] == 0:
                            unsat_set.add(ci)
                    elif not old_truth and new_truth:
                        # Gained a true literal
                        clause_true_count[ci] += 1
                        if ci in unsat_set:
                            unsat_set.remove(ci)
                    break # Found the literal
            
    # Greedy flip selection
    def get_greedy_flip(clause):
        best_cost = float('inf')
        candidates = []
        
        for l in clause:
            v_idx = abs(l) - 1
            target_val = (l > 0)
            
            # If we flip v_idx to target_val, this clause is satisfied.
            # Cost: number of other clauses containing v_idx that are currently satisfied 
            # and will become unsatisfied.
            
            cost = 0
            for ci in var_to_clauses[v_idx]:
                # If this is the current clause, it will become satisfied, so it doesn't count as "newly unsat"
                # But if it was already satisfied? No, we picked it from unsat_set.
                # So we skip it? 
                # Actually, if we flip, it becomes satisfied. So it moves from unsat to sat.
                # We only care about clauses that are SAT and become UNSAT.
                
                # Check if clause ci is currently satisfied
                if clause_true_count[ci] > 0:
                    # Check if it is satisfied ONLY by v_idx's literal
                    # And if that literal will become false.
                    
                    # Find literal for v_idx in ci
                    lit_for_v = None
                    for ll in clauses[ci]:
                        if abs(ll) - 1 == v_idx:
                            lit_for_v = ll
                            break
                            
                    if lit_for_v is not None:
                        # Will this literal become false?
                        # Current truth: assignment[v_idx] == (lit_for_v > 0)
                        # New truth: target_val == (lit_for_v > 0)
                        
                        # If current is True and new is False, and count is 1, then it becomes unsat.
                        if assignment[v_idx] == (lit_for_v > 0) and target_val != (lit_for_v > 0):
                            if clause_true_count[ci] == 1:
                                cost += 1
                                
            if cost < best_cost:
                best_cost = cost
                candidates = [(v_idx, target_val)]
            elif cost == best_cost:
                candidates.append((v_idx, target_val))
                
        if candidates:
            return rng.choice(candidates)
        return None

    start_time = time.time()
    deadline = start_time + 250.0
    
    steps_since_restart = 0
    restart_limit = 1000
    total_steps = 0
    
    while time.time() < deadline:
        if not unsat_set:
            return assignment
            
        if steps_since_restart >= restart_limit:
            # Reset to phase
            # We need to recalculate clause_true_count and unsat_set based on phase
            # This is O(N_c * 3)
            assignment = phase[:]
            for i in range(len(clauses)):
                count = 0
                for l in clauses[i]:
                    if assignment[abs(l)-1] == (l > 0):
                        count += 1
                clause_true_count[i] = count
            unsat_set = set()
            for i in range(len(clauses)):
                if clause_true_count[i] == 0:
                    unsat_set.add(i)
                    
            restart_limit = int(restart_limit * 1.1)
            steps_since_restart = 0
            continue # Re-check unsat_set
            
        # Select clause
        # Weighted random
        # Effective weight = weights[i] * global_decay
        # To avoid multiplying all, we can just pick random from unsat_set 
        # and bias towards higher weights?
        # Simple approach: Pick random unsat clause. 
        # Or: Create list of (eff_weight, idx) and pick.
        # For speed, if unsat_set is small, just list it.
        
        # Let's use a simple heuristic: 
        # 50% random, 50% "most weighted" (approx)
        # Or just random. The greedy flip does the heavy lifting.
        
        clause_idx = rng.choice(list(unsat_set))
        clause = clauses[clause_idx]
        
        # Update weights
        global_decay *= decay_rate
        weights[clause_idx] += inc_amount
        
        # Select flip
        # 80% Greedy, 20% Random
        if rng.random() < 0.8:
            res = get_greedy_flip(clause)
            if res:
                v_idx, val = res
            else:
                l = rng.choice(clause)
                v_idx = abs(l) - 1
                val = (l > 0)
        else:
            l = rng.choice(clause)
            v_idx = abs(l) - 1
            val = (l > 0)
            
        do_flip(v_idx, val)
        
        steps_since_restart += 1
        total_steps += 1
        
    return assignment
