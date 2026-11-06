import time
import random
import sys

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    t_start = time.time()
    # Safety margin: 330s limit, use 300s to be safe
    deadline = t_start + 300 

    # --- Preprocessing ---
    # Convert to 0-based variables
    # clause_vars[ci] = list of (v, sign) where sign is 1 for +v, -1 for -v
    # We use 1 and -1 for signs to make comparison with assignment (0/1) easier?
    # Let's use assignment as 0 (False) and 1 (True).
    # Literal (v, sign) is satisfied if assignment[v] == (1 if sign==1 else 0).
    
    clause_vars = []
    pos_count = [0] * n
    neg_count = [0] * n
    var_clauses = [[] for _ in range(n)] # var_clauses[v] = list of (ci, sign)
    
    for ci, c in enumerate(clauses):
        c_vars = []
        for l in c:
            if l > 0:
                v = l - 1
                sign = 1
                pos_count[v] += 1
            else:
                v = -l - 1
                sign = -1
                neg_count[v] += 1
            c_vars.append((v, sign))
            var_clauses[v].append((ci, sign))
        clause_vars.append(c_vars)
        
    # --- Initialization ---
    rng = random.Random(42)
    
    def init_assignment():
        """
        Initialize with bias towards more frequent sign.
        """
        assign = [0] * n
        for v in range(n):
            if pos_count[v] > neg_count[v]:
                # Bias towards True (1)
                if rng.random() < 0.6:
                    assign[v] = 1
                else:
                    assign[v] = 0
            elif neg_count[v] > pos_count[v]:
                # Bias towards False (0)
                if rng.random() < 0.4:
                    assign[v] = 1
                else:
                    assign[v] = 0
            else:
                assign[v] = 1 if rng.random() < 0.5 else 0
        return assign

    def calc_support(assign):
        """
        Calculate support count for each clause.
        support[ci] = number of satisfied literals in clause ci.
        """
        support = [0] * m
        unsat = 0
        for ci, c_vars in enumerate(clause_vars):
            s = 0
            for v, sign in c_vars:
                # Satisfied if assign[v] is 1 and sign is 1, or assign[v] is 0 and sign is -1
                # i.e., assign[v] == (1 if sign==1 else 0)
                # Or: (assign[v] == 1 and sign == 1) or (assign[v] == 0 and sign == -1)
                # Simplified: assign[v] == (sign + 1) // 2 ? No.
                # If sign=1, need assign=1. If sign=-1, need assign=0.
                # target_val = 1 if sign==1 else 0
                if assign[v] == (1 if sign == 1 else 0):
                    s += 1
            support[ci] = s
            if s == 0:
                unsat += 1
        return support, unsat

    # --- Main Search Loop ---
    
    best_assign = None
    best_unsat = m + 1
    restarts = 0
    max_restarts = 10000 # High number, limited by time
    
    while time.time() < deadline:
        restarts += 1
        
        # Perturb if we have a best solution
        if best_assign is not None and restarts > 5 and rng.random() < 0.2:
            assign = best_assign[:]
            # Flip 3 random variables
            for _ in range(3):
                v = rng.randint(0, n - 1)
                assign[v] = 1 - assign[v]
        else:
            assign = init_assignment()
            
        support, unsat = calc_support(assign)
        
        if unsat < best_unsat:
            best_unsat = unsat
            best_assign = assign[:]
            
        if unsat == 0:
            return assign
        
        # WalkSAT Parameters
        # Adaptive noise: start high, decrease? Or keep constant?
        # Standard WalkSAT uses constant noise.
        # Greedy WalkSAT: pick best flip among the 3 in the random unsat clause.
        
        steps = 1000 + restarts * 10
        # Cap steps to avoid getting stuck in one long run if easy
        if steps > 5000:
            steps = 5000
            
        noise_prob = 0.25
        
        for _ in range(steps):
            if time.time() > deadline - 2:
                break
                
            if unsat == 0:
                return assign
                
            # Pick a random unsatisfied clause
            # Optimization: Instead of scanning all m clauses, we could maintain a list of unsat clauses.
            # But m is small (~100-200). Scanning is O(m).
            # Let's just pick random ci until we find unsat.
            # Since unsat is small, this is fast.
            while True:
                ci = rng.randint(0, m - 1)
                if support[ci] == 0:
                    break
            
            c_vars = clause_vars[ci]
            
            # Decide flip: Noise or Greedy
            if rng.random() < noise_prob:
                # Random flip from the clause
                v, sign = c_vars[rng.randint(0, 2)]
            else:
                # Greedy: pick variable in c_vars with max gain
                best_gain = -100
                best_v = -1
                
                for v, sign in c_vars:
                    gain = 1 # This clause will become satisfied
                    
                    # Check impact on other clauses containing v
                    for cj, sj in var_clauses[v]:
                        # If cj == ci, we already accounted for it? 
                        # No, the loop above calculates gain for flipping v.
                        # For cj == ci:
                        # Current state: support[ci] == 0.
                        # v is in ci.
                        # If we flip v, the literal in ci becomes satisfied.
                        # My gain calculation:
                        # "If assign[v] == target_val_for_sj: ... else: if support[cj]==0: gain += 1"
                        # For cj == ci:
                        # support[ci] == 0.
                        # Is assign[v] == target_val_for_sign?
                        # Since support[ci]==0, NO literal is satisfied. So assign[v] != target_val_for_sign.
                        # So we go to "else" branch.
                        # support[ci] == 0, so gain += 1.
                        # So total gain for ci is 1.
                        # This matches the initialization "gain = 1".
                        # So we can skip cj == ci in the inner loop to save time?
                        # Yes, because we already added 1.
                        if cj == ci:
                            continue
                        
                        # Determine if v currently satisfies cj
                        target_val = 1 if sj == 1 else 0
                        if assign[v] == target_val:
                            # v currently satisfies cj
                            # Flipping v will make it unsatisfied IF support[cj] becomes 0
                            # i.e., if support[cj] == 1
                            if support[cj] == 1:
                                gain -= 1
                        else:
                            # v currently falsifies cj
                            # Flipping v will satisfy cj IF support[cj] becomes 1
                            # i.e., if support[cj] == 0
                            if support[cj] == 0:
                                gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                        
                v = best_v
                
            # Apply Flip
            old_val = assign[v]
            new_val = 1 - old_val
            assign[v] = new_val
            
            # Update Support
            for cj, sj in var_clauses[v]:
                old_support = support[cj]
                # Determine if new_val satisfies the literal
                target_val = 1 if sj == 1 else 0
                if new_val == target_val:
                    new_support = old_support + 1
                else:
                    new_support = old_support - 1
                    
                support[cj] = new_support
                
                # Update unsat count
                if old_support == 0 and new_support == 1:
                    unsat -= 1
                elif old_support == 1 and new_support == 0:
                    unsat += 1
        
        # End of steps for this restart
        if unsat < best_unsat:
            best_unsat = unsat
            best_assign = assign[:]
            
        if unsat == 0:
            return assign

    # Time's up, return best found
    if best_assign is not None:
        return best_assign
    return [0] * n
