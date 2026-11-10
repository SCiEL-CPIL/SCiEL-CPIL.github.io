import random
import time
import sys

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline: 300s limit, keep safety margin of 10s
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Pre-compute variable to clause mapping
    # var_clauses[v] = list of (clause_index, sign) where sign is +1 if literal is v, -1 if -v
    var_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_clauses[v].append((i, sign))
            
    def random_assignment():
        return [random.random() < 0.5 for _ in range(n)]
        
    def initialize_counts(assignment):
        counts = [0] * m
        for i, clause in enumerate(clauses):
            cnt = 0
            for lit in clause:
                v = abs(lit) - 1
                if (assignment[v] and lit > 0) or (not assignment[v] and lit < 0):
                    cnt += 1
            counts[i] = cnt
        return counts

    def is_satisfied(assignment):
        # Quick check
        counts = initialize_counts(assignment)
        return all(c > 0 for c in counts)
        
    # WalkSAT with efficient updates and restarts
    best_assignment = random_assignment()
    
    # Parameters
    p_walk = 0.5 # Probability of random flip
    max_steps_per_restart = 10000
    restart_time_slice = 0.5 # Seconds per restart attempt
    
    while time.time() < deadline:
        assignment = random_assignment()
        counts = initialize_counts(assignment)
        unsat_clauses = set(i for i, c in enumerate(counts) if c == 0)
        
        if not unsat_clauses:
            return assignment
            
        steps = 0
        while unsat_clauses and steps < max_steps_per_restart:
            if time.time() > deadline:
                break
            
            # Check time slice for this restart
            if time.time() > start_time + restart_time_slice:
                break
                
            # Select a random unsat clause
            clause_idx = random.choice(list(unsat_clauses))
            clause = clauses[clause_idx]
            
            # Select literal to flip
            # Heuristic: 
            # 1. Calculate score for each literal in the clause
            # Score = number of other unsat clauses that would become satisfied 
            #        minus number of currently satisfied clauses that would become unsat
            # Since the clause is unsat, flipping any literal makes it satisfied.
            # We want to minimize damage to other clauses.
            
            best_lit = None
            best_score = -1000
            random_lit = random.choice(clause)
            
            if random.random() < p_walk:
                # Random walk step
                lit = random_lit
            else:
                # Greedy step: pick literal that minimizes net change in unsat count
                # Or maximizes satisfied clauses
                min_score = float('inf')
                for lit in clause:
                    v = abs(lit) - 1
                    # Simulate flip
                    # For each clause containing v, check if it changes status
                    score = 0
                    for (c_idx, sign) in var_clauses[v]:
                        if c_idx == clause_idx:
                            continue # We know this clause will be satisfied
                        # Check current status of c_idx
                        # We need to know if it's currently sat or unsat
                        # And if flipping v changes that
                        
                        # Current value of lit in c_idx
                        # lit_val = assignment[v] == (sign == 1)
                        # If we flip v, lit_val becomes not lit_val
                        
                        # If c_idx is currently unsat (count 0), and this lit was the only one 
                        # that could satisfy it? No, count is 0, so no lit satisfies it.
                        # Flipping v might make this lit satisfy it.
                        
                        # Let's just count the change in 'satisfied' status for other clauses.
                        # If clause c_idx is currently satisfied (count > 0):
                        #   If it has count == 1, and this lit is the only satisfying one, it becomes unsat.
                        #   If count > 1, it stays satisfied.
                        # If clause c_idx is currently unsat (count == 0):
                        #   If flipping v makes this lit satisfy it, it becomes satisfied.
                        
                        # We need current counts to evaluate this accurately.
                        # This is expensive to calculate for every lit in the clause.
                        
                        pass # This greedy approach is complex to implement efficiently without pre-computed "impact" tables.
                    
                    # Simplified WalkSAT heuristic:
                    # Pick the literal that appears in the most other unsat clauses?
                    # Or just pick the one with highest "weight".
                    
                    # Let's stick to a simpler heuristic for speed:
                    # Pick random literal with prob p, else pick the one that 
                    # satisfies the most *other* unsat clauses.
                    
                    pass

            # To keep it fast, let's use a simpler greedy:
            # Count how many *other* unsat clauses contain this literal.
            # Flipping it will satisfy the chosen clause.
            # It might satisfy other unsat clauses containing this literal.
            # It might unsatisfy other sat clauses containing this literal (if it was the only sat lit).
            
            # Let's implement the score calculation properly but efficiently.
            # We have `counts` and `unsat_clauses`.
            
            candidate_lits = clause
            best_l = candidate_lits[0]
            best_s = -10000
            
            for lit in candidate_lits:
                v = abs(lit) - 1
                s = 0
                for (c_idx, sign) in var_clauses[v]:
                    if c_idx == clause_idx:
                        continue
                    # Determine if flipping v changes c_idx from sat to unsat or unsat to sat
                    # Current assignment of v: assignment[v]
                    # Current value of literal in c_idx: (assignment[v] if sign==1 else not assignment[v])
                    # New value of literal in c_idx: (not assignment[v] if sign==1 else assignment[v])
                    
                    # If c_idx is currently unsat (counts[c_idx] == 0):
                    #   It becomes sat if the new value is True.
                    #   New value is True if (assignment[v] is False and sign==1) or (assignment[v] is True and sign==-1)
                    #   i.e., if the literal is currently False, it becomes True.
                    #   Since counts[c_idx] == 0, all literals in c_idx are currently False.
                    #   So the new value will be True.
                    #   So c_idx becomes satisfied. Score +1.
                    if counts[c_idx] == 0:
                        s += 1
                    else:
                        # c_idx is currently satisfied.
                        # It becomes unsat if this literal was the ONLY satisfying literal (counts[c_idx] == 1)
                        # AND this literal is currently True.
                        # Is this literal currently True?
                        # lit_val = (assignment[v] == (sign == 1))
                        if counts[c_idx] == 1:
                            # Check if this lit is the one making it satisfied
                            # If assignment[v] is True and sign is 1, lit is True.
                            # If assignment[v] is False and sign is -1, lit is True.
                            if (assignment[v] and sign == 1) or (not assignment[v] and sign == -1):
                                s -= 1
                if s > best_s:
                    best_s = s
                    best_l = lit
            
            if random.random() < p_walk:
                lit = random.choice(clause)
            else:
                lit = best_l
            
            # Flip variable
            v = abs(lit) - 1
            assignment[v] = not assignment[v]
            
            # Update counts and unsat_clauses
            for (c_idx, sign) in var_clauses[v]:
                # Determine if the literal's truth value changed
                # Old value: (assignment[v] before flip == (sign == 1))
                # New value: (assignment[v] after flip == (sign == 1))
                # Since we flipped assignment[v], the truth value of the literal flips.
                # If it was True, it's now False. If False, now True.
                
                # We need to know if it was True or False BEFORE the flip.
                # Let's check the current state of the literal in c_idx.
                # Actually, it's easier to check the change in count.
                # If the literal was True, count decreases by 1.
                # If the literal was False, count increases by 1.
                
                # Was the literal True before flip?
                # lit_true_before = (assignment[v] == (sign == 1)) -- WAIT, assignment[v] is already flipped.
                # So lit_true_after = (assignment[v] == (sign == 1))
                # lit_true_before = not lit_true_after
                
                lit_true_after = (assignment[v] if sign == 1 else not assignment[v])
                
                if lit_true_after:
                    # It is now True, so it was False before. Count increases.
                    counts[c_idx] += 1
                    if counts[c_idx] == 1:
                        # Was unsat, now sat
                        unsat_clauses.discard(c_idx)
                else:
                    # It is now False, so it was True before. Count decreases.
                    counts[c_idx] -= 1
                    if counts[c_idx] == 0:
                        # Was sat, now unsat
                        unsat_clauses.add(c_idx)
            
            steps += 1
            
        if not unsat_clauses:
            return assignment
            
        # If we get here, we failed to solve in this restart
        # Update best_assignment if this one is "better" (fewer unsat clauses)
        # For now, just keep the last one or random.
        
    return best_assignment
