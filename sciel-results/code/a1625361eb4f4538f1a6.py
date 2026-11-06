"""Synthetic planted 3-SAT solver (150 variables) - Weighted WalkSAT with restarts."""
import time
import random
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290  # 40s safety margin before 330s kill
    
    # Pre-process: build literal-to-variable index
    # Convert clauses to 0-based with sign encoding
    # For speed, we'll work with the raw clauses
    
    def get_var(lit):
        return abs(lit) - 1
    
    def lit_satisfied(assignment, lit):
        return assignment[abs(lit) - 1] == (lit > 0)
    
    def clause_satisfied(assignment, clause):
        for l in clause:
            if assignment[abs(l) - 1] == (l > 0):
                return True
        return False
    
    # Luby sequence generator
    def luby_sequence():
        # Classic Luby sequence: 1, 2, 1, 4, 1, 2, 1, 8, ...
        def luby(k):
            # Find the largest power of 2 that divides k
            m = k
            while m % 2 == 0:
                m //= 2
            # Find p such that 2^p <= m < 2^(p+1)
            p = 0
            while (1 << (p + 1)) <= m:
                p += 1
            return 1 << p
        
        k = 1
        while True:
            yield luby(k)
            k += 1
    
    luby_iter = luby_sequence()
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    for attempt in range(200):  # Up to 200 restarts
        if time.time() > deadline:
            break
        
        # Get restart budget from Luby sequence
        base_budget = next(luby_iter)
        # Scale: each unit = 500 flips
        max_flips = base_budget * 500
        
        # Different initialization strategies per attempt
        rng = random.Random(42 + attempt)
        
        if attempt % 3 == 0:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        elif attempt % 3 == 1:
            assignment = [True] * n
        else:
            assignment = [False] * n
        
        # Clause weights
        clause_weights = [1.0] * len(clauses)
        
        # Track unsatisfied clauses
        unsat_clauses = []
        for i, c in enumerate(clauses):
            if not clause_satisfied(assignment, c):
                unsat_clauses.append(i)
        
        if not unsat_clauses:
            return assignment
        
        if len(unsat_clauses) < best_unsat_count:
            best_unsat_count = len(unsat_clauses)
            best_assignment = assignment[:]
        
        flips = 0
        p_walk = 0.01  # probability of random walk step
        
        while flips < max_flips:
            if time.time() > deadline:
                if best_assignment is not None:
                    return best_assignment
                return assignment
            
            if not unsat_clauses:
                return assignment
            
            # Select a random unsatisfied clause
            ci = unsat_clauses[rng.randrange(len(unsat_clauses))]
            clause = clauses[ci]
            
            # Score each literal in the clause
            # Score = number of satisfied literals that would become unsatisfied
            #        - number of unsatisfied literals that would become satisfied
            #        + random tie-breaker
            best_score = -float('inf')
            best_var = -1
            best_literal = clause[0]
            
            for lit in clause:
                v = abs(lit) - 1
                current_val = assignment[v]
                # If we flip v, what happens?
                # For this literal: if it's currently unsatisfied, flipping makes it satisfied
                # For other literals in other clauses involving v: they flip too
                
                # Quick score: just check this clause's contribution
                # A more complete score would check all clauses, but that's expensive
                # Use a heuristic: prefer literals that are currently unsatisfied
                # and have high clause weight
                
                # Compute conflict delta for this variable
                # This is expensive; use a simpler heuristic
                score = 0.0
                # Prefer flipping a variable that makes this literal true
                if not (current_val == (lit > 0)):
                    score += 1.0  # This literal would become satisfied
                
                # Add random noise for tie-breaking
                score += rng.random() * 0.5
                
                if score > best_score:
                    best_score = score
                    best_var = v
                    best_literal = lit
            
            # With probability p_walk, do a random walk step
            if rng.random() < p_walk:
                lit = clause[rng.randrange(len(clause))]
                best_var = abs(lit) - 1
                best_literal = lit
            
            # Flip the variable
            assignment[best_var] = not assignment[best_var]
            
            # Update clause weights
            clause_weights[ci] *= 1.01
            
            # Update unsat_clauses: need to check clauses involving best_var
            # For efficiency, rebuild unsat_clauses periodically
            # For now, do a full check every 100 flips or when needed
            
            if flips % 100 == 0 or flips == max_flips - 1:
                new_unsat = []
                for i, c in enumerate(clauses):
                    if not clause_satisfied(assignment, c):
                        new_unsat.append(i)
                unsat_clauses = new_unsat
                
                if not unsat_clauses:
                    return assignment
                
                if len(unsat_clauses) < best_unsat_count:
                    best_unsat_count = len(unsat_clauses)
                    best_assignment = assignment[:]
            
            flips += 1
        
        # End of this restart
        # Decay clause weights
        for i in range(len(clause_weights)):
            clause_weights[i] = max(1.0, clause_weights[i] * 0.5)
        
        # If we've done enough work, return best
        if time.time() > deadline - 10:
            break
    
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: return last assignment
    return assignment


# Pre-compile for speed
def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Build index: for each variable, list of (clause_index, literal)
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_to_clauses[abs(lit) - 1].append((ci, lit))
    
    def is_satisfied(assignment, clause):
        for lit in clause:
            if assignment[abs(lit) - 1] == (lit > 0):
                return True
        return False
    
    def count_unsat(assignment):
        cnt = 0
        for c in clauses:
            sat = False
            for lit in c:
                if assignment[abs(lit) - 1] == (lit > 0):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt
    
    # Luby sequence
    def luby(k):
        m = k
        while m % 2 == 0:
            m //= 2
        p = 0
        while (1 << (p + 1)) <= m:
            p += 1
        return 1 << p
    
    best_assignment = None
    best_unsat = float('inf')
    
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        budget = luby(attempt + 1) * 200  # Scale factor
        rng = random.Random(12345 + attempt * 7919)
        
        # Diverse initializations
        strat = attempt % 4
        if strat == 0:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        elif strat == 1:
            assignment = [True] * n
        elif strat == 2:
            assignment = [False] * n
        else:
            # Biased: more likely true (planted instances often have biased solution)
            assignment = [rng.random() < 0.7 for _ in range(n)]
        
        unsat_count = count_unsat(assignment)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
        
        # Clause weights for heuristic
        cw = [1.0] * len(clauses)
        
        # Build unsat set
        unsat_set = set()
        for ci, c in enumerate(clauses):
            sat = False
            for lit in c:
                if assignment[abs(lit) - 1] == (lit > 0):
                    sat = True
                    break
            if not sat:
                unsat_set.add(ci)
        
        p_walk = 0.05
        flips = 0
        
        while flips < budget:
            if time.time() > deadline:
                break
            
            if not unsat_set:
                return assignment
            
            # Pick random unsat clause
            ci = next(iter(unsat_set))  # This is O(1) but not uniform
            # Better: pick uniformly
            unsat_list = list(unsat_set)
            ci = unsat_list[rng.randrange(len(unsat_list))]
            clause = clauses[ci]
            
            # Score literals in this clause
            # For each literal, compute the change in unsatisfied count if we flip its variable
            best_score = -1e18
            best_var = -1
            
            for lit in clause:
                v = abs(lit) - 1
                cur = assignment[v]
                # If we flip v, this literal's satisfaction flips
                # And all other literals involving v flip
                
                # Estimate: how many clauses containing v are currently unsat?
                # And how many would become sat/unsat?
                
                # Quick heuristic: 
                # - If this literal is currently unsatisfied, flipping helps this clause
                # - Count how many other clauses with v would be hurt
                
                score = 0.0
                lit_satisfied_now = (cur == (lit > 0))
                
                # Direct effect on this clause
                if not lit_satisfied_now:
                    score += cw[ci]  # Helps this unsat clause
                
                # Check other clauses involving v
                for (oci, olit) in var_to_clauses[v]:
                    if oci == ci:
                        continue
                    # Would this clause's satisfaction change?
                    # The literal olit's value flips
                    olit_satisfied_now = (cur == (olit > 0))
                    
                    # Check if this clause is currently satisfied by other literals
                    clause_o = clauses[oci]
                    satisfied_by_other = False
                    for ll in clause_o:
                        if abs(ll) - 1 == v:
                            continue
                        if assignment[abs(ll) - 1] == (ll > 0):
                            satisfied_by_other = True
                            break
                    
                    if olit_satisfied_now and not satisfied_by_other:
                        # Flipping v would make this clause unsat
                        score -= cw[oci]
                    elif not olit_satisfied_now and satisfied_by_other:
                        # Flipping v would make this clause sat (if it was unsat)
                        score += cw[oci]
                    elif not olit_satisfied_now and not satisfied_by_other:
                        # Clause was unsat, remains unsat (olit was the only hope, but flipping doesn't help if others don't satisfy)
                        # Actually if olit not satisfied and no other satisfied, clause is unsat.
                        # After flip, olit becomes satisfied, so clause becomes sat
                        score += cw[oci]
                
                # Add small random for tie-breaking
                score += rng.random() * 0.1
                
                if score > best_score:
                    best_score = score
                    best_var = v
            
            # Random walk with probability p_walk
            if rng.random() < p_walk:
                lit = clause[rng.randrange(len(clause))]
                best_var = abs(lit) - 1
            
            if best_var < 0:
                best_var = 0
            
            # Flip
            assignment[best_var] = not assignment[best_var]
            
            # Update unsat_set incrementally
            # Clauses involving best_var may change status
            for (oci, olit) in var_to_clauses[best_var]:
                clause_o = clauses[oci]
                sat = False
                for ll in clause_o:
                    if assignment[abs(ll) - 1] == (ll > 0):
                        sat = True
                        break
                if sat:
                    unsat_set.discard(oci)
                else:
                    unsat_set.add(oci)
            
            # Update clause weights
            cw[ci] *= 1.005
            # Decay all weights slightly
            # (expensive, skip for now)
            
            flips += 1
            
            # Periodic full check
            if flips % 500 == 0:
                uc = count_unsat(assignment)
                if uc == 0:
                    return assignment
                if uc < best_unsat:
                    best_unsat = uc
                    best_assignment = assignment[:]
                # Rebuild unsat_set
                unsat_set = set()
                for ci2, c in enumerate(clauses):
                    sat = False
                    for lit2 in c:
                        if assignment[abs(lit2) - 1] == (lit2 > 0):
                            sat = True
                            break
                    if not sat:
                        unsat_set.add(ci2)
        
        # End restart
        if time.time() > deadline - 5:
            break
    
    if best_assignment is not None:
        return best_assignment
    return assignment
