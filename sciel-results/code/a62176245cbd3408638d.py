"""Planted 3-SAT solver with incremental local search (WalkSAT)."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    """
    Solve a planted satisfiable 3-CNF instance with 50 variables.
    Returns a list of booleans (assignment) that satisfies all clauses.
    Strategy:
      1. Build variable-clause incidence lists for fast updates.
      2. Initialize assignment using a heuristic based on literal frequency.
      3. Run an incremental WalkSAT local search:
         - Maintain clause satisfaction counts.
         - Flip variables based on a weighted heuristic (random vs. min-damage).
         - Restart with perturbation if stuck.
    """
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: 330s wall-clock limit, use 325s as safety margin
    deadline = time.time() + 325
    
    rng = random.Random(42)
    
    # Precompute incidence lists:
    # var_clauses[v] = list of (clause_idx, literal_in_clause)
    var_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses):
        for l in c:
            v = abs(l) - 1
            var_clauses[v].append((ci, l))
            
    m = len(clauses)
    
    def count_unsat(assignment, clause_sats):
        return sum(1 for s in clause_sats if s == 0)

    def compute_clause_sats(assignment):
        sats = [0] * m
        for ci, c in enumerate(clauses):
            s = 0
            for l in c:
                if assignment[abs(l) - 1] == (l > 0):
                    s += 1
            sats[ci] = s
        return sats

    # --- Initialization ---
    # Heuristic: Assign variable v to True if +v appears more often than -v
    lit_freq = {i: 0 for i in range(1, 2*n+1)}
    lit_freq_neg = {i: 0 for i in range(1, 2*n+1)}
    # Simpler: just count signed literals
    counts = [0] * (2*n)
    for c in clauses:
        for l in c:
            if l > 0:
                counts[l-1] += 1
            else:
                counts[n + abs(l) - 1] += 1
                
    def init_by_bias():
        assign = [False] * n
        for v in range(n):
            pos_freq = counts[v]
            neg_freq = counts[n + v]
            assign[v] = (pos_freq >= neg_freq)
        return assign
        
    def init_random():
        return [rng.random() < 0.5 for _ in range(n)]

    # Try a few initializations to find a good starting point
    candidates = [init_by_bias()]
    for _ in range(3):
        candidates.append(init_random())
        
    best_init = None
    best_init_unsat = float('inf')
    for cand in candidates:
        sats = compute_clause_sats(cand)
        unsat = count_unsat(cand, sats)
        if unsat < best_init_unsat:
            best_init_unsat = unsat
            best_init = cand[:]
            
    assignment = best_init
    clause_sats = compute_clause_sats(assignment)
    unsat_clauses = [i for i, s in enumerate(clause_sats) if s == 0]
    
    def walk_sat_incremental(assignment, clause_sats, unsat_clauses, max_steps, deadline, rng, walk_prob=0.5):
        """
        Incremental WalkSAT.
        Returns (assignment, clause_sats, unsat_clauses, solved)
        """
        if not unsat_clauses:
            return assignment, clause_sats, unsat_clauses, True
            
        steps = 0
        while steps < max_steps:
            if time.time() > deadline:
                break
            if not unsat_clauses:
                return assignment, clause_sats, unsat_clauses, True
                
            # Pick a random unsatisfied clause
            ci = unsat_clauses[rng.randrange(len(unsat_clauses))]
            clause = clauses[ci]
            
            # Choose a literal to flip
            # Heuristic: With prob walk_prob, pick random. Else pick min-damage.
            if rng.random() < walk_prob:
                l = clause[rng.randrange(len(clause))]
            else:
                # Calculate damage for each literal in the clause
                min_damage = float('inf')
                best_l = None
                for l in clause:
                    v = abs(l) - 1
                    damage = 0
                    # Flipping v changes truth value of +v and -v
                    # For each clause containing v or -v, check if it's satisfied
                    # and if it becomes unsatisfied.
                    for cj, lj in var_clauses[v]:
                        if clause_sats[cj] == 1:
                            # This clause is satisfied by exactly one literal.
                            # If that literal is v or -v, and we flip v, it becomes unsat.
                            # We need to check if the satisfying literal is the one affected.
                            # Since clause_sats[cj] == 1, only one literal in cj is True.
                            # If that literal is l_j and l_j has the same variable v, 
                            # we need to know if l_j is currently True.
                            # Actually, simpler: 
                            # Current state: clause_sats[cj] == 1.
                            # If we flip v, the truth value of all literals with var v flips.
                            # So if the unique true literal in cj is one of the literals with var v,
                            # it will become false, and clause_sats[cj] becomes 0.
                            
                            # Check if the unique true literal in cj is associated with v.
                            # We don't store which literal is true, but we can check:
                            # If clause_sats[cj] == 1, exactly one l in clauses[cj] is True.
                            # If l is var v, and it is True, then flipping v makes it False.
                            
                            # Optimization: Precompute is_true? No, just check.
                            # But we are iterating over var_clauses[v].
                            # lj is a literal in cj with var v.
                            # Is lj currently True?
                            if assignment[v] == (lj > 0):
                                damage += 1
                    if damage < min_damage:
                        min_damage = damage
                        best_l = l
                l = best_l
                
            v = abs(l) - 1
            
            # Flip variable v
            assignment[v] = not assignment[v]
            
            # Update clause_sats for all clauses containing v
            # And update unsat_clauses list
            # We need to be careful with the unsat_clauses list modification
            # Strategy: 
            # 1. Identify clauses affected.
            # 2. Update their counts.
            # 3. Rebuild or update the unsat_clauses list efficiently.
            
            # Since n=50 and m is small, we can just update the counts and then 
            # filter the unsat_clauses list, or maintain a set.
            # Given the speed requirement, maintaining a list and rebuilding it 
            # occasionally or updating it incrementally is key.
            
            # Let's update counts first.
            affected_clauses = set()
            for cj, lj in var_clauses[v]:
                affected_clauses.add(cj)
                
            # Update counts
            for cj in affected_clauses:
                # Determine if the literal in cj with var v was True or False before flip
                # We flipped v, so the truth value of that literal inverted.
                # If it was True, it becomes False -> count decreases by 1.
                # If it was False, it becomes True -> count increases by 1.
                
                # Find the literal in cj that has var v
                # There might be multiple? No, standard 3-SAT usually has distinct vars per clause.
                # Even if not, we just check the specific literal lj from var_clauses.
                # Wait, var_clauses[v] contains (ci, l). So for each such (cj, lj):
                # lj is the literal in cj with var v.
                
                # Current assignment[v] is the NEW value.
                # Old value was not assignment[v].
                # If assignment[v] == (lj > 0), then lj is NOW True.
                # So it was False before. Count increases by 1.
                # If assignment[v] != (lj > 0), then lj is NOW False.
                # So it was True before. Count decreases by 1.
                
                if assignment[v] == (lj > 0):
                    clause_sats[cj] += 1
                else:
                    clause_sats[cj] -= 1
                    
            # Update unsat_clauses
            # This is the expensive part if done naively.
            # We can check only the affected clauses.
            # But we also need to remove clauses that became satisfied and add new unsat ones.
            
            # Simple approach: Rebuild unsat_clauses from scratch? O(m).
            # m is usually < 1000 for 50 vars. O(1000) per step is fast.
            # But we can do better.
            
            # Let's try:
            # 1. For each affected clause, check if its status changed regarding unsat/sat.
            # 2. Maintain a boolean array `is_unsat`? Or just filter the list.
            
            # Given the small size, let's just rebuild the list of unsat indices.
            # It's O(m) which is negligible compared to the search time saved.
            unsat_clauses = [i for i, s in enumerate(clause_sats) if s == 0]
            
            steps += 1
            
        return assignment, clause_sats, unsat_clauses, len(unsat_clauses) == 0

    # Main loop with restarts
    best_unsat = count_unsat(assignment, clause_sats)
    best_assignment = assignment[:]
    best_sats = clause_sats[:]
    best_unsat_list = unsat_clauses[:]
    
    max_rounds = 1000 # High number, controlled by time
    steps_per_round = 2000
    
    for round_num in range(max_rounds):
        if time.time() > deadline - 2:
            break
            
        result_assign, result_sats, result_unsat_list, solved = walk_sat_incremental(
            assignment, clause_sats, unsat_clauses, steps_per_round, deadline, rng, walk_prob=0.5
        )
        
        if solved:
            return result_assign
            
        cur_unsat = len(result_unsat_list)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = result_assign[:]
            best_sats = result_sats[:]
            best_unsat_list = result_unsat_list[:]
            
        # Perturb for next round
        # If we are stuck (unsat count not improving for a while), restart harder
        if round_num % 10 == 0:
             # Random restart
             assignment = init_random()
             clause_sats = compute_clause_sats(assignment)
             unsat_clauses = [i for i, s in enumerate(clause_sats) if s == 0]
        else:
             # Small perturbation
             num_flips = rng.randint(1, 3)
             for _ in range(num_flips):
                 v = rng.randrange(n)
                 assignment[v] = not assignment[v]
                 # Update sats for these flips
                 for cj, lj in var_clauses[v]:
                     if assignment[v] == (lj > 0):
                         clause_sats[cj] += 1
                     else:
                         clause_sats[cj] -= 1
             unsat_clauses = [i for i, s in enumerate(clause_sats) if s == 0]
             
    # Final check
    if not best_unsat_list:
        return best_assignment
        
    # If we didn't solve it, try one more long run from best
    if time.time() < deadline - 1:
        result_assign, result_sats, result_unsat_list, solved = walk_sat_incremental(
            best_assignment, best_sats, best_unsat_list, 5000, deadline, rng, walk_prob=0.5
        )
        if solved:
            return result_assign
            
    return best_assignment

# EVOLVE-BLOCK-END
