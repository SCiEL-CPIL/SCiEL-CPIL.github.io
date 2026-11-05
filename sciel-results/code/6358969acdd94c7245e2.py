"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    # Precompute clause data: list of (var_idx, sign) for each clause
    # sign is True if literal is positive (var must be True), False if negative
    clause_data = []
    for c in clauses:
        cd = []
        for l in c:
            v = abs(l) - 1
            s = (l > 0)
            cd.append((v, s))
        clause_data.append(cd)
    
    # Build variable-to-clauses index for fast update
    var_to_clauses = [[] for _ in range(n)]
    for ci, cd in enumerate(clause_data):
        for v, s in cd:
            var_to_clauses[v].append(ci)
    
    # Pre-compute sign lookup for speed: clause_var_sign[ci][v]
    clause_var_sign = [[None]*n for _ in range(num_clauses)]
    for ci, cd in enumerate(clause_data):
        for v, s in cd:
            clause_var_sign[ci][v] = s
    
    # Deadline guard: 300s limit, leave 10s margin
    start_time = time.time()
    deadline = start_time + 285.0  # 5s extra margin for polish phase
    
    def check_time():
        return time.time() < deadline
    
    def init_state(assignment):
        sat_counts = [0] * num_clauses
        for ci, cd in enumerate(clause_data):
            cnt = 0
            for v, s in cd:
                if assignment[v] == s:
                    cnt += 1
            sat_counts[ci] = cnt
        satisfied = sum(1 for c in sat_counts if c > 0)
        return sat_counts, satisfied
    
    def walksat_run_opt(assignment, max_steps, p):
        """
        Optimized WalkSAT with weighted clause selection.
        Returns (solved: bool, assignment, satisfied_count, sat_counts)
        """
        sat_counts, satisfied = init_state(assignment)
        
        if satisfied == num_clauses:
            return True, assignment, satisfied, sat_counts
        
        for step in range(max_steps):
            if not check_time():
                return False, assignment, satisfied, sat_counts
            
            # Find unsat clauses and build weighted selection
            # Weight: harder clauses (fewer satisfied literals) get higher weight
            # For unsat clauses, all have 0 satisfied, so use inverse of clause "density"
            # Simpler: uniform random over unsat, but with a bias toward clauses
            # that have been unsat for longer (approximated by random)
            
            unsat_indices = []
            for i in range(num_clauses):
                if sat_counts[i] == 0:
                    unsat_indices.append(i)
            
            if not unsat_indices:
                return True, assignment, satisfied, sat_counts
            
            # Weighted random selection: prefer clauses with fewer variables
            # that are "close" to being satisfied (but they're all unsat, so 0)
            # Use uniform random for simplicity, but add slight preference for
            # clauses where flipping one variable might satisfy it
            ci = random.choice(unsat_indices)
            cd = clause_data[ci]
            
            # Adaptive p: start higher, decrease over steps
            # For planted instances, higher greediness helps
            effective_p = p * (1.0 - 0.3 * (step / max_steps)) if max_steps > 0 else p
            effective_p = max(0.3, min(0.95, effective_p))
            
            if random.random() < effective_p:
                best_var = -1
                best_cost = float('inf')
                
                for v, s in cd:
                    cost = 0
                    for c_idx in var_to_clauses[v]:
                        if sat_counts[c_idx] > 0:
                            s_c = clause_var_sign[c_idx][v]
                            if s_c is not None:
                                if assignment[v] == s_c:
                                    if sat_counts[c_idx] == 1:
                                        cost += 1
                    
                    total_cost = cost + random.random() * 0.5
                    
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_var = v
                v_to_flip = best_var
            else:
                v_to_flip = random.choice(cd)[0]
            
            old_val = assignment[v_to_flip]
            new_val = not old_val
            assignment[v_to_flip] = new_val
            
            for c_idx in var_to_clauses[v_to_flip]:
                s_c = clause_var_sign[c_idx][v_to_flip]
                if s_c is not None:
                    was_sat = (old_val == s_c)
                    is_sat = (new_val == s_c)
                    
                    if was_sat and not is_sat:
                        sat_counts[c_idx] -= 1
                        if sat_counts[c_idx] == 0:
                            satisfied -= 1
                    elif not was_sat and is_sat:
                        sat_counts[c_idx] += 1
                        if sat_counts[c_idx] == 1:
                            satisfied += 1
            
            if satisfied == num_clauses:
                return True, assignment, satisfied, sat_counts
        
        return False, assignment, satisfied, sat_counts
    
    def polish_phase(assignment, remaining_time):
        """
        Final polish: take best assignment and run focused WalkSAT
        with high greediness to close remaining gaps.
        """
        if remaining_time < 2.0:
            return assignment
        
        sat_counts, satisfied = init_state(assignment)
        if satisfied == num_clauses:
            return assignment
        
        # Run with high greediness and many steps
        max_steps = int(remaining_time * 5000)  # ~5000 steps per second
        max_steps = min(max_steps, 200000)
        
        solved, assignment, satisfied, _ = walksat_run_opt(assignment, max_steps, 0.85)
        
        if solved:
            return assignment
        
        # Try a few more restarts from this good assignment
        for _ in range(3):
            if not check_time():
                break
            # Perturb the good assignment slightly
            perturbed = assignment[:]
            for _ in range(3):
                v = random.randint(0, n-1)
                perturbed[v] = not perturbed[v]
            solved, assignment, satisfied, _ = walksat_run_opt(perturbed, 5000, 0.8)
            if solved:
                return assignment
        
        return assignment
    
    # Main search loop
    best_assignment = None
    best_satisfied = -1
    
    # Strategy for planted 3-SAT:
    # 1. Start with higher greediness (planted instances are easier)
    # 2. Use warm-starts from best assignments
    # 3. Polish phase at the end
    
    p_values = [0.8, 0.7, 0.6, 0.5]
    steps_list = [500, 2000, 10000, 50000]
    
    # Track best assignment for warm-starting
    warm_start = None
    
    for p in p_values:
        for steps in steps_list:
            if not check_time():
                break
            
            # Number of restarts: do at least 1, more if time allows
            num_restarts = 1
            while check_time():
                # Use warm-start if available, else random
                if warm_start is not None and random.random() < 0.3:
                    assignment = warm_start[:]
                    # Perturb slightly
                    for _ in range(2):
                        v = random.randint(0, n-1)
                        assignment[v] = not assignment[v]
                else:
                    assignment = [random.random() < 0.5 for _ in range(n)]
                
                solved, assignment, satisfied, _ = walksat_run_opt(assignment, steps, p)
                
                if satisfied > best_satisfied:
                    best_satisfied = satisfied
                    best_assignment = assignment[:]
                    warm_start = assignment[:]
                
                if solved:
                    return assignment
                
                # If we made good progress, use as warm start
                if satisfied > best_satisfied * 0.9:
                    warm_start = assignment[:]
                
                num_restarts += 1
                # Limit restarts per parameter set to avoid wasting time
                if num_restarts >= 5 and steps > 1000:
                    break
    
    # Polish phase: use remaining time to close the gap
    if best_assignment is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 2.0:
            polished = polish_phase(best_assignment, remaining_time)
            # Verify the polished assignment
            sat_counts, satisfied = init_state(polished)
            if satisfied == num_clauses:
                return polished
            # If polish improved things, use it
            if satisfied > best_satisfied:
                best_assignment = polished
                best_satisfied = satisfied
    
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: random assignment
    return [random.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
