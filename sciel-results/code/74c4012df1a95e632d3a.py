import random
import time
import math

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
        
    def biased_initialization():
        """Initialize with bias toward satisfying more clauses."""
        assignment = [random.random() < 0.5 for _ in range(n)]
        # Try a few random flips to improve initial satisfaction
        counts = [0] * m
        for i, clause in enumerate(clauses):
            cnt = 0
            for lit in clause:
                v = abs(lit) - 1
                if (assignment[v] and lit > 0) or (not assignment[v] and lit < 0):
                    cnt += 1
            counts[i] = cnt
        initial_unsat = sum(1 for c in counts if c == 0)
        
        best_assign = assignment[:]
        best_unsat = initial_unsat
        
        # Try a few local improvements
        for _ in range(50):
            v = random.randrange(n)
            assignment[v] = not assignment[v]
            for (c_idx, sign) in var_clauses[v]:
                lit_true = (assignment[v] if sign == 1 else not assignment[v])
                if lit_true:
                    counts[c_idx] += 1
                else:
                    counts[c_idx] -= 1
            new_unsat = sum(1 for c in counts if c == 0)
            if new_unsat < best_unsat:
                best_unsat = new_unsat
                best_assign = assignment[:]
            else:
                # Revert
                assignment[v] = not assignment[v]
                for (c_idx, sign) in var_clauses[v]:
                    lit_true = (assignment[v] if sign == 1 else not assignment[v])
                    if lit_true:
                        counts[c_idx] += 1
                    else:
                        counts[c_idx] -= 1
        
        return best_assign

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
        counts = initialize_counts(assignment)
        return all(c > 0 for c in counts)

    def simulated_annealing(assignment, counts, unsat_clauses, time_budget):
        """Run simulated annealing with incremental clause tracking."""
        T = 2.0  # Initial temperature
        cooling_rate = 0.9995
        min_T = 0.001
        
        current_unsat = len(unsat_clauses)
        best_unsat = current_unsat
        best_assignment = assignment[:]
        best_counts = counts[:]
        best_unsat_clauses = set(unsat_clauses)
        
        steps = 0
        max_steps = 200000
        
        while time.time() < time_budget and steps < max_steps:
            if not unsat_clauses:
                return assignment, counts, unsat_clauses, True
            
            # Select a random unsat clause
            clause_idx = random.choice(list(unsat_clauses))
            clause = clauses[clause_idx]
            
            # Evaluate each literal in the clause
            best_lit = None
            best_score = -10000
            
            for lit in clause:
                v = abs(lit) - 1
                s = 0
                for (c_idx, sign) in var_clauses[v]:
                    if c_idx == clause_idx:
                        continue
                    if counts[c_idx] == 0:
                        # Unsat clause, flipping v will satisfy it if the literal becomes true
                        # Since clause is unsat, all its literals are false
                        # After flip, this literal in c_idx will be true (since we're flipping the var)
                        s += 1
                    else:
                        if counts[c_idx] == 1:
                            # Check if this literal is the only satisfying one
                            if (assignment[v] and sign == 1) or (not assignment[v] and sign == -1):
                                s -= 1
                if s > best_score:
                    best_score = s
                    best_lit = lit
            
            # Decide which literal to flip
            if random.random() < 0.3:  # Random walk probability
                lit = random.choice(clause)
            else:
                lit = best_lit
            
            v = abs(lit) - 1
            old_val = assignment[v]
            assignment[v] = not assignment[v]
            
            # Calculate delta in unsatisfied clauses
            delta_unsat = 0
            for (c_idx, sign) in var_clauses[v]:
                lit_true_after = (assignment[v] if sign == 1 else not assignment[v])
                if lit_true_after:
                    old_count = counts[c_idx]
                    counts[c_idx] = old_count + 1
                    if old_count == 0:
                        unsat_clauses.discard(c_idx)
                        delta_unsat -= 1
                else:
                    old_count = counts[c_idx]
                    counts[c_idx] = old_count - 1
                    if old_count == 1:
                        unsat_clauses.add(c_idx)
                        delta_unsat += 1
            
            current_unsat += delta_unsat
            
            # Accept or reject based on simulated annealing
            if delta_unsat <= 0:
                # Always accept improving or neutral moves
                pass
            else:
                # Reject with probability exp(-delta/T)
                if T > min_T:
                    prob = math.exp(-delta_unsat / T)
                    if random.random() > prob:
                        # Reject: revert
                        assignment[v] = old_val
                        for (c_idx, sign) in var_clauses[v]:
                            lit_true_after = (assignment[v] if sign == 1 else not assignment[v])
                            if lit_true_after:
                                counts[c_idx] += 1
                                if counts[c_idx] == 1:
                                    unsat_clauses.discard(c_idx)
                            else:
                                counts[c_idx] -= 1
                                if counts[c_idx] == 0:
                                    unsat_clauses.add(c_idx)
                        current_unsat -= delta_unsat
            
            # Track best solution
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment[:]
                best_counts = counts[:]
                best_unsat_clauses = set(unsat_clauses)
            
            # Cool down
            T *= cooling_rate
            
            steps += 1
            
            if best_unsat == 0:
                return best_assignment, best_counts, best_unsat_clauses, True
        
        # Return best found
        return best_assignment, best_counts, best_unsat_clauses, best_unsat == 0

    def polish(assignment, counts, unsat_clauses):
        """Final deterministic polish pass."""
        improved = True
        max_polish_steps = 10000
        
        while improved and max_polish_steps > 0:
            improved = False
            max_polish_steps -= 1
            
            if not unsat_clauses:
                return assignment, counts, unsat_clauses, True
            
            # Try flipping each variable to see if it reduces unsat count
            best_v = -1
            best_delta = 1  # Must improve by at least 1
            best_new_unsat = len(unsat_clauses)
            
            for v in range(n):
                # Calculate delta if we flip v
                delta = 0
                for (c_idx, sign) in var_clauses[v]:
                    lit_true_after = (not assignment[v] if sign == 1 else assignment[v])
                    if lit_true_after:
                        if counts[c_idx] == 0:
                            delta -= 1  # Satisfies an unsat clause
                    else:
                        if counts[c_idx] == 1:
                            delta += 1  # Unsatisfies a sat clause
                
                if delta < best_delta:
                    best_delta = delta
                    best_v = v
            
            if best_v >= 0:
                v = best_v
                assignment[v] = not assignment[v]
                for (c_idx, sign) in var_clauses[v]:
                    lit_true_after = (assignment[v] if sign == 1 else not assignment[v])
                    if lit_true_after:
                        old_count = counts[c_idx]
                        counts[c_idx] = old_count + 1
                        if old_count == 0:
                            unsat_clauses.discard(c_idx)
                    else:
                        old_count = counts[c_idx]
                        counts[c_idx] = old_count - 1
                        if old_count == 1:
                            unsat_clauses.add(c_idx)
                improved = True
        
        return assignment, counts, unsat_clauses, len(unsat_clauses) == 0

    # Main search loop with restarts
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Try biased initialization first
    assignment = biased_initialization()
    counts = initialize_counts(assignment)
    unsat_clauses = set(i for i, c in enumerate(counts) if c == 0)
    
    if not unsat_clauses:
        return assignment
    
    # Run simulated annealing
    time_slice = 2.0
    restart_count = 0
    
    while time.time() < deadline:
        # Check if we should restart
        if restart_count > 0 and time.time() - start_time > restart_count * time_slice:
            # Try a new random start
            assignment = random_assignment()
            counts = initialize_counts(assignment)
            unsat_clauses = set(i for i, c in enumerate(counts) if c == 0)
            restart_count += 1
            continue
        
        # Run SA for a time budget
        sa_end_time = min(time.time() + 1.0, deadline - 5.0)
        assignment, counts, unsat_clauses, solved = simulated_annealing(
            assignment, counts, unsat_clauses, sa_end_time
        )
        
        if solved:
            return assignment
        
        # Track best
        current_unsat = len(unsat_clauses)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        # If we've been making no progress, restart
        if restart_count >= 5 and time.time() < deadline - 10.0:
            assignment = random_assignment()
            counts = initialize_counts(assignment)
            unsat_clauses = set(i for i, c in enumerate(counts) if c == 0)
            restart_count += 1
    
    # Final polish with best assignment
    if best_assignment is not None:
        counts = initialize_counts(best_assignment)
        unsat_clauses = set(i for i, c in enumerate(counts) if c == 0)
        best_assignment, counts, unsat_clauses, solved = polish(best_assignment, counts, unsat_clauses)
        if solved:
            return best_assignment
    
    # Return best found or current
    if best_assignment is not None:
        return best_assignment
    return assignment
