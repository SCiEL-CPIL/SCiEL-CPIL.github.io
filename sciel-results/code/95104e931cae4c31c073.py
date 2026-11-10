"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math


def solve(instance):
    """
    Portfolio solver: oscillator-based continuous energy minimization + WalkSAT.
    Returns a list of booleans of length n.
    """
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline: 330s wall-clock limit, use 325s as safety margin
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Pre-process clauses for faster evaluation
    processed_clauses = []
    for clause in clauses:
        pos = []
        neg = []
        for l in clause:
            if l > 0:
                pos.append(l - 1)
            else:
                neg.append(-l - 1)
        processed_clauses.append((pos, neg))
    
    num_clauses = len(processed_clauses)
    
    def is_satisfied(assignment, clause_idx):
        pos, neg = processed_clauses[clause_idx]
        for v in pos:
            if assignment[v]:
                return True
        for v in neg:
            if not assignment[v]:
                return True
        return False
    
    def count_unsat(assignment):
        count = 0
        for i in range(num_clauses):
            if not is_satisfied(assignment, i):
                count += 1
        return count
    
    def oscillator_solve(max_time):
        """
        Solve 3-SAT using gradient descent on a continuous energy landscape.
        Variables are phase angles in [-pi/2, pi/2]. Clause energy uses soft-min.
        """
        if time.time() > deadline:
            return None
        
        # Build clause-to-variable mapping for efficient gradient computation
        # For each variable, track which clauses it appears in and with what sign
        var_clauses = [[] for _ in range(n)]  # var_clauses[v] = list of (clause_idx, sign)
        for ci, (pos, neg) in enumerate(processed_clauses):
            for v in pos:
                var_clauses[v].append((ci, 1))
            for v in neg:
                var_clauses[v].append((ci, -1))
        
        # Number of clauses per variable (for normalization)
        var_clause_counts = [len(var_clauses[v]) for v in range(n)]
        max_var_count = max(var_clause_counts) if var_clause_counts else 1
        
        best_assign = None
        best_energy = float('inf')
        
        # Multiple restarts with different initializations
        num_restarts = 20
        for restart in range(num_restarts):
            if time.time() > deadline - max_time:
                break
            
            rng = random.Random(restart * 104729 + 42)
            
            # Initialize phase angles: mix of random and biased toward satisfying
            # Use a soft initialization: angles near 0 for variables with more positive literals
            phases = []
            for v in range(n):
                # Count positive vs negative occurrences
                pos_count = sum(1 for ci, s in var_clauses[v] if s > 0)
                neg_count = sum(1 for ci, s in var_clauses[v] if s < 0)
                bias = (pos_count - neg_count) / max(pos_count + neg_count, 1)
                # Start near bias direction with some noise
                angle = bias * 0.3 + rng.gauss(0, 0.5)
                angle = max(-math.pi/2 + 0.01, min(math.pi/2 - 0.01, angle))
                phases.append(angle)
            
            # Adaptive learning rate schedule
            base_lr = 0.15
            momentum = 0.8
            velocities = [0.0] * n
            
            # Gradient descent with adaptive step size
            max_iters = 2000
            for it in range(max_iters):
                if time.time() > deadline - max_time:
                    break
                
                # Compute energy and gradient
                # Energy: sum over clauses of soft-min of literal satisfaction
                # For a clause with literals l1, l2, l3:
                #   lit_sat(v, sign) = sin(sign * phase[v])  (in [-1,1], 1 means satisfied)
                #   clause_energy = -soft_min(lit_sat values)
                #   We want to MINIMIZE clause_energy (i.e., maximize soft_min)
                
                energy = 0.0
                grad = [0.0] * n
                
                # Compute per-clause contributions
                # Use log-sum-exp for soft-min: soft_min(a,b,c) = -logsumexp(-a,-b,-c)
                # To avoid overflow, use max shift
                for ci in range(num_clauses):
                    pos, neg = processed_clauses[ci]
                    # Compute literal satisfaction values
                    lit_vals = []
                    for v in pos:
                        lit_vals.append(math.sin(phases[v]))
                    for v in neg:
                        lit_vals.append(-math.sin(phases[v]))
                    
                    # soft_min of lit_vals = -logsumexp(-v for v in lit_vals)
                    neg_vals = [-v for v in lit_vals]
                    m = max(neg_vals) if neg_vals else 0
                    if m > 500:
                        m = 500
                    elif m < -500:
                        m = -500
                    exp_sum = sum(math.exp(vi - m) for vi in neg_vals)
                    if exp_sum < 1e-300:
                        exp_sum = 1e-300
                    soft_min_val = -(m + math.log(exp_sum))
                    # Clause energy is -soft_min (we want to minimize this)
                    energy += -soft_min_val
                    
                    # Gradient contribution
                    # d(soft_min)/d(phase[v]) for positive literal v:
                    #   d(-logsumexp(-sin(phase[v]), ...))/d(phase[v])
                    #   = -d(logsumexp)/d(-sin(phase[v])) * d(-sin(phase[v]))/d(phase[v])
                    #   = -(exp(-sin(phase[v])-m)/exp_sum) * (-cos(phase[v]))
                    #   = cos(phase[v]) * exp(-sin(phase[v])-m)/exp_sum
                    # d(energy)/d(phase[v]) = -d(soft_min)/d(phase[v])
                    # For positive literal: grad[v] += -cos(phase[v]) * exp(-sin(phase[v])-m)/exp_sum
                    # For negative literal (lit = -sin(phase[v])):
                    #   d(soft_min)/d(phase[v]) = -(exp(-(-sin(phase[v]))-m)/exp_sum) * (-(-cos(phase[v])))
                    #   = -(exp(sin(phase[v])-m)/exp_sum) * cos(phase[v])
                    #   d(energy)/d(phase[v]) = cos(phase[v]) * exp(sin(phase[v])-m)/exp_sum
                    
                    for v in pos:
                        sv = math.sin(phases[v])
                        cv = math.cos(phases[v])
                        weight = math.exp(-sv - m) / exp_sum
                        grad[v] += -cv * weight
                    
                    for v in neg:
                        sv = math.sin(phases[v])
                        cv = math.cos(phases[v])
                        weight = math.exp(sv - m) / exp_sum
                        grad[v] += cv * weight
                
                # Normalize gradient
                grad_norm = math.sqrt(sum(g*g for g in grad))
                if grad_norm < 1e-10:
                    break
                
                # Update with momentum
                lr = base_lr / (1.0 + 0.1 * it)
                for v in range(n):
                    velocities[v] = momentum * velocities[v] - lr * grad[v] / grad_norm
                    phases[v] += velocities[v]
                    # Clamp to [-pi/2, pi/2]
                    if phases[v] > math.pi/2 - 0.01:
                        phases[v] = math.pi/2 - 0.01
                    elif phases[v] < -math.pi/2 + 0.01:
                        phases[v] = -math.pi/2 + 0.01
                
                # Check if we've converged to a satisfying assignment
                if it % 10 == 0:
                    assign = [phases[v] > 0 for v in range(n)]
                    unsat = count_unsat(assign)
                    if unsat == 0:
                        return assign
                    if unsat < 5:
                        # Track best
                        if energy < best_energy:
                            best_energy = energy
                            best_assign = assign[:]
                
                # Early termination if energy is very low
                if energy < 0.01:
                    assign = [phases[v] > 0 for v in range(n)]
                    if count_unsat(assign) == 0:
                        return assign
                    break
            
            # Final check for this restart
            assign = [phases[v] > 0 for v in range(n)]
            unsat = count_unsat(assign)
            if unsat == 0:
                return assign
            if energy < best_energy:
                best_energy = energy
                best_assign = assign[:]
        
        return best_assign
    
    # Try oscillator solver first with a time budget
    osc_time_budget = min(60.0, (deadline - start_time) * 0.3)
    result = oscillator_solve(osc_time_budget)
    if result is not None:
        if count_unsat(result) == 0:
            return result
    
    # Fall back to WalkSAT
    best_assignment = result if result is not None else None
    best_unsat_count = count_unsat(result) if result is not None else num_clauses + 1
    
    p = 0.5
    max_steps_per_restart = 5000
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        rng = random.Random(restart * 7919 + 13)
        
        # If we have a good partial solution, bias initialization toward it
        if best_assignment is not None and best_unsat_count < num_clauses * 0.3:
            assignment = best_assignment[:]
            # Perturb some variables
            num_perturb = max(1, n // 5)
            for _ in range(num_perturb):
                v = rng.randrange(n)
                assignment[v] = not assignment[v]
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        current_unsat = count_unsat(assignment)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        if current_unsat == 0:
            return assignment
        
        # Walk phase
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break
            
            if current_unsat == 0:
                return assignment
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
            
            # Pick a random unsatisfied clause
            unsat_clause = None
            for _ in range(10):
                ci = rng.randrange(num_clauses)
                if not is_satisfied(assignment, ci):
                    unsat_clause = ci
                    break
            if unsat_clause is None:
                for ci in range(num_clauses):
                    if not is_satisfied(assignment, ci):
                        unsat_clause = ci
                        break
            if unsat_clause is None:
                break
            
            pos, neg = processed_clauses[unsat_clause]
            vars_in_clause = pos + neg
            
            if rng.random() < p:
                v = vars_in_clause[rng.randrange(len(vars_in_clause))]
                assignment[v] = not assignment[v]
            else:
                v = rng.randrange(n)
                assignment[v] = not assignment[v]
            
            new_unsat = count_unsat(assignment)
            
            if new_unsat > current_unsat + 2:
                assignment[v] = not assignment[v]
                new_unsat = current_unsat
            else:
                current_unsat = new_unsat
        
        if time.time() > deadline:
            break
    
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
