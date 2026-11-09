import time
import random
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    start_time = time.time()
    # Safety margin: 295s to avoid hitting 330s hard limit
    deadline = start_time + 295.0
    
    # Precomputation
    # Convert clauses to 0-based indices and signs
    # clause_data[i] = [(var_idx, is_positive), ...]
    clause_data = []
    var_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        cl_store = []
        for l in clause:
            v = abs(l) - 1
            sign = 1 if l > 0 else -1
            cl_store.append((v, sign))
            var_clauses[v].append(i)
        clause_data.append(cl_store)
    
    m = len(clauses)
    
    # Helper to check if a clause is satisfied given assignment
    def is_satisfied(assignment, clause_idx):
        for v, sign in clause_data[clause_idx]:
            if assignment[v] == (sign == 1):
                return True
        return False

    def get_unsat_clauses(assignment):
        return [i for i in range(m) if not is_satisfied(assignment, i)]

    def count_unsat(assignment, unsat_list=None):
        if unsat_list is not None:
            return len(unsat_list)
        return len(get_unsat_clauses(assignment))

    def dpll_limited(rng, depth_limit=10):
        """
        Lightweight DPLL with unit propagation.
        Returns assignment if solved, else None.
        """
        def propagate(assignment, clauses_remaining):
            """
            assignment: list of bool or None
            clauses_remaining: list of clause indices
            Returns (new_assignment, new_clauses, contradiction)
            """
            new_assign = assignment[:]
            new_clauses = list(clauses_remaining)
            changed = True
            while changed:
                changed = False
                next_clauses = []
                for ci in new_clauses:
                    sat = False
                    unassigned = []
                    for v, sign in clause_data[ci]:
                        if new_assign[v] is not None:
                            if new_assign[v] == (sign == 1):
                                sat = True
                                break
                        else:
                            unassigned.append((v, sign))
                    
                    if sat:
                        continue
                    if not unassigned:
                        return None, None, True
                    if len(unassigned) == 1:
                        v, sign = unassigned[0]
                        new_assign[v] = (sign == 1)
                        changed = True
                    else:
                        next_clauses.append(ci)
                new_clauses = next_clauses
            
            return new_assign, new_clauses, False

        def dpll_rec(assignment, clauses_remaining, depth):
            if depth > depth_limit:
                return None
            
            if not clauses_remaining:
                return assignment
            
            new_assign, rem_clauses, contra = propagate(assignment, clauses_remaining)
            if contra:
                return None
            
            if not rem_clauses:
                return new_assign
            
            var_counts = {}
            for ci in rem_clauses:
                for v, _ in clause_data[ci]:
                    if new_assign[v] is None:
                        var_counts[v] = var_counts.get(v, 0) + 1
            
            if not var_counts:
                return None
                
            max_count = max(var_counts.values())
            candidates = [v for v, c in var_counts.items() if c == max_count]
            var = candidates[rng.randint(0, len(candidates) - 1)]
            
            for val in [True, False]:
                trial_assign = new_assign[:]
                trial_assign[var] = val
                
                filtered = []
                for ci in rem_clauses:
                    sat = False
                    for v, sign in clause_data[ci]:
                        if trial_assign[v] is not None:
                            if trial_assign[v] == (sign == 1):
                                sat = True
                                break
                    if not sat:
                        filtered.append(ci)
                        
                res = dpll_rec(trial_assign, filtered, depth + 1)
                if res is not None:
                    final = []
                    for i in range(n):
                        if res[i] is None:
                            final.append(rng.random() < 0.5)
                        else:
                            final.append(res[i])
                    return final
            return None

        initial = [None] * n
        return dpll_rec(initial, list(range(m)), 0)

    # Strategy 1: Quick DPLL
    rng = random.Random(42)
    sol = dpll_limited(rng, depth_limit=12)
    if sol is not None:
        if count_unsat(sol) == 0:
            return sol

    # Strategy 2: Soft-max WalkSAT with smooth surrogate objective
    # Uses log-sum-exp for smooth variable selection score
    
    def soft_max_score(scores, temperature=1.0):
        """
        Compute smooth max using log-sum-exp for stable gradient.
        Returns the soft-max value and the probabilities.
        """
        if not scores:
            return 0.0, []
        max_s = max(scores)
        if max_s == float('-inf'):
            n_scores = len(scores)
            return 0.0, [1.0/n_scores] * n_scores
        exp_scores = [math.exp((s - max_s) / temperature) for s in scores]
        sum_exp = sum(exp_scores)
        if sum_exp == 0:
            n_scores = len(scores)
            return 0.0, [1.0/n_scores] * n_scores
        probs = [e / sum_exp for e in exp_scores]
        # Soft-max value is log(sum(exp(s/T))) + max_s
        soft_max_val = math.log(sum_exp) + max_s
        return soft_max_val, probs

    def weighted_walksat_smooth(rng, max_steps, noise_prob=0.2, weight_decay=0.95, 
                                  soft_temp=1.0, restart_id=0):
        """
        WalkSAT with clause weighting and smooth surrogate for variable selection.
        Uses log-sum-exp to create a smooth objective for variable choice.
        """
        # Initialize with slight bias based on restart to increase diversity
        assignment = []
        for i in range(n):
            # Vary initialization slightly per restart
            bias = 0.5 + 0.1 * math.sin(restart_id * 0.7 + i * 0.3)
            assignment.append(rng.random() < bias)
        
        unsat_clauses = get_unsat_clauses(assignment)
        
        if not unsat_clauses:
            return assignment
        
        weights = [1.0] * m
        
        best_unsat = len(unsat_clauses)
        best_assignment = assignment.copy()
        
        # Track soft objective for potential plateau detection
        prev_soft_obj = None
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if not unsat_clauses:
                return assignment
            
            # Select clause with probability proportional to weight (smoothed)
            # Use temperature-scaled selection for more exploration
            total_weight = sum(weights[i] for i in unsat_clauses)
            if total_weight == 0:
                total_weight = 1.0
                for i in unsat_clauses:
                    weights[i] = 1.0
            
            # Smoothed clause selection: use soft-max over weights
            unsat_weights = [weights[ci] for ci in unsat_clauses]
            _, clause_probs = soft_max_score(unsat_weights, temperature=soft_temp)
            
            # Sample clause
            r = rng.random()
            cumulative = 0
            selected_clause = unsat_clauses[0]
            for idx, p in enumerate(clause_probs):
                cumulative += p
                if r <= cumulative:
                    selected_clause = unsat_clauses[idx]
                    break
            
            # Increase weight of selected clause
            weights[selected_clause] += 1.0
            
            # Select variable to flip using smooth surrogate
            clause_vars = clause_data[selected_clause]
            
            # Compute smooth score for each variable:
            # Score = number of other unsat clauses that would be satisfied
            # Plus a small term for clause weight (smoothed)
            scores = []
            for v, sign in clause_vars:
                score = 0.0
                for other_ci in unsat_clauses:
                    if other_ci == selected_clause:
                        continue
                    # Check if flipping v satisfies other_ci
                    for v2, sign2 in clause_data[other_ci]:
                        if v2 == v:
                            if (not assignment[v]) == (sign2 == 1):
                                score += 1.0
                            break
                # Add small weight-based bonus for smoothness
                score += 0.1 * weights[selected_clause] / max(1, len(unsat_clauses))
                scores.append(score)
            
            # Use soft-max to get probabilities over variables
            _, var_probs = soft_max_score(scores, temperature=soft_temp)
            
            # Apply noise: with probability noise_prob, pick random var from clause
            if rng.random() < noise_prob:
                v_to_flip = clause_vars[rng.randint(0, 2)][0]
            else:
                # Sample from smooth distribution
                r = rng.random()
                cumulative = 0
                v_to_flip = clause_vars[0][0]
                for idx, p in enumerate(var_probs):
                    cumulative += p
                    if r <= cumulative:
                        v_to_flip = clause_vars[idx][0]
                        break
            
            # Flip
            assignment[v_to_flip] = not assignment[v_to_flip]
            
            # Update unsat clauses and weights
            old_unsat_set = set(unsat_clauses)
            new_unsat_clauses = get_unsat_clauses(assignment)
            new_unsat_set = set(new_unsat_clauses)
            
            # Decay weights for unsat clauses
            for ci in new_unsat_clauses:
                weights[ci] *= weight_decay
            
            # Reset weight for clauses that became satisfied
            for ci in old_unsat_set:
                if ci not in new_unsat_set:
                    weights[ci] = 1.0
            
            # Reset weight for newly unsat clauses
            for ci in new_unsat_set:
                if ci not in old_unsat_set:
                    weights[ci] = 1.0
            
            unsat_clauses = new_unsat_clauses
            
            if len(unsat_clauses) < best_unsat:
                best_unsat = len(unsat_clauses)
                best_assignment = assignment.copy()
                
            if not unsat_clauses:
                return assignment
        
        if best_unsat == 0:
            return best_assignment
        return None

    # Strategy 3: Simulated Annealing polish pass
    # Final high-precision pass with geometric cooling
    
    def simulated_annealing_polish(rng, initial_assignment, max_iterations=20000, 
                                     t_initial=1.0, t_final=0.01, cooling_rate=0.9995):
        """
        Simulated annealing with geometric cooling for final polish.
        Uses the number of unsatisfied clauses as energy.
        """
        assignment = initial_assignment[:]
        current_unsat = get_unsat_clauses(assignment)
        current_energy = len(current_unsat)
        
        if current_energy == 0:
            return assignment
        
        best_unsat = current_energy
        best_assignment = assignment.copy()
        
        t = t_initial
        i = 0
        while i < max_iterations and t > t_final:
            if time.time() > deadline:
                break
            
            # Pick a random variable to flip
            v = rng.randint(0, n - 1)
            old_val = assignment[v]
            assignment[v] = not assignment[v]
            
            new_unsat = get_unsat_clauses(assignment)
            new_energy = len(new_unsat)
            
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < math.exp(-delta / t):
                # Accept move
                current_energy = new_energy
                current_unsat = new_unsat
                
                if current_energy < best_unsat:
                    best_unsat = current_energy
                    best_assignment = assignment.copy()
                    
                    if best_unsat == 0:
                        return best_assignment
            else:
                # Reject move
                assignment[v] = old_val
            
            # Cool down
            t *= cooling_rate
            i += 1
        
        if best_unsat == 0:
            return best_assignment
        return None

    # Run Weighted WalkSAT with smooth surrogate and restarts
    max_restarts = 150
    base_steps = 5000
    
    # Track best solution across restarts for polish
    global_best_unsat = float('inf')
    global_best_assignment = None
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Vary parameters for diversity
        # Noise decreases with restarts, temperature varies
        noise = max(0.1, 0.4 - r * 0.002)
        soft_temp = 0.5 + 0.5 * math.exp(-r * 0.02)  # Anneal the soft-max temperature
        steps = base_steps + r * 300
        
        sol = weighted_walksat_smooth(rng, steps, noise_prob=noise, 
                                       soft_temp=soft_temp, restart_id=r)
        if sol is not None:
            unsat_count = count_unsat(sol)
            if unsat_count == 0:
                return sol
            if unsat_count < global_best_unsat:
                global_best_unsat = unsat_count
                global_best_assignment = sol.copy()

    # Final polish pass with simulated annealing if we have a good candidate
    if global_best_assignment is not None and global_best_unsat <= 5 and time.time() < deadline - 30:
        polished = simulated_annealing_polish(rng, global_best_assignment, 
                                                max_iterations=50000,
                                                t_initial=1.0, t_final=0.005,
                                                cooling_rate=0.9998)
        if polished is not None:
            if count_unsat(polished) == 0:
                return polished

    # Fallback: Try one more long smooth walk if time permits
    if time.time() < deadline - 15:
        if global_best_assignment is not None:
            # Continue from best with lower temperature
            sol = weighted_walksat_smooth(rng, 20000, noise_prob=0.1, 
                                           soft_temp=0.3, restart_id=max_restarts)
            if sol is not None:
                if count_unsat(sol) == 0:
                    return sol
                if count_unsat(sol) < global_best_unsat:
                    global_best_unsat = count_unsat(sol)
                    global_best_assignment = sol
        
        # Last resort: pure random walk
        assignment = [rng.random() < 0.5 for _ in range(n)]
        unsat_clauses = get_unsat_clauses(assignment)
        while time.time() < deadline - 1 and unsat_clauses:
            ci = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            v, sign = clause_data[ci][rng.randint(0, 2)]
            assignment[v] = not assignment[v]
            unsat_clauses = get_unsat_clauses(assignment)
        if not unsat_clauses:
            return assignment
        if global_best_assignment is not None:
            return global_best_assignment
        return assignment

    if global_best_assignment is not None:
        return global_best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
