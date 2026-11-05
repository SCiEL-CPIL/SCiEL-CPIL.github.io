"""Planted 3-SAT solver with Weighted WalkSAT (WSAT) and Restarts."""
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    time_limit = 250  # leave margin under 300s

    def is_sat(assignment):
        for c in clauses:
            sat = False
            for l in c:
                if assignment[abs(l) - 1] == (l > 0):
                    sat = True
                    break
            if not sat:
                return False
        return True

    def count_unsat(assignment):
        cnt = 0
        for c in clauses:
            sat = False
            for l in c:
                if assignment[abs(l) - 1] == (l > 0):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

    def wsat(seed, max_steps, growth=2.0, decay=0.99):
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        weights = [0.0] * n
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                return None
            
            # Check if satisfied
            unsat_clauses = []
            for c in clauses:
                sat = False
                for l in c:
                    if assignment[abs(l) - 1] == (l > 0):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(c)
            
            if not unsat_clauses:
                return assignment

            # Pick a random unsatisfied clause
            clause = unsat_clauses[rng.randrange(len(unsat_clauses))]
            
            # Select variable from clause based on weights
            # Weights are for variables 0..n-1
            # Get indices of variables in this clause
            var_indices = [abs(l) - 1 for l in clause]
            
            # Calculate total weight for selection
            # If all weights are 0, pick uniformly
            total_weight = sum(weights[v] for v in var_indices)
            
            if total_weight == 0:
                v = var_indices[rng.randrange(len(var_indices))]
            else:
                # Weighted random selection
                r = rng.random() * total_weight
                cumulative = 0
                v = var_indices[-1] # default
                for idx in var_indices:
                    cumulative += weights[idx]
                    if r <= cumulative:
                        v = idx
                        break

            # Flip variable
            assignment[v] = not assignment[v]
            
            # Update weights
            # Increase weight of flipped variable
            weights[v] += growth
            # Decay all weights
            for i in range(n):
                weights[i] *= decay

        return None

    # Strategy: Multiple restarts with increasing step budgets
    # WSAT benefits from restarts to escape local minima
    
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000,
             7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000]
    
    # First pass: moderate steps
    for i, seed in enumerate(seeds):
        if time.time() - start_time > time_limit:
            break
        max_steps = 2000 + i * 1000
        result = wsat(seed, max_steps)
        if result is not None:
            return result

    # Second pass: longer steps with different parameters
    for i, seed in enumerate(seeds):
        if time.time() - start_time > time_limit:
            break
        max_steps = 5000 + i * 2000
        result = wsat(seed, max_steps, growth=1.5, decay=0.98)
        if result is not None:
            return result

    # Third pass: very long single run or many short ones
    for i in range(100, 200):
        if time.time() - start_time > time_limit:
            break
        max_steps = 10000
        result = wsat(i, max_steps, growth=2.5, decay=0.995)
        if result is not None:
            return result

    # Fallback
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
