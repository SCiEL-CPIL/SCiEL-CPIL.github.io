"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s out of 90s budget
    N = 180
    
    # 1. Precompute all reciprocal triples (a, b, c) with 1/a = 1/b + 1/c and a,b,c distinct in [1, N]
    # Note: 1/a = 1/b + 1/c  <=>  (b-a)(c-a) = a^2
    # Since a,b,c are distinct and positive, and 1/a = 1/b + 1/c > 1/c, we have a < c.
    # Similarly a < b. So a is the smallest element in any such triple.
    
    triples = []
    for a in range(1, N + 1):
        a_sq = a * a
        for d in range(1, a + 1):
            if a_sq % d == 0:
                e = a_sq // d
                b = a + d
                c = a + e
                if b <= N and c <= N and b != c:
                    triples.append((a, b, c))
    
    # Remove duplicates just in case
    triples = list(set(triples))
    
    # 2. Build conflict lists
    # For each element x, we need to know which pairs (y, z) cannot both be in the set if x is in the set.
    forbidden_pairs = [set() for _ in range(N + 1)]
    for (a, b, c) in triples:
        forbidden_pairs[a].add((b, c))
        forbidden_pairs[b].add((a, c))
        forbidden_pairs[c].add((a, b))
    
    # 3. Weighted Max-SAT formulation
    # Each variable x_i represents whether element i is in the set (True) or not (False)
    # Each triple (a, b, c) creates a clause: NOT(x_a AND x_b AND x_c)
    # Which is equivalent to: (NOT x_a) OR (NOT x_b) OR (NOT x_c)
    # We want to maximize the number of True variables while satisfying all clauses
    
    # WalkSAT-style local search with dynamic clause weighting
    def walk_sat_solver(initial_assignment, time_limit):
        # initial_assignment: list of booleans, index 0 unused, 1..N used
        # Returns: best assignment found
        
        # Clause weights: start with uniform weight 1.0
        clause_weights = [1.0] * len(triples)
        
        # Track which clauses are violated
        def count_violations(assignment):
            violations = 0
            for i, (a, b, c) in enumerate(triples):
                if assignment[a] and assignment[b] and assignment[c]:
                    violations += 1
            return violations
        
        def get_violated_clauses(assignment):
            violated = []
            for i, (a, b, c) in enumerate(triples):
                if assignment[a] and assignment[b] and assignment[c]:
                    violated.append(i)
            return violated
        
        # Objective: maximize number of True variables - penalty for violations
        # But we want a valid solution, so we prioritize validity first
        # Score = (number of True variables) * 1000 - (number of violations) * 10000
        # Actually, let's use a different approach:
        # We want to find a valid assignment (0 violations) with maximum True variables
        # If no valid assignment is found, return the one with minimum violations and max True variables
        
        current = list(initial_assignment)
        best = list(current)
        best_true_count = sum(1 for x in current[1:] if x)
        best_violations = count_violations(current)
        
        # We'll track the best valid solution and the best overall solution
        best_valid = None
        best_valid_count = 0
        
        start = time.time()
        improved = True
        no_improve_count = 0
        max_no_improve = 10000
        
        while time.time() < time_limit and no_improve_count < max_no_improve:
            # Check if current is valid
            curr_violations = count_violations(current)
            curr_true_count = sum(1 for x in current[1:] if x)
            
            if curr_violations == 0:
                # Valid solution
                if curr_true_count > best_valid_count:
                    best_valid = list(current)
                    best_valid_count = curr_true_count
            
            # Update best overall (for tracking)
            if curr_true_count > best_true_count or (curr_true_count == best_true_count and curr_violations < best_violations):
                best = list(current)
                best_true_count = curr_true_count
                best_violations = curr_violations
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= max_no_improve:
                break
            
            # If we have a valid solution, we can stop early if we've been stuck
            if best_valid is not None and no_improve_count > 5000:
                break
            
            # WalkSAT step: pick a clause to flip a variable in
            violated = get_violated_clauses(current)
            
            if not violated:
                # No violations, try to add more True variables
                # Find a variable that can be set to True without creating violations
                added = False
                for i in range(1, N + 1):
                    if not current[i]:
                        # Check if setting current[i] = True creates any violations
                        trial = list(current)
                        trial[i] = True
                        if count_violations(trial) == 0:
                            current = trial
                            added = True
                            break
                if not added:
                    # Can't add any more, break
                    break
                continue
            
            # Pick a violated clause, weighted by clause_weights
            total_weight = sum(clause_weights[i] for i in violated)
            r = random.random() * total_weight
            cumulative = 0
            selected_clause = violated[0]
            for i in violated:
                cumulative += clause_weights[i]
                if cumulative >= r:
                    selected_clause = i
                    break
            
            a, b, c = triples[selected_clause]
            
            # Try flipping each variable in the clause
            # For each variable, compute the change in number of violated clauses
            best_var = None
            best_delta = float('inf')
            
            for var in [a, b, c]:
                if not current[var]:
                    continue  # Only consider flipping True variables to False
                
                # Compute delta: how many clauses become satisfied - how many become violated
                # If we flip var from True to False:
                # Clauses containing var that were violated become satisfied
                # Clauses containing var that were satisfied might become violated (if they depend on var being True)
                
                satisfied_delta = 0
                violated_delta = 0
                
                for i, (x, y, z) in enumerate(triples):
                    if var not in (x, y, z):
                        continue
                    
                    was_violated = current[x] and current[y] and current[z]
                    # After flipping var to False:
                    new_violated = (not False) and current[y] and current[z] if var == x else \
                                   current[x] and (not False) and current[z] if var == y else \
                                   current[x] and current[y] and (not False)
                    
                    # More carefully:
                    if var == x:
                        new_violated = False and current[y] and current[z]
                    elif var == y:
                        new_violated = current[x] and False and current[z]
                    else:
                        new_violated = current[x] and current[y] and False
                    
                    new_violated = False  # Since one variable is now False, the clause is satisfied
                    
                    if was_violated and not new_violated:
                        satisfied_delta += 1
                    elif not was_violated and new_violated:
                        violated_delta += 1
                
                delta = violated_delta - satisfied_delta
                if delta < best_delta:
                    best_delta = delta
                    best_var = var
            
            if best_var is not None:
                current[best_var] = False
                # Update clause weights: increase weight of violated clauses
                violated_after = get_violated_clauses(current)
                for i in violated_after:
                    clause_weights[i] *= 1.1
                # Decay weights slightly
                for i in range(len(triples)):
                    clause_weights[i] *= 0.99
            
            # Occasionally, try to flip a False variable to True to increase density
            if random.random() < 0.1:
                # Try to add a random False variable
                false_vars = [i for i in range(1, N + 1) if not current[i]]
                if false_vars:
                    var = random.choice(false_vars)
                    trial = list(current)
                    trial[var] = True
                    if count_violations(trial) <= count_violations(current):
                        current = trial
        
        # Return the best valid solution if found, otherwise the best overall
        if best_valid is not None:
            return best_valid
        return best
    
    # 4. Multi-start WalkSAT
    random.seed(42)
    best_overall = None
    best_size_overall = 0
    
    # First, get a good initial solution using greedy
    def greedy_construct(order):
        S = set()
        for x in order:
            valid = True
            for (y, z) in forbidden_pairs[x]:
                if y in S and z in S:
                    valid = False
                    break
            if valid:
                S.add(x)
        return S
    
    # Generate initial solutions using different orderings
    initial_solutions = []
    for i in range(20):
        order = list(range(1, N + 1))
        random.shuffle(order)
        S = greedy_construct(order)
        if len(S) > best_size_overall:
            best_size_overall = len(S)
            best_overall = S
        
        # Convert to assignment
        assignment = [False] * (N + 1)
        for x in S:
            assignment[x] = True
        initial_solutions.append(assignment)
    
    # Run WalkSAT on each initial solution
    time_per_run = (deadline - time.time() - 5.0) / len(initial_solutions)
    if time_per_run < 1.0:
        time_per_run = 1.0
    
    for i, init in enumerate(initial_solutions):
        remaining = deadline - time.time() - 5.0
        if remaining < 2.0:
            break
        
        run_time = min(time_per_run, remaining)
        result = walk_sat_solver(init, run_time)
        
        # Convert result to set
        S = set()
        for x in range(1, N + 1):
            if result[x]:
                S.add(x)
        
        # Verify validity
        valid = True
        for (a, b, c) in triples:
            if a in S and b in S and c in S:
                valid = False
                break
        
        if valid and len(S) > best_size_overall:
            best_size_overall = len(S)
            best_overall = S
    
    # If no valid solution found, use the best greedy solution
    if best_overall is None:
        # Fallback to greedy
        order = list(range(1, N + 1))
        random.shuffle(order)
        best_overall = greedy_construct(order)
        best_size_overall = len(best_overall)
    
    return sorted(best_overall)

# EVOLVE-BLOCK-END
