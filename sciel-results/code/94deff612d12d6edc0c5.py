"""Synthetic planted 3-SAT solver (150 variables) with efficient WalkSAT."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    # Deadline guard: finish well before 330s wall-clock
    deadline = time.time() + 280.0
    
    # Precompute clause structure
    clause_data = []
    for c in clauses:
        cd = [(abs(l) - 1, l > 0) for l in c]
        clause_data.append(cd)
    
    # Build variable-to-clause index
    var_to_clauses = defaultdict(list)
    for ci, cd in enumerate(clause_data):
        for vi, pos in cd:
            var_to_clauses[vi].append((ci, pos))
    
    def run_walksat(rng, max_steps):
        """Efficient WalkSAT."""
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Track which clauses are unsatisfied
        is_unsat = [False] * num_clauses
        
        # Initial evaluation
        for ci, cd in enumerate(clause_data):
            is_sat = False
            for vi, pos in cd:
                if assignment[vi] == pos:
                    is_sat = True
                    break
            is_unsat[ci] = not is_sat
        
        # List of unsatisfied clause indices (use a set for O(1) lookup, list for random access)
        unsat_set = set(ci for ci in range(num_clauses) if is_unsat[ci])
        
        w = 0.55  # WalkSAT weight parameter
        
        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False
            
            if not unsat_set:
                return assignment, True
            
            # Pick a random unsatisfied clause
            ci = next(iter(unsat_set))
            cd = clause_data[ci]
            
            # For each variable in the clause, compute the cost of flipping it
            # Cost = (number of currently-sat clauses that would become unsat) - 
            #        (number of currently-unsat clauses that would become sat)
            # Note: the chosen clause ci is currently unsat, so flipping any var in it
            #       that satisfies it will reduce unsat count by 1 (for ci)
            
            candidate_costs = []
            for vi, pos in cd:
                cost = 0
                for cj, pos_j in var_to_clauses[vi]:
                    if assignment[vi] == pos_j:
                        # This literal is currently satisfied. After flip, it won't be.
                        if not is_unsat[cj]:
                            # Clause cj is currently sat. Check if this was the only satisfied literal
                            cd_j = clause_data[cj]
                            only_this = True
                            for vi2, pos2 in cd_j:
                                if vi2 != vi and assignment[vi2] == pos2:
                                    only_this = False
                                    break
                            if only_this:
                                cost += 1
                    else:
                        # This literal is currently not satisfied. After flip, it will be.
                        if is_unsat[cj]:
                            cost -= 1
                candidate_costs.append((vi, cost))
            
            # With probability w, pick the best variable; otherwise, pick random
            if rng.random() < w:
                min_cost = min(c for _, c in candidate_costs)
                best_vis = [vi for vi, c in candidate_costs if c == min_cost]
                vi = best_vis[rng.randrange(len(best_vis))]
            else:
                vi = cd[rng.randrange(len(cd))][0]
            
            # Flip variable vi
            assignment[vi] = not assignment[vi]
            
            # Update clause statuses for all clauses containing vi
            for cj, pos_j in var_to_clauses[vi]:
                # Determine new satisfaction status of clause cj
                cd_j = clause_data[cj]
                is_now_sat = False
                for vi2, pos2 in cd_j:
                    if assignment[vi2] == pos2:
                        is_now_sat = True
                        break
                
                if is_unsat[cj] and is_now_sat:
                    # Clause became satisfied
                    is_unsat[cj] = False
                    unsat_set.discard(cj)
                elif not is_unsat[cj] and not is_now_sat:
                    # Clause became unsatisfied
                    is_unsat[cj] = True
                    unsat_set.add(cj)
        
        return assignment, False
    
    # Run with multiple restarts
    rng = random.Random(0)
    assignment = None
    solved = False
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        rng_attempt = random.Random(attempt * 7919 + 104729)
        assignment, solved = run_walksat(rng_attempt, 30000)
        
        if solved:
            break
    
    return assignment
# EVOLVE-BLOCK-END
