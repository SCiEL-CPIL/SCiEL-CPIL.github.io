import time
import random
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 285  # 285s to be safe within 300s limit

    # Precomputation for fast updates
    var_clauses = [[] for _ in range(n)]
    clauses_store = []
    
    for i, clause in enumerate(clauses):
        cl_store = []
        for l in clause:
            v = abs(l) - 1
            sign = 1 if l > 0 else -1
            cl_store.append((v, sign))
            var_clauses[v].append((i, sign))
        clauses_store.append(cl_store)
    
    m = len(clauses)
    
    def evaluate(assignment):
        """Returns number of unsatisfied clauses."""
        unsat_count = 0
        for i in range(m):
            satisfied = False
            for v, sign in clauses_store[i]:
                if (assignment[v] == (sign == 1)):
                    satisfied = True
                    break
            if not satisfied:
                unsat_count += 1
        return unsat_count

    def solve_one_reinit(rng, max_steps=20000, no_improve_limit=1000):
        """
        Runs local search from a random initialization.
        Returns assignment if solved, else None.
        """
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initial count of satisfied literals per clause
        clause_sat_counts = [0] * m
        for i in range(m):
            cnt = 0
            for v, sign in clauses_store[i]:
                if assignment[v] == (sign == 1):
                    cnt += 1
            clause_sat_counts[i] = cnt
        
        total_unsat = sum(1 for c in clause_sat_counts if c == 0)
        if total_unsat == 0:
            return assignment
        
        max_steps = 10000
        no_improve_count = 0
        best_unsat = total_unsat
        best_assignment = assignment.copy()
        
        for step in range(max_steps):
            if time.time() > deadline:
                return best_assignment if evaluate(best_assignment) == 0 else None
            
            if total_unsat == 0:
                return assignment
            
            # Pick a random unsatisfied clause
            unsat_clauses = [i for i in range(m) if clause_sat_counts[i] == 0]
            
            if not unsat_clauses:
                return assignment
            
            cl_idx = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            
            # Evaluate flipping each variable in this clause
            best_var_idx = -1
            best_gain = -1
            
            for v, sign in clauses_store[cl_idx]:
                current_satisfies_cl = (assignment[v] == (sign == 1))
                delta_cl_idx = -1 if current_satisfies_cl else 1
                
                total_delta_sat = delta_cl_idx
                for c_idx, c_sign in var_clauses[v]:
                    if c_idx == cl_idx:
                        continue
                    c_satisfies_v = (assignment[v] == (c_sign == 1))
                    if c_satisfies_v:
                        total_delta_sat -= 1
                    else:
                        total_delta_sat += 1
                
                if total_delta_sat > best_gain:
                    best_gain = total_delta_sat
                    best_var_idx = v
            
            # If no variable increases satisfaction, pick a random one from the clause
            if best_var_idx == -1:
                v, sign = clauses_store[cl_idx][rng.randint(0, 2)]
                best_var_idx = v

            # Perform the flip
            v = best_var_idx
            old_val = assignment[v]
            assignment[v] = not assignment[v]
            
            # Update clause_sat_counts
            for c_idx, c_sign in var_clauses[v]:
                if old_val == (c_sign == 1):
                    clause_sat_counts[c_idx] -= 1
                else:
                    clause_sat_counts[c_idx] += 1
            
            new_total_unsat = sum(1 for c in clause_sat_counts if c == 0)
            
            if new_total_unsat < best_unsat:
                best_unsat = new_total_unsat
                best_assignment = assignment.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > no_improve_limit:
                    break
                    
            total_unsat = new_total_unsat
            
        # Check if best_assignment is a solution
        if evaluate(best_assignment) == 0:
            return best_assignment
        return None

    # Multiple restarts with different seeds
    rng = random.Random(42)
    
    max_restarts = 20
    for r in range(max_restarts):
        if time.time() > deadline:
            break
        
        sol = solve_one_reinit(rng)
        if sol is not None:
            return sol
            
    # If we run out of time or restarts without finding a solution,
    # return the best assignment we had, even if unsat, to avoid crash.
    # We'll just generate a random one as a last resort.
    return [rng.random() < 0.5 for _ in range(n)]
