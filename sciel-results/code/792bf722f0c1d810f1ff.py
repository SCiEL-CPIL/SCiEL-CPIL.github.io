<<<<<<< SEARCH
"""SAT solver for planted 3-SAT instances."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.linalg import eigh


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin before 300s limit
    
    # Convert to 0-based internal representation
    clauses_internal = []
    for c in clauses:
        clause_int = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            clause_int.append((var, sign))
        clauses_internal.append(clause_int)
    
    # Build variable-to-clause index
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses_internal):
        for var, sign in clause:
            var_to_clauses[var].append((ci, sign))
    
    # Precompute clause data as numpy arrays for efficient WalkSAT
    clause_vars = np.array([[v for v, s in c] for c in clauses_internal], dtype=np.int32)
    clause_signs = np.array([[s for v, s in c] for c in clauses_internal], dtype=np.int8)
=======
"""SAT solver for planted 3-SAT instances."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.linalg import eigh


def variable_elimination(clauses_internal, n, time_limit):
    """
    Greedy variable elimination: repeatedly fix the variable whose assignment
    minimizes the ratio of newly-unsatisfied to newly-satisfied clauses.
    Returns (assignment_dict, simplified_clauses) where simplified_clauses has
    eliminated variables removed.
    """
    assignment = {}
    active_clauses = list(range(len(clauses_internal)))
    
    # Build var -> list of (clause_idx, sign) for active clauses
    var_to_active = [[] for _ in range(n)]
    for ci in active_clauses:
        for var, sign in clauses_internal[ci]:
            var_to_active[var].append((ci, sign))
    
    active_vars = set()
    for ci in active_clauses:
        for var, sign in clauses_internal[ci]:
            active_vars.add(var)
    
    # Track which clauses are currently unsatisfied under partial assignment
    clause_status = {}  # ci -> 'sat', 'unsat', 'pending'
    for ci in active_clauses:
        sat = False
        pending = False
        for var, sign in clauses_internal[ci]:
            if var in assignment:
                if assignment[var] == (sign == 1):
                    sat = True
                    break
            else:
                pending = True
        if sat:
            clause_status[ci] = 'sat'
        elif not pending:
            clause_status[ci] = 'unsat'
        else:
            clause_status[ci] = 'pending'
    
    while active_clauses and time.time() < time_limit:
        if not active_vars:
            break
        
        # For each active variable, compute:
        #   gain_true  = # clauses that become sat if var=True, minus # that become unsat
        #   gain_false = same for var=False
        # We want the variable with best (minimum conflict / maximum consistency)
        
        best_var = None
        best_gain = -float('inf')
        best_val = None
        
        for var in list(active_vars):
            if not var_to_active[var]:
                continue
            
            # Count effect of setting var=True and var=False
            gain_true = 0
            gain_false = 0
            
            for ci, sign in var_to_active[var]:
                if clause_status[ci] == 'sat':
                    continue  # already satisfied, no change either way
                
                # Clause is pending or unsat
                # Check other literals in this clause
                others_sat = False
                others_pending = False
                for v2, s2 in clauses_internal[ci]:
                    if v2 == var:
                        continue
                    if v2 in assignment:
                        if assignment[v2] == (s2 == 1):
                            others_sat = True
                            break
                    else:
                        others_pending = True
                
                if others_sat:
                    continue  # clause would be sat regardless of var
                
                if sign == 1:
                    # var=True satisfies this literal
                    if not others_pending:
                        gain_true += 1  # was unsat/pending, now sat
                    # if others_pending, still pending after setting var
                else:
                    # var=False satisfies this literal
                    if not others_pending:
                        gain_false += 1
            
            # Also count clauses where setting this var would make them unsat
            # (clauses where the other literals are all false and this var is the only hope)
            # Actually, for a 3-CNF clause with 3 literals, if we set var and the clause
            # is still pending (other vars unassigned), it doesn't become unsat yet.
            # It only becomes unsat if all other literals are already False.
            
            # The key insight: for planted SAT, the "true" assignment of the planted
            # solution should have higher consistency. Use gain as proxy.
            
            total_gain = gain_true + gain_false
            if total_gain == 0:
                # No immediate gain, skip or use small penalty
                continue
            
            # Prefer the assignment with higher gain
            if gain_true >= gain_false:
                candidate_gain = gain_true
                candidate_val = True
            else:
                candidate_gain = gain_false
                candidate_val = False
            
            # Score: prefer variables with high gain (they resolve many clauses)
            # and low ambiguity (gain difference)
            score = candidate_gain / (1.0 + abs(gain_true - gain_false) * 0.1)
            
            if score > best_gain:
                best_gain = score
                best_var = var
                best_val = candidate_val
        
        if best_var is None:
            # No variable has a clear gain; pick one with most clauses
            if not active_vars:
                break
            best_var = max(active_vars, key=lambda v: len(var_to_active[v]))
            # Default to True for simplicity
            best_val = True
        
        # Fix the variable
        assignment[best_var] = best_val
        
        # Update clause statuses and remove satisfied clauses
        new_active_clauses = []
        for ci in active_clauses:
            if clause_status[ci] == 'sat':
                continue  # remove satisfied clauses
            
            # Check if this clause is now satisfied or unsatisfied
            sat = False
            all_false = True
            for var, sign in clauses_internal[ci]:
                if var in assignment:
                    if assignment[var] == (sign == 1):
                        sat = True
                        break
                    else:
                        # this literal is false
                        pass
                else:
                    all_false = False
            
            if sat:
                continue  # remove satisfied
            elif all_false:
                # Clause is unsatisfied -> conflict in our heuristic
                # For planted SAT this shouldn't happen often; keep it
                pass
            
            new_active_clauses.append(ci)
        
        # Update var_to_active
        for ci in new_active_clauses:
            for var, sign in clauses_internal[ci]:
                if var == best_var:
                    continue  # remove from active tracking
                # var is still active
                pass
        
        # Rebuild var_to_active from scratch (simpler, slower but correct)
        var_to_active = [[] for _ in range(n)]
        active_vars = set()
        for ci in new_active_clauses:
            for var, sign in clauses_internal[ci]:
                if var not in assignment:
                    var_to_active[var].append((ci, sign))
                    active_vars.add(var)
        
        # Update clause_status
        for ci in new_active_clauses:
            sat = False
            pending = False
            for var, sign in clauses_internal[ci]:
                if var in assignment:
                    if assignment[var] == (sign == 1):
                        sat = True
                        break
                else:
                    pending = True
            if sat:
                clause_status[ci] = 'sat'
            elif not pending:
                clause_status[ci] = 'unsat'
            else:
                clause_status[ci] = 'pending'
        
        active_clauses = new_active_clauses
        active_vars = set()
        for ci in active_clauses:
            for var, sign in clauses_internal[ci]:
                if var not in assignment:
                    active_vars.add(var)
    
    # Build simplified clauses (remove assigned variables)
    simplified_clauses = []
    for ci in active_clauses:
        new_clause = []
        for var, sign in clauses_internal[ci]:
            if var not in assignment:
                new_clause.append((var, sign))
        if new_clause:
            simplified_clauses.append(new_clause)
    
    return assignment, simplified_clauses


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin before 300s limit
    
    # Convert to 0-based internal representation
    clauses_internal = []
    for c in clauses:
        clause_int = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            clause_int.append((var, sign))
        clauses_internal.append(clause_int)
    
    # Build variable-to-clause index
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses_internal):
        for var, sign in clause:
            var_to_clauses[var].append((ci, sign))
    
    # Precompute clause data as numpy arrays for efficient WalkSAT
    clause_vars = np.array([[v for v, s in c] for c in clauses_internal], dtype=np.int32)
    clause_signs = np.array([[s for v, s in c] for c in clauses_internal], dtype=np.int8)
    
    # === VARIABLE ELIMINATION PREPROCESSING ===
    elim_time_limit = start_time + 3.0  # Spend up to 3s on elimination
    elim_assignment, simplified_clauses = variable_elimination(
        clauses_internal, n, elim_time_limit
    )
    
    # If elimination solved everything or simplified significantly, use it
    # Build final assignment from elimination + solve remaining
    final_assignment = [None] * n
    for var, val in elim_assignment.items():
        final_assignment[var] = val
    
    # If no remaining clauses, we're done
    if not simplified_clauses:
        for v in range(n):
            if final_assignment[v] is None:
                final_assignment[v] = False
        return [bool(x) for x in final_assignment]
    
    # Collect remaining unassigned variables
    remaining_vars = set()
    for clause in simplified_clauses:
        for var, sign in clause:
            remaining_vars.add(var)
    
    remaining_vars = sorted(remaining_vars)
    var_map = {v: i for i, v in enumerate(remaining_vars)}
    local_n = len(remaining_vars)
    
    # Build local clauses
    local_clauses = []
    for clause in simplified_clauses:
        local_clause = []
        for var, sign in clause:
            local_clause.append((var_map[var], sign))
        local_clauses.append(local_clause)
    
    # Build local variable-to-clause index
    local_var_to_clauses = [[] for _ in range(local_n)]
    for ci, lc in enumerate(local_clauses):
        for li, sign in lc:
            local_var_to_clauses[li].append((ci, sign))
    
    # === WALKSAT on simplified instance ===
    def local_unsat_count(assign):
        count = 0
        for lc in local_clauses:
            sat = False
            for li, sign in lc:
                if assign[li] == (sign == 1):
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    def compute_delta(li, new_val, assign):
        delta = 0
        for ci, sign in local_var_to_clauses[li]:
            lc = local_clauses[ci]
            was_sat = False
            for lli, s in lc:
                if assign[lli] == (s == 1):
                    was_sat = True
                    break
            
            would_be_sat = False
            for lli, s in lc:
                if lli == li:
                    if new_val == (s == 1):
                        would_be_sat = True
                        break
                else:
                    if assign[lli] == (s == 1):
                        would_be_sat = True
                        break
            
            if was_sat and not would_be_sat:
                delta += 1
            elif not was_sat and would_be_sat:
                delta -= 1
        return delta
    
    best_assign = None
    best_unsat = float('inf')
    noise_start = 0.45
    patience = 5000
    rng = random.Random(42)
    
    ws_deadline = deadline - 5.0
    
    while time.time() < ws_deadline:
        current = [rng.random() < 0.5 for _ in range(local_n)]
        current_unsat = local_unsat_count(current)
        run_best = current_unsat
        no_improve = 0
        
        for step in range(100000):
            if time.time() > ws_deadline:
                break
            
            if current_unsat == 0:
                return [bool(final_assignment[v]) if final_assignment[v] is not None else False for v in range(n)]
            
            if current_unsat < run_best:
                run_best = current_unsat
                no_improve = 0
                if current_unsat < best_unsat:
                    best_unsat = current_unsat
                    best_assign = list(current)
            else:
                no_improve += 1
            
            if no_improve > patience:
                break
            
            # Find unsatisfied clauses
            unsat_clauses = []
            for ci, lc in enumerate(local_clauses):
                sat = False
                for li, sign in lc:
                    if current[li] == (sign == 1):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(ci)
            
            if not unsat_clauses:
                return [bool(final_assignment[v]) if final_assignment[v] is not None else False for v in range(n)]
            
            chosen = rng.choice(unsat_clauses)
            lc = local_clauses[chosen]
            
            # Pick variable to flip
            if rng.random() < noise_start:
                candidates = [li for li, sign in lc if current[li] != (sign == 1)]
                if not candidates:
                    candidates = [li for li, sign in lc]
                best_var = rng.choice(candidates)
            else:
                best_var = None
                best_delta = float('inf')
                for li, sign in lc:
                    if current[li] == (sign == 1):
                        continue
                    new_val = not current[li]
                    delta = compute_delta(li, new_val, current)
                    if delta < best_delta:
                        best_delta = delta
                        best_var = li
                if best_var is None:
                    candidates = [li for li, sign in lc if current[li] != (sign == 1)]
                    if not candidates:
                        candidates = [li for li, sign in lc]
                    best_var = rng.choice(candidates)
            
            current[best_var] = not current[best_var]
            current_unsat = local_unsat_count(current)
        
        noise_start = max(0.15, noise_start * 0.95)
    
    # Use best found
    if best_assign is not None:
        for i, v in enumerate(remaining_vars):
            final_assignment[v] = best_assign[i]
    
    # Fill any unassigned variables
    for v in range(n):
        if final_assignment[v] is None:
            final_assignment[v] = False
    
    return [bool(x) for x in final_assignment]
>>>>>>> REPLACE

Now I need to remove the old code that comes after the original `clause_signs` line since I've replaced the entire function body:

