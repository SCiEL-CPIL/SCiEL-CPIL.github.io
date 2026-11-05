"""Hybrid 3-SAT solver: Incremental WalkSAT with Simulated Annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import defaultdict


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    deadline = time.time() + 290.0  # Safety margin
    num_clauses = len(clauses)

    if num_clauses == 0:
        return [True] * n

    # Build variable-to-clauses index for fast incremental updates
    var_to_clauses = defaultdict(list)
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_to_clauses[abs(lit) - 1].append(ci)

    # Precompute clause info for fast satisfaction check
    # clause_info[ci] = list of (var_idx, target_val)
    clause_info = []
    for clause in clauses:
        info = []
        for lit in clause:
            var_idx = abs(lit) - 1
            target_val = (lit > 0)
            info.append((var_idx, target_val))
        clause_info.append(info)

    # Precompute for each clause, a map from var_idx to its position (for quick lookup)
    # clause_var_pos[ci] = {var_idx: position_in_clause}
    clause_var_pos = []
    for info in clause_info:
        pos_map = {}
        for i, (v, t) in enumerate(info):
            pos_map[v] = i
        clause_var_pos.append(pos_map)

    def is_clause_satisfied(ci, assignment):
        for var_idx, target_val in clause_info[ci]:
            if assignment[var_idx] == target_val:
                return True
        return False

    def compute_unsat_set(assignment):
        """Returns a set of unsatisfied clause indices."""
        unsat = set()
        for ci in range(num_clauses):
            if not is_clause_satisfied(ci, assignment):
                unsat.add(ci)
        return unsat

    def update_unsat_after_flip(var_idx, old_val, new_val, unsat_set, assignment):
        """Update unsat set after flipping variable var_idx from old_val to new_val.
        Returns delta (change in count). Modifies unsat_set in place."""
        old_count = len(unsat_set)
        
        for ci in var_to_clauses[var_idx]:
            was_unsat = ci in unsat_set
            # Check if clause is now satisfied
            now_sat = False
            for v_idx, t_val in clause_info[ci]:
                if assignment[v_idx] == t_val:
                    now_sat = True
                    break
            
            if now_sat:
                if was_unsat:
                    unsat_set.discard(ci)
            else:
                if not was_unsat:
                    unsat_set.add(ci)
        
        new_count = len(unsat_set)
        return new_count - old_count

    def make_random_assignment(seed):
        rng = random.Random(seed)
        return [rng.random() < 0.5 for _ in range(n)]

    def min_confl_score(var_idx, assignment, unsat_set, current_clause_ci):
        """
        Calculate the 'conflict' score for flipping var_idx.
        We want to minimize the number of currently satisfied clauses that would become unsatisfied.
        """
        conflict = 0
        for other_ci in var_to_clauses[var_idx]:
            if other_ci == current_clause_ci:
                continue
            
            # If the clause is already unsatisfied, skip
            if other_ci in unsat_set:
                continue
            
            # Check if flipping var_idx would break this satisfied clause
            # A satisfied clause becomes unsatisfied if the flipped variable was the ONLY satisfying literal
            other_sat_lit_count = 0
            for v_idx, t_val in clause_info[other_ci]:
                if assignment[v_idx] == t_val:
                    other_sat_lit_count += 1
                    if other_sat_lit_count > 1:
                        break
            
            if other_sat_lit_count == 1:
                # Only one literal satisfies it; check if it's var_idx
                for v_idx, t_val in clause_info[other_ci]:
                    if v_idx == var_idx and assignment[var_idx] == t_val:
                        conflict += 1
                        break
        
        return conflict

    def sa_search(assignment, unsat_set, rng, max_steps, T_start, T_end, cooling_rate):
        """
        Simulated Annealing search.
        Returns (assignment, unsat_count, unsat_set)
        """
        current_unsat = len(unsat_set)
        T = T_start
        
        if current_unsat == 0:
            return assignment, 0, unsat_set

        best_unsat = current_unsat
        best_assignment = assignment[:]
        best_unsat_set = unsat_set.copy()
        
        steps_since_improve = 0
        unsat_list = list(unsat_set)  # Keep a list for random selection
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if current_unsat == 0:
                return assignment, 0, unsat_set

            if not unsat_list:
                break
            
            # Select a random unsatisfied clause
            ci = rng.choice(unsat_list)
            clause = clause_info[ci]
            
            # Select a variable to flip from the clause
            var_idx, target_val = rng.choice(clause)
            
            old_val = assignment[var_idx]
            new_val = not old_val
            
            # Apply flip temporarily to calculate delta
            assignment[var_idx] = new_val
            delta = update_unsat_after_flip(var_idx, old_val, new_val, unsat_set, assignment)
            new_unsat_count = current_unsat + delta
            
            # Update unsat_list to reflect changes
            if delta < 0:
                # Some clauses became satisfied
                unsat_list = list(unsat_set)
            elif delta > 0:
                # Some clauses became unsatisfied
                unsat_list = list(unsat_set)
            
            # Acceptance criterion
            accepted = False
            if delta < 0:
                accepted = True
            elif delta == 0:
                accepted = True
            else:
                # Worsening move
                if T > 0:
                    prob = math.exp(-delta / T)
                    if rng.random() < prob:
                        accepted = True
            
            if accepted:
                current_unsat = new_unsat_count
                
                if current_unsat < best_unsat:
                    best_unsat = current_unsat
                    best_assignment = assignment[:]
                    best_unsat_set = unsat_set.copy()
                    steps_since_improve = 0
                else:
                    steps_since_improve += 1
                    
                T *= cooling_rate
                if T < T_end:
                    T = T_end
            else:
                # Revert flip
                assignment[var_idx] = old_val
                delta_revert = update_unsat_after_flip(var_idx, new_val, old_val, unsat_set, assignment)
                # delta_revert should be -delta
                steps_since_improve += 1
                T *= cooling_rate
                if T < T_end:
                    T = T_end
                
                if steps_since_improve > 1000:
                    break

        return best_assignment, best_unsat, best_unsat_set

    def greedy_polish(assignment, unsat_set, rng, max_steps):
        """
        Greedy hill climbing using min-confl.
        Used for final polishing when close to solution.
        """
        current_unsat = len(unsat_set)
        if current_unsat == 0:
            return assignment, 0, unsat_set
        
        best_unsat = current_unsat
        best_assignment = assignment[:]
        best_unsat_set = unsat_set.copy()
        
        unsat_list = list(unsat_set)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if current_unsat == 0:
                return assignment, 0, unsat_set
            
            if not unsat_list:
                break
                
            ci = rng.choice(unsat_list)
            clause = clause_info[ci]
            
            # Find the literal with minimum conflict
            best_var = None
            best_score = float('inf')
            
            for var_idx, target_val in clause:
                score = min_confl_score(var_idx, assignment, unsat_set, ci)
                score += rng.random() * 0.01
                
                if score < best_score:
                    best_score = score
                    best_var = var_idx
            
            if best_var is None:
                break
                
            old_val = assignment[best_var]
            new_val = not old_val
            
            assignment[best_var] = new_val
            delta = update_unsat_after_flip(best_var, old_val, new_val, unsat_set, assignment)
            new_unsat_count = current_unsat + delta
            
            if delta <= 0:
                current_unsat = new_unsat_count
                unsat_list = list(unsat_set)
                
                if current_unsat < best_unsat:
                    best_unsat = current_unsat
                    best_assignment = assignment[:]
                    best_unsat_set = unsat_set.copy()
            else:
                # Revert if it worsens (greedy)
                assignment[best_var] = old_val
                delta_revert = update_unsat_after_flip(best_var, new_val, old_val, unsat_set, assignment)
                break

        return best_assignment, best_unsat, best_unsat_set

    best_global_assignment = None
    best_global_unsat = num_clauses + 1

    max_restarts = 200
    for attempt in range(max_restarts):
        if time.time() > deadline:
            break
        
        seed = attempt * 7919 + 13
        assignment = make_random_assignment(seed)
        unsat_set = compute_unsat_set(assignment)
        cur_unsat = len(unsat_set)
        
        if cur_unsat < best_global_unsat:
            best_global_unsat = cur_unsat
            best_global_assignment = assignment[:]
            
        if cur_unsat == 0:
            return assignment

        rng = random.Random(attempt * 31337 + 42)
        
        steps_sa = 20000 
        T_start = 5.0
        T_end = 0.01
        cooling_rate = 0.9995
        
        if n > 40:
            steps_sa = 30000
            cooling_rate = 0.9997
            
        assignment, cur_unsat, unsat_set = sa_search(
            assignment, unsat_set, rng, 
            steps_sa, T_start, T_end, cooling_rate
        )
        
        if cur_unsat < best_global_unsat:
            best_global_unsat = cur_unsat
            best_global_assignment = assignment[:]
            
        if cur_unsat == 0:
            return assignment

        if cur_unsat <= 10:
            polish_steps = 5000
            assignment, cur_unsat, unsat_set = greedy_polish(
                assignment, unsat_set, rng, polish_steps
            )
            
            if cur_unsat < best_global_unsat:
                best_global_unsat = cur_unsat
                best_global_assignment = assignment[:]
                
            if cur_unsat == 0:
                return assignment

    if best_global_assignment is not None and best_global_unsat <= 5:
        rng = random.Random(99999)
        final_unsat_set = compute_unsat_set(best_global_assignment)
        final_assignment, final_unsat, _ = greedy_polish(best_global_assignment, final_unsat_set, rng, 20000)
        if final_unsat == 0:
            return final_assignment

    return best_global_assignment if best_global_assignment else make_random_assignment(0)

# EVOLVE-BLOCK-END
