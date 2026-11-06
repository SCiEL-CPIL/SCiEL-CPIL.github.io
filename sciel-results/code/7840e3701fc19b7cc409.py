"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before 330s kill
    
    # Pre-process: convert clauses to 0-based indices with sign
    # literal l > 0 means variable (l-1) should be True
    # literal l < 0 means variable (abs(l)-1) should be False
    num_clauses = len(clauses)
    
    # Build a literal-to-variable mapping and sign
    # For efficiency, store clauses as (var_idx, is_positive)
    clauses_struct = []
    for c in clauses:
        cl = []
        for lit in c:
            var_idx = abs(lit) - 1
            is_pos = lit > 0
            cl.append((var_idx, is_pos))
        clauses_struct.append(cl)
    
    # Build reverse index: for each variable, which clauses contain it
    var_to_clauses = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses_struct):
        for var_idx, _ in cl:
            var_to_clauses[var_idx].append(ci)
    
    def is_satisfied(ci, assignment):
        cl = clauses_struct[ci]
        for var_idx, is_pos in cl:
            val = assignment[var_idx]
            if is_pos and val:
                return True
            elif not is_pos and not val:
                return True
        return False
    
    def count_unsat(assignment):
        return sum(1 for ci in range(num_clauses) if not is_satisfied(ci, assignment))
    
    # Phase 1: Random restart WalkSAT with phase saving
    # Use a list of best assignments found
    
    def walksat_restart(rng, max_steps, phase):
        # Initialize assignment
        if phase is not None:
            assignment = list(phase)
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Track unsatisfied clauses
        unsat_clauses = [ci for ci in range(num_clauses) if not is_satisfied(ci, assignment)]
        
        if not unsat_clauses:
            return assignment, True
        
        for step in range(max_steps):
            if time.time() > deadline - 10:
                return assignment, False
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            cl = clauses_struct[ci]
            
            # Count how many literals in this clause are satisfied
            sat_count = 0
            for var_idx, is_pos in cl:
                val = assignment[var_idx]
                if (is_pos and val) or (not is_pos and not val):
                    sat_count += 1
            
            # Choose a literal to flip
            # Weighted random: prefer literals that would satisfy more other clauses
            # Simple approach: pick random literal from the clause
            var_idx, is_pos = rng.choice(cl)
            
            # Update unsat_clauses efficiently
            old_val = assignment[var_idx]
            new_val = not old_val
            assignment[var_idx] = new_val
            
            # Update unsat status for clauses containing this variable
            new_unsat = []
            for cj in var_to_clauses[var_idx]:
                if cj == ci:
                    # Check if this clause is now satisfied
                    if is_satisfied(cj, assignment):
                        # Remove from unsat
                        pass
                    else:
                        new_unsat.append(cj)
                else:
                    if is_satisfied(cj, assignment):
                        # Remove from unsat
                        pass
                    else:
                        new_unsat.append(cj)
            
            # Rebuild unsat_clauses: keep those that were unsat and still are, 
            # plus any that became unsat
            # This is O(num_clauses) which is expensive. Let's use a different approach.
            
            # Actually, let's just recompute unsat_clauses periodically
            # For now, use a simpler approach: maintain a set
            
            # Let's restart the approach with a set for unsat clauses
            break
        
        return assignment, False
    
    # Better WalkSAT implementation with proper tracking
    def walksat_proper(rng, max_steps, phase):
        if phase is not None:
            assignment = list(phase)
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Build unsat set
        unsat_set = set()
        for ci in range(num_clauses):
            if not is_satisfied(ci, assignment):
                unsat_set.add(ci)
        
        if not unsat_set:
            return assignment, True
        
        # Track clause satisfaction counts for faster updates
        # For each clause, count how many literals are satisfied
        clause_sat_count = [0] * num_clauses
        for ci in range(num_clauses):
            cl = clauses_struct[ci]
            count = 0
            for var_idx, is_pos in cl:
                val = assignment[var_idx]
                if (is_pos and val) or (not is_pos and not val):
                    count += 1
            clause_sat_count[ci] = count
        
        for step in range(max_steps):
            if time.time() > deadline - 10:
                return assignment, False
            
            if not unsat_set:
                return assignment, True
            
            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_set))
            cl = clauses_struct[ci]
            
            # Choose a literal to flip with weighting
            # Prefer literals whose flip satisfies more other clauses
            best_var = None
            best_score = -1
            weights = []
            for var_idx, is_pos in cl:
                # Score: number of currently unsat clauses that would become sat
                # minus number of currently sat clauses that would become unsat
                score = 0
                for cj in var_to_clauses[var_idx]:
                    if clause_sat_count[cj] == 0:
                        # Currently unsat
                        # Check if flipping this var would satisfy cj
                        clj = clauses_struct[cj]
                        for v, p in clj:
                            if v == var_idx:
                                # This is the literal we're considering
                                # If we flip, does it satisfy cj?
                                # Current value: assignment[var_idx]
                                # New value: not assignment[var_idx]
                                # Literal is satisfied if (p and new_val) or (not p and not new_val)
                                new_val = not assignment[var_idx]
                                if (p and new_val) or (not p and not new_val):
                                    score += 1
                                break
                    elif clause_sat_count[cj] == 1:
                        # Currently sat by exactly one literal
                        # Check if that literal is this variable
                        clj = clauses_struct[cj]
                        for v, p in clj:
                            val = assignment[v]
                            if (p and val) or (not p and not val):
                                if v == var_idx:
                                    # Flipping would make this clause unsat
                                    score -= 1
                                break
                weights.append(score)
            
            # Weighted random selection
            max_w = max(weights) if weights else 0
            min_w = min(weights) if weights else 0
            if max_w == min_w:
                idx = rng.randrange(len(cl))
            else:
                # Exponential weighting
                total = 0.0
                cum = []
                for w in weights:
                    total += math.exp((w - min_w) / max(1, (max_w - min_w)))
                    cum.append(total)
                r = rng.random() * total
                idx = 0
                for i, c in enumerate(cum):
                    if r <= c:
                        idx = i
                        break
                else:
                    idx = len(cl) - 1
            
            var_idx, is_pos = cl[idx]
            old_val = assignment[var_idx]
            new_val = not old_val
            assignment[var_idx] = new_val
            
            # Update clause_sat_count for all clauses containing var_idx
            for cj in var_to_clauses[var_idx]:
                clj = clauses_struct[cj]
                # Find the literal for var_idx in this clause
                for v, p in clj:
                    if v == var_idx:
                        # Old contribution
                        old_sat = (p and old_val) or (not p and not old_val)
                        # New contribution
                        new_sat = (p and new_val) or (not p and not new_val)
                        if old_sat and not new_sat:
                            clause_sat_count[cj] -= 1
                        elif not old_sat and new_sat:
                            clause_sat_count[cj] += 1
                        break
                
                # Update unsat_set
                if clause_sat_count[cj] == 0:
                    unsat_set.add(cj)
                elif cj in unsat_set:
                    unsat_set.discard(cj)
        
        return assignment, False
    
    # Try multiple restarts with phase saving
    best_assignment = None
    best_unsat_count = float('inf')
    phase = None
    
    num_restarts = 50
    steps_per_restart = 5000
    
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        success = False
        if phase is None:
            assignment, success = walksat_proper(random.Random(r * 12345 + 67890), steps_per_restart, None)
        else:
            assignment, success = walksat_proper(random.Random(r * 12345 + 67890), steps_per_restart, phase)
        
        if success:
            return assignment
        
        unsat_count = count_unsat(assignment)
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = list(assignment)
            phase = list(assignment)
    
    # Phase 2: Local search with tabu list on best assignment
    if best_assignment is not None:
        assignment = list(best_assignment)
        unsat_set = set()
        for ci in range(num_clauses):
            if not is_satisfied(ci, assignment):
                unsat_set.add(ci)
        
        clause_sat_count = [0] * num_clauses
        for ci in range(num_clauses):
            cl = clauses_struct[ci]
            count = 0
            for var_idx, is_pos in cl:
                val = assignment[var_idx]
                if (is_pos and val) or (not is_pos and not val):
                    count += 1
            clause_sat_count[ci] = count
        
        tabu = [0] * n
        tabu_len = 20
        rng = random.Random(99999)
        
        for step in range(20000):
            if time.time() > deadline - 5:
                break
            
            if not unsat_set:
                return assignment
            
            # Find best move
            best_var = -1
            best_delta = 0
            best_clauses = []
            
            # Only consider variables in unsat clauses
            candidates = set()
            for ci in unsat_set:
                for var_idx, _ in clauses_struct[ci]:
                    candidates.add(var_idx)
            
            for var_idx in candidates:
                if tabu[var_idx] > step - tabu_len:
                    continue
                
                old_val = assignment[var_idx]
                new_val = not old_val
                
                delta = 0
                for cj in var_to_clauses[var_idx]:
                    clj = clauses_struct[cj]
                    for v, p in clj:
                        if v == var_idx:
                            old_sat = (p and old_val) or (not p and not old_val)
                            new_sat = (p and new_val) or (not p and not new_val)
                            if old_sat and not new_sat:
                                delta -= 1
                            elif not old_sat and new_sat:
                                delta += 1
                            break
                
                if delta > best_delta:
                    best_delta = delta
                    best_var = var_idx
            
            if best_var == -1:
                # Pick random candidate
                if candidates:
                    best_var = rng.choice(list(candidates))
                else:
                    break
            
            old_val = assignment[best_var]
            new_val = not old_val
            assignment[best_var] = new_val
            tabu[best_var] = step
            
            # Update
            for cj in var_to_clauses[best_var]:
                clj = clauses_struct[cj]
                for v, p in clj:
                    if v == best_var:
                        old_sat = (p and old_val) or (not p and not old_val)
                        new_sat = (p and new_val) or (not p and not new_val)
                        if old_sat and not new_sat:
                            clause_sat_count[cj] -= 1
                        elif not old_sat and new_sat:
                            clause_sat_count[cj] += 1
                        break
                
                if clause_sat_count[cj] == 0:
                    unsat_set.add(cj)
                elif cj in unsat_set:
                    unsat_set.discard(cj)
    
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: return random assignment
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
