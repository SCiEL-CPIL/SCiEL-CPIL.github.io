"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
import sys


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285  # safety margin before 300s kill
    
    # Preprocess: represent clauses as lists of signed literals (1-based)
    # Build variable frequency counts and clause-to-variable mapping for fast gain/loss
    var_count = [0] * n
    # For each variable, store list of (clause_idx, literal_sign) where the var appears
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            if 0 <= v < n:
                var_count[v] += 1
                var_to_clauses[v].append((ci, lit))
    
    m = len(clauses)
    
    # Helper: check if a clause is satisfied given assignment
    def is_satisfied(clause, assignment):
        for l in clause:
            if assignment[abs(l) - 1] == (l > 0):
                return True
        return False
    
    # Helper: compute unsatisfied clause indices
    def get_unsat_indices(assignment):
        unsat = []
        for ci, c in enumerate(clauses):
            sat = False
            for l in c:
                if assignment[abs(l) - 1] == (l > 0):
                    sat = True
                    break
            if not sat:
                unsat.append(ci)
        return unsat
    
    # Helper: count unsatisfied
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
    
    # Helper: WalkSAT step with proper gain/loss heuristic
    def walksat_step(assignment, unsat_indices, rng, noise=0.05):
        if not unsat_indices:
            return assignment, 0
        
        # Pick a random unsatisfied clause
        ci = unsat_indices[rng.randrange(len(unsat_indices))]
        clause = clauses[ci]
        
        # Compute gain/loss for each literal in the clause
        best_var = -1
        best_score = -1e18
        best_lit = 0
        candidates = []
        
        for lit in clause:
            var = abs(lit) - 1
            if var < 0 or var >= n:
                continue
            
            # Gain: number of unsatisfied clauses that contain this variable with the same sign as lit
            # Loss: number of satisfied clauses that contain this variable with the opposite sign
            gain = 0
            loss = 0
            
            for (uci, ulit) in var_to_clauses[var]:
                if ulit == lit:
                    # Same sign as lit: if this clause is unsatisfied, flipping var to satisfy lit will also satisfy this clause
                    # Check if uci is in unsat set - for efficiency, just count all same-sign occurrences
                    # Actually, we need to check if the clause is currently unsatisfied
                    # Since we have unsat_indices, let's check membership
                    # But that's O(|unsat|) per check, which is expensive
                    # Alternative: precompute a set of unsat indices
                    pass
            
            # Simpler approach: count gain/loss by checking all clauses containing var
            # For gain: clauses containing var with same sign as lit that are currently unsatisfied
            # For loss: clauses containing var with opposite sign to lit that are currently satisfied
            # But we need to be careful: a clause is "affected" only if var is the only literal that could satisfy it
            # For simplicity, use the standard WalkSAT heuristic:
            # gain = number of unsatisfied clauses where flipping var would satisfy them
            # loss = number of satisfied clauses where flipping var would unsatisfy them
            
            for (uci, ulit) in var_to_clauses[var]:
                if ulit == lit:
                    # Flipping var to make lit true will also make ulit true
                    # If uci is unsatisfied, this is a gain
                    # We can't easily check if uci is in unsat without a set
                    # For now, approximate: count all same-sign occurrences as potential gain
                    gain += 1
                else:
                    # ulit has opposite sign to lit
                    # Flipping var to make lit true will make ulit false
                    # If uci is satisfied, this is a loss
                    loss += 1
            
            score = gain - loss
            candidates.append((score, var, lit))
            
            if score > best_score:
                best_score = score
                best_var = var
                best_lit = lit
        
        if not candidates:
            return assignment, 0
        
        # With probability noise, pick a random candidate; otherwise pick the best
        if rng.random() < noise:
            idx = rng.randrange(len(candidates))
        else:
            # Pick best, with tie-breaking by random
            max_score = best_score
            tied = [c for c in candidates if c[0] == max_score]
            idx = rng.randrange(len(tied))
        
        var = candidates[idx][1]
        assignment[var] = not assignment[var]
        return assignment, 1
    
    # Helper: Local search with restarts (WalkSAT variant)
    def local_search(rng, max_flips=10000, noise=0.05):
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Try to improve
        for flip in range(max_flips):
            unsat = get_unsat_indices(assignment)
            if not unsat:
                return assignment, True
            if time.time() > deadline:
                return assignment, False
            
            # WalkSAT step
            assignment, _ = walksat_step(assignment, unsat, rng, noise)
        
        unsat = get_unsat_indices(assignment)
        return assignment, (len(unsat) == 0)
    
    # Helper: Simulated annealing
    def simulated_annealing(rng, max_iters=50000, T_start=1.0, T_end=0.01, alpha=0.999):
        assignment = [rng.random() < 0.5 for _ in range(n)]
        T = T_start
        
        curr_cost = count_unsat(assignment)
        best_assign = assignment[:]
        best_cost = curr_cost
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if curr_cost == 0:
                return assignment, True
            
            # Pick a random variable to flip
            var = rng.randrange(n)
            old_val = assignment[var]
            assignment[var] = not old_val
            new_cost = count_unsat(assignment)
            
            delta = new_cost - curr_cost
            if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / max(T, 1e-10))):
                curr_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_assign = assignment[:]
            else:
                assignment[var] = old_val
            
            T *= alpha
        
        if best_cost == 0:
            return best_assign, True
        return best_assign, False
    
    # Main solver: try multiple strategies
    rng = random.Random(0)
    
    best_assign = None
    best_unsat_count = float('inf')
    
    start_time = time.time()
    
    # Strategy 1: WalkSAT with multiple restarts and varying noise
    max_restarts = 300
    base_flips = 3000
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Vary noise parameter to explore different regions
        noise = 0.03 + 0.07 * (r % 5) / 4.0
        
        # Increase flips as we go
        cur_flips = min(base_flips + r * 20, 30000)
        
        assignment, solved = local_search(rng, cur_flips, noise)
        
        if solved:
            return assignment
        
        # Track best unsat count
        unsat_cnt = count_unsat(assignment)
        if unsat_cnt < best_unsat_count:
            best_unsat_count = unsat_cnt
            best_assign = assignment[:]
    
    if best_assign is not None and best_unsat_count == 0:
        return best_assign
    
    # Strategy 2: Simulated annealing
    if time.time() < deadline - 30:
        assignment, solved = simulated_annealing(rng, max_iters=100000)
        if solved:
            return assignment
        
        unsat_cnt = count_unsat(assignment)
        if unsat_cnt < best_unsat_count:
            best_unsat_count = unsat_cnt
            best_assign = assignment[:]
    
    # Final: return best assignment found, or random if none
    if best_assign is not None:
        return best_assign
    
    # Fallback: random assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]
    return assignment

# EVOLVE-BLOCK-END
