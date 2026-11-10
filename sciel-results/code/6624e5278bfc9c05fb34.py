"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def _build_clause_data(n, clauses):
    """Precompute clause structure as numpy arrays for vectorized operations."""
    m = len(clauses)
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_sgn = np.zeros((m, 3), dtype=np.int8)
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            clause_vars[i, j] = abs(lit) - 1
            clause_sgn[i, j] = 1 if lit > 0 else -1
    # For each variable, precompute which clauses contain it and with what sign
    var_to_clauses = {}
    for v in range(n):
        var_to_clauses[v] = []
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_to_clauses[v].append((i, j))
    return clause_vars, clause_sgn, var_to_clauses


def _compute_sat_state(clause_vars, clause_sgn, assignment):
    """Compute per-clause satisfaction state: sat_count (how many literals satisfied)."""
    m, k = clause_vars.shape
    var_idx = clause_vars  # (m, k)
    signs = clause_sgn     # (m, k)
    truth = assignment[var_idx]  # (m, k) bool
    sat_lit = (signs == 1) & truth | (signs == -1) & (~truth)
    sat_count = sat_lit.sum(axis=1)  # (m,)
    return sat_count


def _phase_seed(n, clauses, rng):
    """Initialize assignment using phase information from clause polarities."""
    pos_count = [0] * n
    neg_count = [0] * n
    for clause in clauses:
        for lit in clause:
            if lit > 0:
                pos_count[lit - 1] += 1
            else:
                neg_count[-lit - 1] += 1
    assignment = []
    for i in range(n):
        if pos_count[i] > neg_count[i]:
            assignment.append(True)
        elif neg_count[i] > pos_count[i]:
            assignment.append(False)
        else:
            assignment.append(rng.random() < 0.5)
    return assignment


def _count_unsat(assignment, clauses):
    """Count number of unsatisfied clauses."""
    count = 0
    for clause in clauses:
        sat = False
        for lit in clause:
            if assignment[abs(lit) - 1] == (lit > 0):
                sat = True
                break
        if not sat:
            count += 1
    return count


def _vectorized_walksat(assignment, clause_vars, clause_sgn, var_to_clauses, n, m, rng, max_steps, deadline, noise=0.2):
    """
    Vectorized WalkSAT with adaptive noise:
    - Maintains sat_count per clause
    - On flip of variable v, updates sat_count for all clauses containing v
    - Greedy: pick literal in unsat clause that maximizes net gain
    - Random: with probability noise, flip random literal in clause
    """
    assignment = np.array(assignment, dtype=bool)
    sat_count = _compute_sat_state(clause_vars, clause_sgn, assignment)
    
    for step in range(max_steps):
        if time.time() >= deadline:
            break
        
        unsat_mask = sat_count == 0
        n_unsat = int(unsat_mask.sum())
        if n_unsat == 0:
            break
        
        # Adaptive noise: reduce randomness as we get closer to solution
        current_noise = noise * (0.5 + 0.5 * n_unsat / max(m, 1))
        
        # Pick a random unsatisfied clause
        unsat_indices = np.where(unsat_mask)[0]
        ci = int(rng.choice(unsat_indices.tolist()))
        
        # Get the 3 literals in this clause
        lits_vars = clause_vars[ci]  # (3,)
        lits_sgn = clause_sgn[ci]    # (3,)
        
        # Decide: random flip or greedy
        if rng.random() < current_noise:
            # Random flip of one literal in the clause
            li = rng.randint(0, 2)
            v = int(lits_vars[li])
            old_val = assignment[v]
            assignment[v] = not assignment[v]
            # Update sat_count for all clauses containing v
            for (ci2, li2) in var_to_clauses[v]:
                ci2 = int(ci2)
                li2 = int(li2)
                v2 = int(clause_vars[ci2, li2])
                s2 = int(clause_sgn[ci2, li2])
                if s2 == 1:
                    old_lit_sat = old_val
                else:
                    old_lit_sat = not old_val
                new_val = not old_val
                if s2 == 1:
                    new_lit_sat = new_val
                else:
                    new_lit_sat = not new_val
                if new_lit_sat and not old_lit_sat:
                    sat_count[ci2] += 1
                elif old_lit_sat and not new_lit_sat:
                    sat_count[ci2] -= 1
        else:
            # Greedy: for each of the 3 literals, compute net gain
            best_gain = -999
            best_v = -1
            
            for li in range(3):
                v = int(lits_vars[li])
                s = int(lits_sgn[li])
                old_val = assignment[v]
                new_val = not old_val
                
                # Compute gain: for each clause containing v, check if flipping v changes its sat status
                gain = 0
                for (ci2, li2) in var_to_clauses[v]:
                    ci2 = int(ci2)
                    li2 = int(li2)
                    v2 = int(clause_vars[ci2, li2])
                    s2 = int(clause_sgn[ci2, li2])
                    
                    if s2 == 1:
                        cur_lit_sat = old_val
                    else:
                        cur_lit_sat = not old_val
                    
                    if s2 == 1:
                        new_lit_sat = new_val
                    else:
                        new_lit_sat = not new_val
                    
                    if ci2 == ci:
                        # This is the target clause (currently unsat)
                        new_sc = 0
                        for jj in range(3):
                            vv = int(clause_vars[ci, jj])
                            ss = int(clause_sgn[ci, jj])
                            val = new_val if vv == v else assignment[vv]
                            if ss == 1:
                                if val:
                                    new_sc += 1
                            else:
                                if not val:
                                    new_sc += 1
                        if new_sc > 0:
                            gain += 1
                    else:
                        if new_lit_sat and not cur_lit_sat:
                            if sat_count[ci2] == 0:
                                gain += 1
                        elif cur_lit_sat and not new_lit_sat:
                            if sat_count[ci2] == 1:
                                gain -= 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            
            if best_v >= 0:
                old_val = assignment[best_v]
                assignment[best_v] = not assignment[best_v]
                new_val = not old_val
                for (ci2, li2) in var_to_clauses[best_v]:
                    ci2 = int(ci2)
                    li2 = int(li2)
                    v2 = int(clause_vars[ci2, li2])
                    s2 = int(clause_sgn[ci2, li2])
                    if s2 == 1:
                        old_lit_sat = old_val
                        new_lit_sat = new_val
                    else:
                        old_lit_sat = not old_val
                        new_lit_sat = not new_val
                    if new_lit_sat and not old_lit_sat:
                        sat_count[ci2] += 1
                    elif old_lit_sat and not new_lit_sat:
                        sat_count[ci2] -= 1
    
    return assignment.tolist()


def _polish_local_search(assignment, clause_vars, clause_sgn, var_to_clauses, n, m, deadline):
    """
    Final polish: try all single-variable flips to reduce unsat count.
    Repeat until no improvement or deadline.
    """
    assignment = np.array(assignment, dtype=bool)
    sat_count = _compute_sat_state(clause_vars, clause_sgn, assignment)
    
    for _ in range(200):  # Max 200 rounds of polishing
        if time.time() >= deadline:
            break
        
        improved = False
        for v in range(n):
            if time.time() >= deadline:
                break
            
            old_val = assignment[v]
            new_val = not old_val
            
            # Compute change in sat_count for all clauses containing v
            delta = 0
            for (ci2, li2) in var_to_clauses[v]:
                ci2 = int(ci2)
                li2 = int(li2)
                s2 = int(clause_sgn[ci2, li2])
                
                if s2 == 1:
                    cur_lit_sat = old_val
                    new_lit_sat = new_val
                else:
                    cur_lit_sat = not old_val
                    new_lit_sat = not new_val
                
                if new_lit_sat and not cur_lit_sat:
                    delta += 1
                elif cur_lit_sat and not new_lit_sat:
                    delta -= 1
            
            if delta > 0:
                # Flip improves things
                assignment[v] = new_val
                for (ci2, li2) in var_to_clauses[v]:
                    ci2 = int(ci2)
                    li2 = int(li2)
                    s2 = int(clause_sgn[ci2, li2])
                    
                    if s2 == 1:
                        old_lit_sat = old_val
                        new_lit_sat = new_val
                    else:
                        old_lit_sat = not old_val
                        new_lit_sat = not new_val
                    
                    if new_lit_sat and not old_lit_sat:
                        sat_count[ci2] += 1
                    elif old_lit_sat and not new_lit_sat:
                        sat_count[ci2] -= 1
                improved = True
        
        if not improved:
            break
    
    return assignment.tolist()


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280  # Leave 50s margin before 330s kill
    
    m = len(clauses)
    
    # Precompute clause structure
    clause_vars, clause_sgn, var_to_clauses = _build_clause_data(n, clauses)
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Strategy 1: Phase-seeded vectorized WalkSAT with multiple restarts
    for attempt in range(100):
        if time.time() >= deadline:
            break
        seed = attempt * 7919 + 12345
        rng = random.Random(seed)
        
        # Use phase seed, but for later attempts, perturb the best assignment found
        if attempt < 3 or best_assignment is None:
            assignment = _phase_seed(n, clauses, rng)
        else:
            # Start from best assignment with some random perturbations
            assignment = best_assignment[:]
            num_flips = max(1, n // 10)
            flip_indices = rng.sample(range(n), num_flips)
            for idx in flip_indices:
                assignment[idx] = not assignment[idx]
        
        # Initial check
        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # Vectorized WalkSAT with adaptive noise
        steps = 1500 + attempt * 200
        assignment = _vectorized_walksat(assignment, clause_vars, clause_sgn, var_to_clauses, n, m, rng, steps, deadline, noise=0.25)
        
        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
    
    # Strategy 2: Polish the best assignment found
    if best_assignment is not None:
        assignment = _polish_local_search(best_assignment, clause_vars, clause_sgn, var_to_clauses, n, m, deadline)
        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
    
    # Strategy 3: Pure random vectorized WalkSAT from scratch
    for attempt in range(40):
        if time.time() >= deadline:
            break
        seed = 100000 + attempt * 31337
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        assignment = _vectorized_walksat(assignment, clause_vars, clause_sgn, var_to_clauses, n, m, rng, 2500, deadline, noise=0.2)
        
        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
    
    # Final polish on best assignment
    if best_assignment is not None:
        assignment = _polish_local_search(best_assignment, clause_vars, clause_sgn, var_to_clauses, n, m, deadline)
        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: all False
    return [False] * n
# EVOLVE-BLOCK-END
