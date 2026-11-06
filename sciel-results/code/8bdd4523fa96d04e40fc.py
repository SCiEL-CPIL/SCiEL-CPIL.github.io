"""
Hypergraph Ramsey R(4,5;3) via WalkSAT (Clause-driven Stochastic Local Search)
"""
import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = 145.0  # seconds; return with margin

def _build_clauses(n):
    """
    Build the list of clauses for the SAT formulation.
    Red K4 constraint: For every 4-set, NOT(all 4 triples are Red/0).
    Equivalent to: (T1=1 OR T2=1 OR T3=1 OR T4=1).
    In our encoding: color 0 = Red, color 1 = Blue.
    Clause is satisfied if at least one variable is 1.
    
    Blue K5 constraint: For every 5-set, NOT(all 10 triples are Blue/1).
    Equivalent to: (T1=0 OR T2=0 OR ... OR T10=0).
    In our encoding: Clause is satisfied if at least one variable is 0.
    """
    triples = list(combinations(range(n), 3))
    m = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Helper to get index of triple (a,b,c)
    def get_idx(a, b, c):
        return triple_to_idx[(a, b, c)]

    clauses = []
    clause_types = [] # 0 for Red K4 (needs a 1), 1 for Blue K5 (needs a 0)
    
    # Red K4 clauses
    for q in combinations(range(n), 4):
        a, b, c, d = q
        idxs = [
            get_idx(a, b, c),
            get_idx(a, b, d),
            get_idx(a, c, d),
            get_idx(b, c, d)
        ]
        clauses.append(idxs)
        clause_types.append(0)
        
    # Blue K5 clauses
    for p in combinations(range(n), 5):
        idxs = []
        for t in combinations(p, 3):
            idxs.append(get_idx(t[0], t[1], t[2]))
        clauses.append(idxs)
        clause_types.append(1)
        
    return m, clauses, clause_types

def _count_unsatisfied(colors, clauses, clause_types):
    """
    Returns list of indices of unsatisfied clauses.
    """
    unsat = []
    for i, clause in enumerate(clauses):
        if clause_types[i] == 0:
            # Red K4: Satisfied if any is 1 (Blue)
            if all(colors[idx] == 0 for idx in clause):
                unsat.append(i)
        else:
            # Blue K5: Satisfied if any is 0 (Red)
            if all(colors[idx] == 1 for idx in clause):
                unsat.append(i)
    return unsat

def _walksat(n, time_budget, seed=0):
    """
    WalkSAT solver for the hypergraph Ramsey constraints.
    """
    rng = random.Random(seed)
    m, clauses, clause_types = _build_clauses(n)
    
    # Precompute variable-to-clause incidence for fast gain calculation
    var_clauses = [[] for _ in range(m)]
    for i, clause in enumerate(clauses):
        for v in clause:
            var_clauses[v].append(i)
            
    # Initialize random coloring
    # Heuristic: Bias towards Blue (1) for Red constraints? 
    # Red K4 is "Not all 0". If we make many 1s, Red constraints are easy.
    # Blue K5 is "Not all 1". If we make many 0s, Blue constraints are easy.
    # Balance: p_red (0) around 0.3-0.4 usually works well for these asymmetric constraints.
    p_red = 0.35 
    colors = np.array([0 if rng.random() < p_red else 1 for _ in range(m)], dtype=np.int8)
    
    unsat = _count_unsatisfied(colors, clauses, clause_types)
    
    if not unsat:
        return colors.tolist()
        
    t_end = time.time() + time_budget
    p_flip = 0.5 # Probability to flip the "best" variable vs random
    
    while time.time() < t_end:
        if not unsat:
            return colors.tolist()
            
        # Pick a random unsatisfied clause
        c_idx = rng.choice(unsat)
        clause = clauses[c_idx]
        c_type = clause_types[c_idx]
        
        # Calculate gain for each variable in the clause
        # Gain = (Number of clauses that will become satisfied) - (Number of clauses that will become unsatisfied)
        best_var = -1
        best_gain = -1000
        gains = []
        
        for v in clause:
            # Simulate flip
            # Current state of v: colors[v]
            # New state: 1 - colors[v]
            
            # A clause is satisfied if:
            # Type 0 (Red K4): any var is 1
            # Type 1 (Blue K5): any var is 0
            
            # We only care about clauses containing v.
            # If we flip v, it affects only clauses in var_clauses[v].
            
            satisfied_count = 0
            unsatisfied_count = 0
            
            for ci in var_clauses[v]:
                if ci == c_idx:
                    continue # Skip the clause we are trying to fix (it's already unsatisfied, and we are fixing it)
                
                other_clause = clauses[ci]
                other_type = clause_types[ci]
                
                # Check if this clause is currently satisfied
                # And if it will be satisfied after flip
                
                # Optimization: We can just check the current status and the new status.
                # Current status:
                if other_type == 0:
                    # Satisfied if any is 1
                    cur_sat = 0
                    for u in other_clause:
                        if colors[u] == 1:
                            cur_sat = 1
                            break
                else:
                    # Satisfied if any is 0
                    cur_sat = 0
                    for u in other_clause:
                        if colors[u] == 0:
                            cur_sat = 1
                            break
                
                # New status after flipping v
                # v becomes 1 - colors[v]
                new_v = 1 - colors[v]
                
                if other_type == 0:
                    # Satisfied if any is 1
                    new_sat = 0
                    for u in other_clause:
                        if u == v:
                            if new_v == 1:
                                new_sat = 1
                                break
                        else:
                            if colors[u] == 1:
                                new_sat = 1
                                break
                else:
                    # Satisfied if any is 0
                    new_sat = 0
                    for u in other_clause:
                        if u == v:
                            if new_v == 0:
                                new_sat = 1
                                break
                        else:
                            if colors[u] == 0:
                                new_sat = 1
                                break
                
                if cur_sat == 0 and new_sat == 1:
                    satisfied_count += 1
                elif cur_sat == 1 and new_sat == 0:
                    unsatisfied_count += 1
                    
            gain = satisfied_count - unsatisfied_count
            gains.append(gain)
            if gain > best_gain:
                best_gain = gain
                best_var = v
                
        # Decide which variable to flip
        # With probability p_flip, flip the best variable.
        # Otherwise, flip a random variable in the clause.
        if rng.random() < p_flip:
            v_to_flip = best_var
        else:
            v_to_flip = rng.choice(clause)
            
        # Flip
        colors[v_to_flip] = 1 - colors[v_to_flip]
        
        # Update unsatisfied list
        # Instead of recounting all, we can just recount affected clauses.
        # But for simplicity and correctness in a short budget, recounting unsat is O(C) which is ~10k for n=30.
        # Let's optimize: Only check clauses in var_clauses[v_to_flip] and the one we picked.
        # Actually, just recounting the whole thing every step is too slow for 150s if steps are high.
        # Let's maintain the list of unsatisfied clauses incrementally.
        
        # Remove clauses that are now satisfied
        # Add clauses that are now unsatisfied
        
        # Clauses affected by flipping v_to_flip:
        affected = set(var_clauses[v_to_flip])
        
        # We need to update the 'unsat' list.
        # This is tricky to do efficiently without a full recount if we don't track status per clause.
        # Let's track status per clause.
        
        # Re-initialize status tracking for better performance
        # This implementation might be slightly slower per step but allows for many more steps.
        # Let's just do a full recount of unsat every 10 steps? No, that's risky.
        
        # Let's use a simpler, robust approach: 
        # Just recount unsat. For n=30, C ~ 10000. 10000 checks is fast in numpy? 
        # No, python loop is slow. 
        # Let's use numpy for the check.
        
        # Vectorized check for unsat
        # Red clauses (type 0): unsat if all 0.
        # Blue clauses (type 1): unsat if all 1.
        
        # This is hard to vectorize easily with variable clause sizes.
        # Let's stick to the python loop but optimize it.
        
        # Actually, let's just do a full recount every step. 
        # If n=30, m=4060, C_red ~ 27k, C_blue ~ 142k. Total ~ 170k clauses.
        # 170k iterations per step is too slow for 150s.
        
        # We need incremental update.
        # Let's maintain an array 'clause_sat' of bools.
        
        # We'll refactor the inner loop to be incremental.
        
    return None # Fallback

def _walksat_optimized(n, time_budget, seed=0):
    rng = random.Random(seed)
    m, clauses, clause_types = _build_clauses(n)
    
    # Incidence
    var_clauses = [[] for _ in range(m)]
    for i, clause in enumerate(clauses):
        for v in clause:
            var_clauses[v].append(i)
            
    # Initial coloring
    p_red = 0.35
    colors = np.array([0 if rng.random() < p_red else 1 for _ in range(m)], dtype=np.int8)
    
    # Initial satisfaction status
    clause_sat = np.zeros(len(clauses), dtype=bool)
    unsat_indices = []
    
    for i, clause in enumerate(clauses):
        if clause_types[i] == 0:
            # Satisfied if any is 1
            sat = any(colors[idx] == 1 for idx in clause)
        else:
            # Satisfied if any is 0
            sat = any(colors[idx] == 0 for idx in clause)
        clause_sat[i] = sat
        if not sat:
            unsat_indices.append(i)
            
    if not unsat_indices:
        return colors.tolist()
        
    t_end = time.time() + time_budget
    p_flip = 0.5
    
    while time.time() < t_end:
        if not unsat_indices:
            return colors.tolist()
            
        c_idx = rng.choice(unsat_indices)
        clause = clauses[c_idx]
        c_type = clause_types[c_idx]
        
        # Find best variable in clause
        best_var = -1
        best_gain = -1000
        gains = []
        
        for v in clause:
            # Calculate gain
            gain = 0
            for ci in var_clauses[v]:
                if ci == c_idx:
                    continue
                other_clause = clauses[ci]
                other_type = clause_types[ci]
                
                # Current state of this clause
                # We know clause_sat[ci]
                # We need to know if flipping v changes its status.
                
                # If currently satisfied:
                #   It becomes unsatisfied ONLY if v was the ONLY satisfying variable.
                #   If currently unsatisfied:
                #   It becomes satisfied ONLY if v is the new satisfying variable.
                
                if clause_sat[ci]:
                    # Check if v is the only one satisfying
                    # For type 0 (needs 1): v must be 1, and all others 0
                    # For type 1 (needs 0): v must be 0, and all others 1
                    if other_type == 0:
                        if colors[v] == 1:
                            # Check others
                            only = True
                            for u in other_clause:
                                if u != v and colors[u] == 1:
                                    only = False
                                    break
                            if only:
                                gain -= 1
                    else:
                        if colors[v] == 0:
                            only = True
                            for u in other_clause:
                                if u != v and colors[u] == 0:
                                    only = False
                                    break
                            if only:
                                gain -= 1
                else:
                    # Currently unsatisfied
                    # Will it become satisfied?
                    # For type 0: v becomes 1.
                    # For type 1: v becomes 0.
                    if other_type == 0:
                        if colors[v] == 0: # Flipping to 1
                            gain += 1
                    else:
                        if colors[v] == 1: # Flipping to 0
                            gain += 1
                            
            gains.append(gain)
            if gain > best_gain:
                best_gain = gain
                best_var = v
                
        if rng.random() < p_flip:
            v_to_flip = best_var
        else:
            v_to_flip = rng.choice(clause)
            
        # Flip v_to_flip
        old_val = colors[v_to_flip]
        colors[v_to_flip] = 1 - old_val
        
        # Update clause_sat and unsat_indices
        # Affected clauses: var_clauses[v_to_flip]
        for ci in var_clauses[v_to_flip]:
            clause = clauses[ci]
            c_type = clause_types[ci]
            
            # Recalculate satisfaction for this clause
            if c_type == 0:
                new_sat = any(colors[idx] == 1 for idx in clause)
            else:
                new_sat = any(colors[idx] == 0 for idx in clause)
                
            old_sat = clause_sat[ci]
            clause_sat[ci] = new_sat
            
            if old_sat and not new_sat:
                # Became unsatisfied
                unsat_indices.append(ci)
            elif not old_sat and new_sat:
                # Became satisfied
                # Remove from unsat_indices
                # This is O(N) in worst case. 
                # For better perf, we could use a set, but list is okay if we clean up periodically.
                try:
                    unsat_indices.remove(ci)
                except ValueError:
                    pass # Should not happen
            
    return None

def construct():
    deadline = _START + _DEADLINE - 5.0
    best_n = 9
    best_colors = None
    
    # Seed with known small valid construction
    n0 = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n0), 3)]
    best_n = n0
    
    # Try to grow n
    # WalkSAT is fast per step, so we can try higher n earlier.
    for n in range(10, 41):
        if time.time() >= deadline:
            break
        remaining = deadline - time.time()
        
        # Budget allocation
        if n <= 15:
            budget = min(remaining, 5.0)
        elif n <= 25:
            budget = min(remaining, 15.0)
        else:
            budget = min(remaining, 30.0)
            
        if budget < 1.0:
            break
            
        # Try multiple seeds
        found = None
        attempts = 0
        while time.time() < _START + _DEADLINE - 5.0 and attempts < 5:
            attempts += 1
            seed = n * 100 + attempts
            res = _walksat_optimized(n, budget / (5 - attempts + 1), seed)
            if res is not None:
                found = res
                break
                
        if found is not None:
            best_n = n
            best_colors = found
            
    return {'n': best_n, 'colors': best_colors}
