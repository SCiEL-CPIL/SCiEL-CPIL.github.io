import random
import time
from itertools import combinations

def construct():
    """
    Finds a coloring of the 3-uniform hypergraph on n vertices with no all-red K4 
    and no all-blue K5. Uses a local search algorithm with precomputed clause-variable 
    mappings to maximize n.
    """
    
    def solve_for_n(n, time_limit):
        """
        Attempts to find a valid coloring for n vertices within time_limit seconds.
        Returns (n, colors) if found, else None.
        """
        start_time = time.time()
        
        # Generate all triples
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        # Generate Clauses
        # Clause type 0: Red K4 (4 vertices). Constraint: Not all 4 triples are 0.
        # Equivalent: At least one triple is 1.
        # Clause type 1: Blue K5 (5 vertices). Constraint: Not all 10 triples are 1.
        # Equivalent: At least one triple is 0.
        
        clause_vars = []
        clause_types = []
        
        # Red K4 constraints
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            t1 = (a, b, c)
            t2 = (a, b, d)
            t3 = (a, c, d)
            t4 = (b, c, d)
            
            vars_in_clause = [triple_index[t1], triple_index[t2], triple_index[t3], triple_index[t4]]
            clause_vars.append(vars_in_clause)
            clause_types.append(0)
            
        # Blue K5 constraints
        for quint in combinations(range(n), 5):
            # Get all 10 triples
            verts = list(quint)
            vars_in_clause = []
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        t = (verts[i], verts[j], verts[k])
                        vars_in_clause.append(triple_index[t])
            
            clause_vars.append(vars_in_clause)
            clause_types.append(1)
            
        num_clauses = len(clause_vars)
        
        # Precompute variable to clauses mapping
        var_clauses = [[] for _ in range(num_triples)]
        for c_idx, vars_in_c in enumerate(clause_vars):
            for v_idx in vars_in_c:
                var_clauses[v_idx].append(c_idx)
                
        # Initialize colors randomly
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Function to check if a clause is satisfied
        def is_satisfied(c_idx):
            ctype = clause_types[c_idx]
            vars_in_c = clause_vars[c_idx]
            if ctype == 0:
                # Red K4: Need at least one 1
                for v in vars_in_c:
                    if colors[v] == 1:
                        return True
                return False
            else:
                # Blue K5: Need at least one 0
                for v in vars_in_c:
                    if colors[v] == 0:
                        return True
                return False
                # Note: The logic above is correct.
                # If ctype 0 (Red K4 forbidden), we forbid all 0s. So we need at least one 1.
                # If ctype 1 (Blue K5 forbidden), we forbid all 1s. So we need at least one 0.
                
        # Initialize unsatisfied set
        unsat_set = set()
        for c_idx in range(num_clauses):
            if not is_satisfied(c_idx):
                unsat_set.add(c_idx)
                
        # Local Search
        while time.time() - start_time < time_limit:
            if not unsat_set:
                return n, colors
                
            # Pick a random unsatisfied clause
            # Using list(unsat_set) is O(N_unsat). If N_unsat is large, this is slow.
            # But as we converge, it gets smaller.
            # To optimize, we can keep a list of unsat clauses and update it.
            # But set is easier for add/remove.
            # Let's use a list for selection if set is too slow?
            # For now, let's stick to set and hope it's fast enough.
            # Optimization: If len(unsat_set) is small, list() is fine.
            
            # A faster way to pick a random element from a set without converting to list:
            # It's not directly possible in standard python efficiently for large sets.
            # However, we can maintain a parallel list `unsat_list` and a dict `unsat_dict` for O(1) removal?
            # That's complex.
            # Let's just use `random.choice(list(unsat_set))`.
            # If this is too slow, we can switch to a different strategy.
            
            c_idx = random.choice(list(unsat_set))
            
            # Pick a random variable in this clause
            v_idx = random.choice(clause_vars[c_idx])
            
            # Flip the variable
            old_color = colors[v_idx]
            colors[v_idx] = 1 - old_color
            
            # Update unsat_set for all clauses containing v_idx
            for c in var_clauses[v_idx]:
                # Check if clause c is now satisfied
                if is_satisfied(c):
                    if c in unsat_set:
                        unsat_set.remove(c)
                else:
                    unsat_set.add(c)
                    
            # Optional: If we got stuck (many restarts), we could randomize some variables.
            # But simple hill climbing with random restarts is usually better.
            # We are doing one long run here. If it gets stuck, it might stay stuck.
            # Let's add a small probability of a random flip if we haven't improved in a while?
            # Or just rely on the random choice of clause/variable to explore.
            
        return None

    # Try n=34 first
    # The time limit for the whole function is 120s.
    # Leave some buffer for overhead.
    # Try 34 with 110s.
    res = solve_for_n(34, 110)
    if res:
        return {'n': res[0], 'colors': res[1]}
        
    # If 34 fails, try smaller n quickly
    # We have ~10s left.
    for n in range(33, 8, -1):
        res = solve_for_n(n, 1)
        if res:
            return {'n': res[0], 'colors': res[1]}
            
    # Fallback to seed solution if all else fails
    # Seed solution: n=9, sum mod 3
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}
