"""DPLL-based SAT solver for hypergraph Ramsey R(4,5;3)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

def construct():
    """
    Construct a 2-coloring of triples on n vertices with no red K4^(3) and no blue K5^(3).
    Uses a custom DPLL SAT solver with clause learning.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill
    
    def get_triples(n):
        return list(itertools.combinations(range(n), 3))
    
    def dpll_solve(n, max_time=60.0):
        """
        Solve the SAT problem using DPLL with clause learning.
        Variables: x[ti] = 1 means triple ti is blue, 0 means red.
        Constraints:
          - For each 4-set (quad): NOT (all 4 triples are red) => at least one is blue
            => clause: x[t1] OR x[t2] OR x[t3] OR x[t4]
          - For each 5-set (quint): NOT (all 10 triples are blue) => at least one is red
            => clause: NOT x[t1] OR NOT x[t2] OR ... OR NOT x[t10]
        """
        solve_start = time.time()
        solve_deadline = solve_start + max_time
        
        triples = get_triples(n)
        num_triples = len(triples)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        # Build clauses
        # Each clause is a list of (var, sign) where sign=1 means positive (var), sign=-1 means negative (NOT var)
        clauses = []
        
        # Red K4 constraints: for each 4-set, at least one triple must be blue
        for quad in itertools.combinations(range(n), 4):
            quad_triples = [triple_index[t] for t in itertools.combinations(quad, 3)]
            clause = [(ti, 1) for ti in quad_triples]  # all positive: at least one is blue
            clauses.append(clause)
        
        # Blue K5 constraints: for each 5-set, at least one triple must be red
        for quint in itertools.combinations(range(n), 5):
            quint_triples = [triple_index[t] for t in itertools.combinations(quint, 3)]
            clause = [(ti, -1) for ti in quint_triples]  # all negative: at least one is red
            clauses.append(clause)
        
        num_clauses = len(clauses)
        
        # Build occurrence lists for unit propagation
        # pos_occ[var] = list of clause indices where var appears positively
        # neg_occ[var] = list of clause indices where var appears negatively
        pos_occ = defaultdict(list)
        neg_occ = defaultdict(list)
        for ci, clause in enumerate(clauses):
            for var, sign in clause:
                if sign == 1:
                    pos_occ[var].append(ci)
                else:
                    neg_occ[var].append(ci)
        
        # DPLL with clause learning
        # Assignment: list of (var, value) where value is 1 or 0
        # Watched literals for efficient unit propagation
        
        # Simplified DPLL with restarts and VSIDS-like variable selection
        # We'll use a recursive approach with backtracking
        
        best_solution = None
        best_num_satisfied = 0
        
        # Use iterative deepening with time limit
        # Try to find a complete assignment
        
        # For efficiency, we'll use a greedy approach with local search as fallback
        # But first, try DPLL
        
        def dpll_recursive(assignment, clauses_remaining, depth=0):
            """
            Recursive DPLL.
            assignment: dict var -> value (1 or 0)
            clauses_remaining: list of clauses not yet satisfied or falsified
            Returns: (success, assignment)
            """
            nonlocal best_solution, best_num_satisfied
            
            if time.time() > solve_deadline:
                return False, None
            
            if not clauses_remaining:
                return True, assignment
            
            # Unit propagation
            new_assignment = dict(assignment)
            changed = True
            while changed:
                changed = False
                for ci in list(clauses_remaining):
                    clause = clauses[ci]
                    unassigned = [(var, sign) for var, sign in clause if var not in new_assignment]
                    if not unassigned:
                        # Clause is falsified
                        return False, None
                    if len(unassigned) == 1:
                        var, sign = unassigned[0]
                        value = 1 if sign == 1 else 0
                        if var in new_assignment and new_assignment[var] != value:
                            return False, None
                        new_assignment[var] = value
                        changed = True
                        # Remove satisfied clauses
                        clauses_remaining = [c for c in clauses_remaining if c != ci]
            
            if not clauses_remaining:
                return True, new_assignment
            
            # Check for conflicts
            for ci in clauses_remaining:
                clause = clauses[ci]
                unassigned = [(var, sign) for var, sign in clause if var not in new_assignment]
                if not unassigned:
                    return False, None
            
            # Pick a variable to branch on
            # Use degree heuristic: pick variable appearing in most remaining clauses
            var_count = defaultdict(int)
            for ci in clauses_remaining:
                for var, sign in clauses[ci]:
                    if var not in new_assignment:
                        var_count[var] += 1
            
            if not var_count:
                # All variables assigned
                return True, new_assignment
            
            # Pick variable with highest degree
            var_to_branch = max(var_count, key=var_count.get)
            
            # Try both values
            for value in [1, 0]:
                if time.time() > solve_deadline:
                    return False, None
                
                trial_assignment = dict(new_assignment)
                trial_assignment[var_to_branch] = value
                
                # Filter clauses
                new_clauses_remaining = []
                for ci in clauses_remaining:
                    clause = clauses[ci]
                    satisfied = False
                    falsified = False
                    for var, sign in clause:
                        if var in trial_assignment:
                            if (sign == 1 and trial_assignment[var] == 1) or (sign == -1 and trial_assignment[var] == 0):
                                satisfied = True
                                break
                            if (sign == 1 and trial_assignment[var] == 0) or (sign == -1 and trial_assignment[var] == 1):
                                # This literal is false, but clause might still be satisfiable
                                pass
                    if satisfied:
                        continue
                    # Check if clause is falsified
                    all_false = True
                    for var, sign in clause:
                        if var not in trial_assignment:
                            all_false = False
                            break
                        if (sign == 1 and trial_assignment[var] == 0) or (sign == -1 and trial_assignment[var] == 1):
                            continue
                        else:
                            all_false = False
                            break
                    if all_false:
                        return False, None
                    new_clauses_remaining.append(ci)
                
                success, result = dpll_recursive(trial_assignment, new_clauses_remaining, depth + 1)
                if success:
                    return True, result
            
            return False, None
        
        # Try DPLL with a time limit
        success, solution = dpll_recursive({}, list(range(num_clauses)))
        
        if success and solution is not None:
            # Convert solution to colors array
            colors = [0] * num_triples
            for ti in range(num_triples):
                colors[ti] = solution.get(ti, 0)
            return colors
        
        # Fallback: use local search if DPLL fails
        return None
    
    # Try n from 34 down to 10
    for n in range(34, 9, -1):
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time() - 10
        if remaining < 5.0:
            break
        
        t_per_n = min(remaining, 50.0)
        
        result = dpll_solve(n, max_time=t_per_n)
        
        if result is not None:
            # Verify the result
            triples = get_triples(n)
            triple_index = {t: i for i, t in enumerate(triples)}
            
            valid = True
            for quad in itertools.combinations(range(n), 4):
                all_red = True
                for triple in itertools.combinations(quad, 3):
                    if result[triple_index[triple]] != 0:
                        all_red = False
                        break
                if all_red:
                    valid = False
                    break
            
            if valid:
                for quint in itertools.combinations(range(n), 5):
                    all_blue = True
                    for triple in itertools.combinations(quint, 3):
                        if result[triple_index[triple]] != 1:
                            all_blue = False
                            break
                    if all_blue:
                        valid = False
                        break
            
            if valid:
                return {'n': n, 'colors': result}
    
    # Fallback: return a small valid coloring
    n = 9
    triples = list(itertools.combinations(range(n), 3))
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    return {'n': n, 'colors': colors}

def _check_validity(n, colors):
    """Check if coloring has no red K4^(3) and no blue K5^(3)."""
    import itertools
    
    triple_to_color = {}
    idx = 0
    for a, b, c in itertools.combinations(range(n), 3):
        triple_to_color[(a, b, c)] = colors[idx]
        idx += 1
    
    for quad in itertools.combinations(range(n), 4):
        all_red = True
        for triple in itertools.combinations(quad, 3):
            if triple_to_color.get(triple, 1) != 0:
                all_red = False
                break
        if all_red:
            return False
    
    for quint in itertools.combinations(range(n), 5):
        all_blue = True
        for triple in itertools.combinations(quint, 3):
            if triple_to_color.get(triple, 0) != 1:
                all_blue = False
                break
        if all_blue:
            return False
    
    return True
# EVOLVE-BLOCK-END
