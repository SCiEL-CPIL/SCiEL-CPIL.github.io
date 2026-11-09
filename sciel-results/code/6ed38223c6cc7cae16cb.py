"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Convert to 0-based: each clause is list of (var_idx, is_positive)
    def to_int(clause):
        return [(abs(l) - 1, l > 0) for l in clause]
    
    int_clauses = [to_int(c) for c in clauses]
    
    # DPLL with unit propagation
    assignment = [None] * n  # None, True, False
    
    def lit_sat(var, pos):
        return assignment[var] is not None and assignment[var] == pos
    
    def lit_unsat(var, pos):
        return assignment[var] is not None and assignment[var] != pos
    
    def lit_unassigned(var):
        return assignment[var] is None
    
    def propagate():
        """Unit propagation. Returns True if consistent, False if conflict."""
        changed = True
        while changed:
            changed = False
            for ci, clause in enumerate(int_clauses):
                # Count assigned and check for conflict
                unassigned = 0
                falsified = 0
                for var, pos in clause:
                    if lit_sat(var, pos):
                        break  # Clause satisfied
                    elif lit_unsat(var, pos):
                        falsified += 1
                    else:
                        unassigned += 1
                else:
                    # No satisfied literal
                    if falsified == len(clause):
                        return False  # Conflict
                    if unassigned == 1:
                        # Unit clause
                        for var, pos in clause:
                            if lit_unassigned(var):
                                assignment[var] = pos
                                changed = True
                                break
        return True
    
    def all_assigned():
        return all(a is not None for a in assignment)
    
    def select_var():
        """Select unassigned variable with highest occurrence count."""
        best_var = -1
        best_count = -1
        for v in range(n):
            if assignment[v] is None:
                count = sum(1 for c in int_clauses for var, pos in c if var == v)
                if count > best_count:
                    best_count = count
                    best_var = v
        return best_var
    
    def get_polarity(var):
        pos = sum(1 for c in int_clauses for v, p in c if v == var and p)
        neg = sum(1 for c in int_clauses for v, p in c if v == var and not p)
        return pos >= neg
    
    def dpll():
        if not propagate():
            return None
        if all_assigned():
            return list(assignment)
        
        var = select_var()
        if var == -1:
            return None
        
        # Try preferred polarity first
        for pol in [get_polarity(var), not get_polarity(var)]:
            assignment[var] = pol
            result = dpll()
            if result is not None:
                return result
            assignment[var] = None
        
        return None
    
    # Try DPLL
    result = dpll()
    
    if result is not None:
        return result
    
    # Fallback: WalkSAT with restarts
    rng = random.Random(123)
    for _ in range(50):
        assignment = [rng.random() < 0.5 for _ in range(n)]
        for _ in range(100000):
            unsat = [c for c in clauses if not any(assignment[abs(l)-1] == (l>0) for l in c)]
            if not unsat:
                return assignment
            clause = rng.choice(unsat)
            # Pick literal that maximizes number of satisfied clauses
            best_l, best_score = None, -1
            for l in clause:
                v = abs(l) - 1
                new_val = (l > 0)
                if assignment[v] == new_val:
                    continue
                score = 0
                for c in clauses:
                    if any(assignment[abs(ll)-1] == (ll>0) for ll in c):
                        continue
                    if any(abs(ll)-1 == v and (ll>0) == new_val for ll in c):
                        score += 1
                if score > best_score:
                    best_score = score
                    best_l = l
            if best_l is None:
                best_l = rng.choice(clause)
            assignment[abs(best_l)-1] = (best_l > 0)
    
    return assignment
# EVOLVE-BLOCK-END
