"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Deadline guard: must finish well before 120s (evaluator kills at 150s)
_START_TIME = time.time()
DEADLINE = _START_TIME + 110  # 10s safety margin

def _time_left():
    return DEADLINE - time.time()

def _is_not_2_colorable(n, edges):
    """
    Check if a 5-uniform hypergraph has NO proper 2-coloring.
    A proper 2-coloring means every edge has both colors present (not monochromatic).
    We use backtracking with unit propagation on the constraint:
      For each edge e, NOT(all same color) i.e., at least one vertex is 0 AND at least one is 1.
    
    Actually, "no proper 2-coloring" means: there is NO assignment of 2 colors such that
    every edge contains both colors. Equivalently, for every 2-coloring, some edge is monochromatic.
    
    We check satisfiability of: for all edges, the edge is NOT monochromatic.
    If UNSAT, then no proper 2-coloring exists.
    
    Use DPLL with unit propagation.
    Variables: x_0, ..., x_{n-1} in {0,1}.
    Constraint per edge e = {v1,...,v5}: 
      (x_v1 OR x_v2 OR x_v3 OR x_v4 OR x_v5) AND 
      (NOT x_v1 OR NOT x_v2 OR NOT x_v3 OR NOT x_v4 OR NOT x_v5)
    
    This is a 2-SAT-like structure but with 5-clauses. We do backtracking.
    """
    # Build clauses as tuples of (var, sign) where sign=1 means var, sign=-1 means not var
    # Clause 1: at least one true -> (x_v1 OR x_v2 OR ... OR x_v5)
    # Clause 2: at least one false -> (NOT x_v1 OR NOT x_v2 OR ... OR NOT x_v5)
    
    clauses = []
    for e in edges:
        # All positive
        clauses.append(tuple((v, 1) for v in e))
        # All negative
        clauses.append(tuple((v, -1) for v in e))
    
    # Backtracking solver with unit propagation
    def solve(assign):
        # assign: dict var -> 0 or 1, or None for unassigned
        # Check for contradictions and unit propagation
        # We'll work with a set of remaining clauses
        
        # First, filter clauses by current assignment
        remaining = []
        for clause in clauses:
            satisfied = False
            all_false = True
            new_clause = []
            for (v, s) in clause:
                if v in assign:
                    val = assign[v]
                    # Clause literal is satisfied if (s==1 and val==1) or (s==-1 and val==0)
                    lit_val = val if s == 1 else (1 - val)
                    if lit_val == 1:
                        satisfied = True
                        break
                    else:
                        # This literal is false, continue
                        pass
                else:
                    new_clause.append((v, s))
            if satisfied:
                continue
            if len(new_clause) == 0:
                # Clause is falsified
                return None  # contradiction
            remaining.append(tuple(new_clause))
        
        # Unit propagation
        changed = True
        while changed:
            changed = False
            new_remaining = []
            for clause in remaining:
                if len(clause) == 1:
                    (v, s) = clause[0]
                    # Must assign v = 1 if s==1, v=0 if s==-1
                    val = 1 if s == 1 else 0
                    if v in assign:
                        if assign[v] != val:
                            return None  # contradiction
                    else:
                        assign[v] = val
                        changed = True
                        # This clause is satisfied, skip it
                        continue
                new_remaining.append(clause)
            remaining = new_remaining
        
        # Check if all assigned
        unassigned = [v for v in range(n) if v not in assign]
        if not unassigned:
            # Check all clauses are satisfied
            for clause in clauses:
                sat = False
                for (v, s) in clause:
                    val = assign[v]
                    lit_val = val if s == 1 else (1 - val)
                    if lit_val == 1:
                        sat = True
                        break
                if not sat:
                    return None
            return assign  # SAT
        
        # Pick a variable to branch on
        # Heuristic: pick variable appearing in most remaining clauses
        var_count = {}
        for clause in remaining:
            for (v, s) in clause:
                var_count[v] = var_count.get(v, 0) + 1
        if not var_count:
            return assign  # All clauses satisfied
        
        var = max(unassigned, key=lambda v: var_count.get(v, 0))
        
        # Try 1 then 0
        for val in [1, 0]:
            assign[var] = val
            result = solve(assign)
            if result is not None:
                return result
            del assign[var]
        
        return None
    
    result = solve({})
    if result is None:
        return True  # UNSAT -> no proper 2-coloring -> not 2-colorable
    else:
        return False  # SAT -> there IS a proper 2-coloring

def construct():
    """
    Find a small 5-uniform hypergraph with no proper 2-coloring.
    Strategy: 
    1. Start with known small constructions or random search.
    2. Use the fact that the complete hypergraph on 9 vertices works (126 edges).
    3. Try to reduce edges via random sampling and local search.
    4. Also try structured constructions.
    """
    best = None
    best_edges = None
    best_n = 9
    
    # Known: complete 5-uniform on 9 vertices
    from itertools import combinations as combos
    complete_9 = [list(e) for e in combos(range(9), 5)]
    # Verify
    if _is_not_2_colorable(9, complete_9):
        best_edges = complete_9
        best_n = 9
        best = len(complete_9)
    
    # Try to find smaller by random search on various n
    rng = random.Random(42)
    
    for n in range(9, 15):
        if _time_left() < 5:
            break
        # Try random hypergraphs with varying edge counts
        all_edges = list(combos(range(n), 5))
        num_all = len(all_edges)
        
        # Binary search / random sampling for minimum edges
        # Start with random subsets and check
        for trial in range(50):
            if _time_left() < 5:
                break
            
            # Try different sizes
            for e_count in [20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110, 120]:
                if e_count > num_all:
                    continue
                if e_count >= best:
                    continue
                
                edges = rng.sample(all_edges, e_count)
                if _is_not_2_colorable(n, edges):
                    if best is None or e_count < best:
                        best = e_count
                        best_edges = edges
                        best_n = n
                    break  # Found one at this size, try smaller
    
    # Local search: try to remove edges from best found
    if best_edges is not None:
        if _time_left() > 10:
            current = list(best_edges)
            current_n = best_n
            # Try to remove one edge at a time
            for _ in range(1000):
                if _time_left() < 5:
                    break
                if not current:
                    break
                idx = rng.randrange(len(current))
                candidate = current[:idx] + current[idx+1:]
                if _is_not_2_colorable(current_n, candidate):
                    current = candidate
                    best = len(current)
                    best_edges = current
                else:
                    pass
    
    # Also try structured: take complete on 9, randomly remove edges
    if _time_left() > 10:
        complete_9 = [list(e) for e in combos(range(9), 5)]
        current = list(complete_9)
        rng2 = random.Random(123)
        for _ in range(2000):
            if _time_left() < 5:
                break
            if len(current) < best:
                break
            idx = rng2.randrange(len(current))
            candidate = current[:idx] + current[idx+1:]
            if _is_not_2_colorable(9, candidate):
                current = candidate
                best = len(current)
                best_edges = current
                best_n = 9
            else:
                pass
    
    if best_edges is None:
        # Fallback: complete 5-uniform on 9 vertices
        best_edges = [list(e) for e in combos(range(9), 5)]
        best_n = 9
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
