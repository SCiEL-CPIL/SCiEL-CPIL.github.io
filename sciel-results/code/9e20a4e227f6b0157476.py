"""Rewritten program: SAT-based greedy edge selection for minimal non-2-colorable 5-uniform hypergraph."""
import random
from itertools import combinations

def construct():
    """
    Construct a 5-uniform hypergraph with no proper 2-colouring, minimizing edges.
    
    Approach: Model as SAT where each vertex is a Boolean variable.
    Each edge {v1,...,v5} forbids all-true and all-false, i.e., adds clauses:
      (not v1 or not v2 or not v3 or not v4 or not v5)  # forbids all-true
      (v1 or v2 or v3 or v4 or v5)                      # forbids all-false
    
    Greedy: iteratively add edges that maximize reduction in solution space,
    until the formula becomes unsatisfiable.
    """
    random.seed(42)
    
    # We'll work with a fixed number of vertices. Start with a reasonable n.
    # The best known construction has 51 edges. Let's try to find something competitive.
    # We'll use n=16 vertices as a starting point (enough vertices for interesting structure).
    
    def count_solutions_with_propagation(n, edges, max_solutions=100000):
        """
        Count the number of satisfying assignments using backtracking with unit propagation.
        Returns the count (capped at max_solutions for efficiency).
        """
        if n > 26:
            return 0
        
        # Variables are 0..n-1
        # Each edge contributes two clauses:
        # Clause 1: not all vertices in edge are True  -> (¬v1 ∨ ¬v2 ∨ ¬v3 ∨ ¬v4 ∨ ¬v5)
        # Clause 2: not all vertices in edge are False -> (v1 ∨ v2 ∨ v3 ∨ v4 ∨ v5)
        
        # Represent clauses as lists of literals.
        # Literal: (var, sign) where sign=True means positive, False means negative.
        clauses = []
        for edge in edges:
            # Clause forbidding all-true: at least one vertex is False
            for v in edge:
                clauses.append((v, False))  # This isn't right; a clause is a disjunction
            # Actually, let me restructure: each clause is a list of literals
            clauses = []
            for edge in edges:
                # Forbid all-true: (¬v1 ∨ ¬v2 ∨ ¬v3 ∨ ¬v4 ∨ ¬v5)
                clauses.append([(v, False) for v in edge])
                # Forbid all-false: (v1 ∨ v2 ∨ v3 ∨ v4 ∨ v5)
                clauses.append([(v, True) for v in edge])
        
        # Backtracking with unit propagation
        assignment = {}  # var -> True/False
        
        def propagate(clauses, assignment):
            """
            Apply unit propagation. Returns (new_assignment, remaining_clauses, contradiction)
            """
            changed = True
            while changed:
                changed = False
                remaining = []
                for clause in clauses:
                    # Check if clause is satisfied
                    satisfied = False
                    # Check for unit clause
                    unassigned = []
                    for (v, sign) in clause:
                        if v in assignment:
                            if assignment[v] == sign:
                                satisfied = True
                                break
                            # else: literal is False, ignore
                        else:
                            unassigned.append((v, sign))
                    
                    if satisfied:
                        continue
                    if len(unassigned) == 0:
                        # Clause is falsified
                        return None, None, True
                    if len(unassigned) == 1:
                        v, sign = unassigned[0]
                        assignment[v] = sign
                        changed = True
                        continue
                    remaining.append(unassigned)
            return assignment, remaining, False
        
        # Initial propagation
        assignment, clauses, contradiction = propagate(clauses, assignment)
        if contradiction:
            return 0
        
        if not clauses:
            # All clauses satisfied, count free variables
            free_vars = [v for v in range(n) if v not in assignment]
            return min(2 ** len(free_vars), max_solutions)
        
        count = 0
        
        def backtrack(assignment, clauses):
            nonlocal count
            if count >= max_solutions:
                return
            
            if not clauses:
                free_vars = [v for v in range(n) if v not in assignment]
                count += min(2 ** len(free_vars), max_solutions - count)
                return
            
            # Find the variable to branch on
            # Pick the first unassigned variable
            var = None
            for clause in clauses:
                for (v, sign) in clause:
                    if v not in assignment:
                        var = v
                        break
                if var is not None:
                    break
            
            if var is None:
                # Should not happen if clauses exist
                free_vars = [v for v in range(n) if v not in assignment]
                count += min(2 ** len(free_vars), max_solutions - count)
                return
            
            # Try var = True
            new_assignment = dict(assignment)
            new_assignment[var] = True
            new_clauses = [list(c) for c in clauses]
            _, new_clauses, contradiction = propagate(new_clauses, new_assignment)
            if not contradiction:
                backtrack(new_assignment, new_clauses)
            
            # Try var = False
            new_assignment = dict(assignment)
            new_assignment[var] = False
            new_clauses = [list(c) for c in clauses]
            _, new_clauses, contradiction = propagate(new_clauses, new_assignment)
            if not contradiction:
                backtrack(new_assignment, new_clauses)
        
        backtrack(assignment, clauses)
        return count
    
    def is_unsat(n, edges):
        """Check if the hypergraph is non-2-colorable (SAT instance is unsatisfiable)."""
        return count_solutions_with_propagation(n, edges, max_solutions=1) == 0
    
    # Greedy construction
    # Start with a pool of candidate edges
    n = 16  # Number of vertices
    all_edges = list(combinations(range(n), 5))
    random.shuffle(all_edges)
    
    selected_edges = []
    # We want to minimize edges, so greedily pick edges that most reduce solution count
    
    # Initial solution count (no edges: all 2^n assignments work)
    current_count = 2 ** n
    
    # Precompute which edges are candidates
    # For efficiency, we'll evaluate a subset of candidates at each step
    
    max_edges = 60  # We expect to find a solution well under 51
    
    while len(selected_edges) < max_edges:
        # Evaluate a sample of candidate edges
        candidates = [e for e in all_edges if e not in selected_edges]
        if not candidates:
            break
        
        # Sample up to 50 candidates for evaluation
        sample_size = min(50, len(candidates))
        sample = random.sample(candidates, sample_size)
        
        best_edge = None
        best_count = current_count
        
        for edge in sample:
            new_edges = selected_edges + [edge]
            new_count = count_solutions_with_propagation(n, new_edges, max_solutions=current_count)
            if new_count < best_count:
                best_count = new_count
                best_edge = edge
        
        if best_edge is None or best_count >= current_count:
            # No improvement found in sample; try all remaining if few left
            if len(candidates) <= 100:
                for edge in candidates:
                    new_edges = selected_edges + [edge]
                    new_count = count_solutions_with_propagation(n, new_edges, max_solutions=current_count)
                    if new_count < best_count:
                        best_count = new_count
                        best_edge = edge
            
            if best_edge is None:
                break
        
        selected_edges.append(best_edge)
        current_count = best_count
        
        if current_count == 0:
            break
    
    # Check if we achieved unsatisfiability
    if current_count > 0:
        # Try with more vertices or different strategy
        # Fall back to a known construction approach
        # For now, return what we have if it's unsat, otherwise try n=15
        pass
    
    # Verify and return
    final_count = count_solutions_with_propagation(n, selected_edges, max_solutions=1)
    if final_count == 0:
        # Convert tuples to lists
        edges_list = [list(e) for e in selected_edges]
        return {'n': n, 'edges': edges_list}
    
    # Fallback: use a known small construction if available
    # For now, return the complete 5-uniform on 9 vertices as a safe fallback
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
