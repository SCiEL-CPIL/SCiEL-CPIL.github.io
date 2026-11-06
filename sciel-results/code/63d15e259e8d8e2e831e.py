"""SAT-based construction for K4-free graphs with alpha <= 5."""
import itertools
import sys
import os
import subprocess
import tempfile
import time

def construct():
    # We will try to construct a graph on n vertices that is K4-free and has alpha <= 5.
    # We use a SAT-based approach to extend a base graph.
    # Since we can't guarantee a SAT solver is available, we implement a DPLL solver.
    
    # First, let's try to find a large graph. The known record is 35 vertices.
    # We'll try to build up to 35 or 36 vertices.
    
    # Strategy: 
    # 1. Start with a small valid graph (e.g., Paley graph on 17 vertices, or a known small one).
    # 2. Incrementally add vertices, using SAT to determine edges for the new vertex.
    # 3. If we get stuck, backtrack or try a different base.
    
    # Actually, a simpler approach for SAT:
    # For a fixed n, we can encode the entire problem as SAT.
    # Variables: x_{i,j} for 0 <= i < j < n, indicating whether edge (i,j) exists.
    # Constraints:
    # 1. No K4: For every 4-clique {a,b,c,d}, at least one edge is missing.
    #    Clause: (¬x_{a,b} ∨ ¬x_{a,c} ∨ ¬x_{a,d} ∨ ¬x_{b,c} ∨ ¬x_{b,d} ∨ ¬x_{c,d})
    # 2. No independent set of size 6: For every 6-set {a,b,c,d,e,f}, at least one edge exists.
    #    Clause: (x_{a,b} ∨ x_{a,c} ∨ x_{a,d} ∨ x_{a,e} ∨ x_{a,f} ∨ x_{b,c} ∨ x_{b,d} ∨ x_{b,e} ∨ x_{b,f} ∨ x_{c,d} ∨ x_{c,e} ∨ x_{c,f} ∨ x_{d,e} ∨ x_{d,f} ∨ x_{e,f})
    
    # This encoding has O(n^4) clauses for K4-free and O(n^6) clauses for alpha<=5.
    # For n=35, that's ~35^6/6 ≈ 1.8 million clauses, which is too many for a simple DPLL.
    
    # Better approach: incremental vertex addition.
    # Start with a known valid graph on m vertices.
    # Add vertex v. We need to determine its adjacency to the m existing vertices.
    # This is a SAT problem with m variables (edges from v to existing vertices).
    # Constraints:
    # 1. No K4: For any triangle {a,b,c} in the existing graph, v cannot be adjacent to all three.
    #    Also, for any edge {a,b} and any vertex c adjacent to both a and b, etc.
    #    More precisely: For any set of 3 vertices {a,b,c} that form a triangle, v cannot connect to all 3.
    #    For any set of 2 vertices {a,b} that are adjacent, and any vertex c such that {v,a,b,c} would form K4... 
    #    Actually, K4 involving v: v plus 3 existing vertices {a,b,c} such that a,b,c form a triangle and v is adjacent to all.
    #    So: For every triangle {a,b,c} in G, not (v~a and v~b and v~c).
    #    Clause: (¬x_a ∨ ¬x_b ∨ ¬x_c) for each triangle {a,b,c}.
    #    
    # 2. No independent set of size 6: Any independent set of size 6 in the new graph must involve v.
    #    So: For any set of 5 existing vertices {a,b,c,d,e} that form an independent set in G,
    #    v must be adjacent to at least one of them.
    #    Clause: (x_a ∨ x_b ∨ x_c ∨ x_d ∨ x_e) for each independent 5-set {a,b,c,d,e} in G.
    
    # This is much smaller! For m=34, number of triangles and independent 5-sets is manageable.
    
    # Let's implement a DPLL solver for this incremental approach.
    
    def dpll(clauses, num_vars):
        """Simple DPLL SAT solver. clauses is a list of lists of integers (literals).
        Positive literal i means x_i is true, negative -i means x_i is false."""
        # Unit propagation
        while True:
            # Check for empty clause (contradiction)
            for clause in clauses:
                if len(clause) == 0:
                    return None
            # Check for unit clauses
            changed = True
            while changed:
                changed = False
                for i, clause in enumerate(clauses):
                    if len(clause) == 1:
                        lit = clause[0]
                        var = abs(lit)
                        val = lit > 0
                        # Assign var = val
                        new_clauses = []
                        for c in clauses:
                            if lit in c:
                                continue  # clause satisfied
                            elif -lit in c:
                                continue  # clause becomes empty, but we already checked for empty
                            else:
                                new_clauses.append(c)
                        # Actually, we need to be more careful. Let's restart with assignment.
                        # This simple version has issues. Let's use a more standard DPLL.
                        break
                else:
                    break
            
            # If we found a unit clause, handle it properly
            unit_found = False
            for i, clause in enumerate(clauses):
                if len(clause) == 1:
                    lit = clause[0]
                    var = abs(lit)
                    val = lit > 0
                    # Simplify clauses with this assignment
                    new_clauses = []
                    for c in clauses:
                        if lit in c:
                            continue  # satisfied
                        else:
                            new_c = [l for l in c if l != -lit]
                            new_clauses.append(new_c)
                    clauses = new_clauses
                    unit_found = True
                    break
            
            if not unit_found:
                break
        
        # Check for empty clause
        for clause in clauses:
            if len(clause) == 0:
                return None
        
        # If no clauses, all satisfied
        if not clauses:
            return {}
        
        # Choose a variable
        # Pick the variable appearing in the most clauses
        var_counts = {}
        for clause in clauses:
            for lit in clause:
                var = abs(lit)
                var_counts[var] = var_counts.get(var, 0) + 1
        
        if not var_counts:
            return {}
        
        var = max(var_counts, key=var_counts.get)
        
        # Try true
        assignment = dpll_with_assignment(clauses, var, True)
        if assignment is not None:
            return assignment
        
        # Try false
        assignment = dpll_with_assignment(clauses, var, False)
        if assignment is not None:
            return assignment
        
        return None
    
    def dpll_with_assignment(clauses, var, val):
        """DPLL with a specific assignment for var."""
        lit = var if val else -var
        new_clauses = []
        for c in clauses:
            if lit in c:
                continue  # satisfied
            else:
                new_c = [l for l in c if l != -lit]
                new_clauses.append(new_c)
        
        return dpll(new_clauses, var)
    
    # This DPLL is too slow for large instances. Let's use a different approach.
    # Since the incremental SAT problem is small (m variables, O(m^3) clauses),
    # we can use a brute-force or smarter search for small m.
    
    # Actually, for m up to ~30, the number of triangles and independent 5-sets might be manageable.
    # Let's try a different strategy: use a known large graph and verify it.
    
    # The Paley graph on 17 vertices is K4-free and has alpha=6? Let's check.
    # Actually, the seed program uses Paley on 17, which scores 1.0 (n=17).
    
    # Let's try to construct a graph on 35 vertices using a known construction or SAT.
    
    # Given the time constraint (120s), let's try a heuristic approach with local search,
    # or use a known graph.
    
    # Actually, let's try to use the SAT approach but with a better solver.
    # For the incremental step, we have at most ~34 variables and maybe a few thousand clauses.
    # A simple backtracking with clause learning might work.
    
    # Let's implement a proper DPLL with unit propagation and pure literal elimination.
    
    def sat_solver(clauses):
        """DPLL with unit propagation."""
        clauses = [list(c) for c in clauses]
        
        def propagate(clauses):
            """Unit propagation. Returns (clauses, assignments) or None if contradiction."""
            assignments = {}
            changed = True
            while changed:
                changed = False
                i = 0
                while i < len(clauses):
                    clause = clauses[i]
                    if len(clause) == 0:
                        return None
                    if len(clause) == 1:
                        lit = clause[0]
                        var = abs(lit)
                        val = lit > 0
                        if var in assignments and assignments[var] != val:
                            return None
                        assignments[var] = val
                        # Remove clause and simplify others
                        clauses.pop(i)
                        new_clauses = []
                        for c in clauses:
                            if lit in c:
                                continue
                            else:
                                new_c = [l for l in c if l != -lit]
                                if len(new_c) == 0:
                                    return None
                                new_clauses.append(new_c)
                        clauses = new_clauses
                        changed = True
                        i = 0
                    else:
                        i += 1
            return clauses, assignments
        
        def solve(clauses, assignments):
            result = propagate(clauses)
            if result is None:
                return None
            clauses, new_assignments = result
            assignments.update(new_assignments)
            
            if not clauses:
                return assignments
            
            # Choose variable with highest occurrence
            var_counts = {}
            for clause in clauses:
                for lit in clause:
                    var = abs(lit)
                    var_counts[var] = var_counts.get(var, 0) + 1
            
            if not var_counts:
                return assignments
            
            var = max(var_counts, key=var_counts.get)
            
            # Try true
            lit = var
            new_clauses = []
            for c in clauses:
                if lit in c:
                    continue
                else:
                    new_c = [l for l in c if l != -lit]
                    new_clauses.append(new_c)
            
            sol = solve(new_clauses, dict(assignments, **{var: True}))
            if sol is not None:
                return sol
            
            # Try false
            lit = -var
            new_clauses = []
            for c in clauses:
                if lit in c:
                    continue
                else:
                    new_c = [l for l in c if l != -lit]
                    new_clauses.append(new_c)
            
            sol = solve(new_clauses, dict(assignments, **{var: False}))
            if sol is not None:
                return sol
            
            return None
        
        return solve(clauses, {})
    
    # Start with a small valid graph
    # Use Paley graph on 17 vertices as base
    n_base = 17
    squares = {x * x % n_base for x in range(1, n_base)}
    edges = set()
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if (j - i) % n_base in squares:
                edges.add((i, j))
    
    n = n_base
    edge_set = edges
    
    # Try to add vertices one by one
    max_n = 36
    for target_n in range(n_base + 1, max_n + 1):
        m = n  # current number of vertices
        new_vertex = m
        
        # Build adjacency matrix for current graph
        adj = [[False] * m for _ in range(m)]
        for u, v in edge_set:
            adj[u][v] = True
            adj[v][u] = True
        
        # Find all triangles in current graph
        triangles = []
        for a in range(m):
            for b in range(a + 1, m):
                if adj[a][b]:
                    for c in range(b + 1, m):
                        if adj[a][c] and adj[b][c]:
                            triangles.append((a, b, c))
        
        # Find all independent 5-sets in current graph
        independent_5sets = []
        for combo in itertools.combinations(range(m), 5):
            is_independent = True
            for i in range(5):
                for j in range(i + 1, 5):
                    if adj[combo[i]][combo[j]]:
                        is_independent = False
                        break
                if not is_independent:
                    break
            if is_independent:
                independent_5sets.append(combo)
        
        # Build SAT clauses
        # Variables: x_0, ..., x_{m-1} where x_i indicates edge between new_vertex and i
        num_vars = m
        clauses = []
        
        # K4 constraint: for each triangle (a,b,c), not (x_a and x_b and x_c)
        for a, b, c in triangles:
            clauses.append([-1 - a, -1 - b, -1 - c])  # Wait, let's use 1-indexed or 0-indexed consistently
            # Let's use 1-indexed: variable i+1 for vertex i
            # Actually, let's use 0-indexed in the solver for simplicity
            # Literal: positive i means x_i = True, negative -(i+1) means x_i = False
            # Let's redefine: use literals where positive value = variable index + 1
            pass
        
        # Let me redo the SAT encoding with 1-indexed literals
        clauses = []
        
        # K4 constraint: for each triangle (a,b,c), not all three edges to new vertex
        for a, b, c in triangles:
            # Clauses: (¬x_a ∨ ¬x_b ∨ ¬x_c)
            # Using 1-indexed: literal for x_i is (i+1), negation is -(i+1)
            clauses.append([-(a+1), -(b+1), -(c+1)])
        
        # Alpha constraint: for each independent 5-set (a,b,c,d,e), at least one edge to new vertex
        for a, b, c, d, e in independent_5sets:
            clauses.append([(a+1), (b+1), (c+1), (d+1), (e+1)])
        
        # Solve SAT
        if clauses:
            solution = sat_solver(clauses)
        else:
            solution = {}
        
        if solution is not None:
            # Add edges from new vertex to existing vertices based on solution
            for i in range(m):
                if solution.get(i+1, False):
                    edge_set.add((new_vertex, i))
                    edge_set.add((i, new_vertex))
            n += 1
        else:
            # Can't add this vertex, try next target or break
            break
    
    # Convert edge_set to list of edges
    edge_list = []
    for u, v in edge_set:
        if u < v:
            edge_list.append([u, v])
    
    edge_list.sort()
    
    return {'n': n, 'edges': edge_list}
