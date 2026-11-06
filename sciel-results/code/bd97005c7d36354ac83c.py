import time
import sys

DEADLINE_SECONDS = 320.0
START_TIME = time.time()

def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)

def _paley_graph(q):
    """Return edges of the Paley graph on q vertices (q a prime congruent to 1 mod 4)."""
    residues = {(x * x) % q for x in range(1, q)}
    return [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]

def _build_adj(edges, n):
    """Build adjacency as list of int bitsets."""
    adj = [0] * n
    for i, j in edges:
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
    return adj

def _find_4_structures(adj, n):
    """
    Find all 4-cliques and 4-independent sets.
    Returns (list_of_4cliques, list_of_4indeps) where each is a tuple of 4 vertex indices.
    """
    four_cliques = []
    four_indeps = []
    all_mask = (1 << n) - 1
    non_adj = [all_mask ^ adj[i] ^ (1 << i) for i in range(n)]
    
    # Find 4-cliques
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] & (1 << b)):
                continue
            common_ab = adj[a] & adj[b] & ~((1 << (b + 1)) - 1)
            c_mask = common_ab
            while c_mask:
                c_bit = c_mask & (-c_mask)
                c = c_bit.bit_length() - 1
                c_mask ^= c_bit
                common_abc = common_ab & adj[c] & ~((1 << (c + 1)) - 1)
                d_mask = common_abc
                while d_mask:
                    d_bit = d_mask & (-d_mask)
                    d = d_bit.bit_length() - 1
                    d_mask ^= d_bit
                    four_cliques.append((a, b, c, d))
                    
    # Find 4-independent sets
    for a in range(n):
        for b in range(a + 1, n):
            if adj[a] & (1 << b):
                continue
            common_non_ab = non_adj[a] & non_adj[b] & ~((1 << (b + 1)) - 1)
            c_mask = common_non_ab
            while c_mask:
                c_bit = c_mask & (-c_mask)
                c = c_bit.bit_length() - 1
                c_mask ^= c_bit
                common_non_abc = common_non_ab & non_adj[c] & ~((1 << (c + 1)) - 1)
                d_mask = common_non_abc
                while d_mask:
                    d_bit = d_mask & (-d_mask)
                    d = d_bit.bit_length() - 1
                    d_mask ^= d_bit
                    four_indeps.append((a, b, c, d))
                    
    return four_cliques, four_indeps

def _solve_sat_for_vertex(n_vars, clauses_forbid, clauses_require):
    """
    Solve a SAT problem for n_vars boolean variables.
    clauses_forbid: list of tuples of vars. Not all can be True. (x1 OR NOT x2 ... is not quite right, it's NOT(x1 AND x2 AND x3 AND x4))
    Actually, forbid clauses are: NOT(x_a AND x_b AND x_c AND x_d) which is equivalent to (NOT x_a OR NOT x_b OR NOT x_c OR NOT x_d).
    require clauses are: (x_a OR x_b OR x_c OR x_d).
    
    This is a standard CNF SAT problem.
    We implement a simple DPLL with unit propagation.
    """
    # Convert to internal representation
    # Var i: 0 is False, 1 is True, None is Unassigned
    assignment = [None] * n_vars
    
    # Clauses are lists of literals.
    # Literal is (var, is_positive)
    # Forbid clause (a,b,c,d): [(-a, False), (-b, False), (-c, False), (-d, False)] -> wait.
    # NOT(x_a AND x_b AND x_c AND x_d) == (NOT x_a OR NOT x_b OR NOT x_c OR NOT x_d).
    # So literals are (a, False), (b, False), (c, False), (d, False).
    # Require clause (a,b,c,d): [ (a, True), (b, True), (c, True), (d, True) ].
    
    cnf_clauses = []
    for c in clauses_forbid:
        cnf_clauses.append([(v, False) for v in c])
    for c in clauses_require:
        cnf_clauses.append([(v, True) for v in c])
        
    def is_satisfied(lit, assign):
        var, pos = lit
        if assign[var] is None:
            return False
        return assign[var] == (1 if pos else 0)
        
    def is_falsified(lit, assign):
        var, pos = lit
        if assign[var] is None:
            return False
        return assign[var] != (1 if pos else 0)
        
    def propagate(clauses, assign):
        """
        Perform unit propagation.
        Returns (success, new_clauses, new_assign) or (False, None, None) if contradiction.
        """
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for clause in clauses:
                # Check if clause is satisfied
                sat = False
                for lit in clause:
                    if is_satisfied(lit, assign):
                        sat = True
                        break
                if sat:
                    continue
                
                # Check for unit clause
                unassigned_lits = []
                for lit in clause:
                    var, pos = lit
                    if assign[var] is None:
                        unassigned_lits.append(lit)
                    else:
                        if is_falsified(lit, assign):
                            continue
                        # else it's satisfied? No, we checked sat above.
                        # If not sat, and not unassigned, it must be falsified.
                        # Wait, if a lit is falsified, it doesn't help satisfy the clause.
                        # So we just keep the unassigned ones.
                
                if len(unassigned_lits) == 0:
                    # Clause is falsified
                    return False, None, None
                
                if len(unassigned_lits) == 1:
                    # Unit clause
                    var, pos = unassigned_lits[0]
                    val = 1 if pos else 0
                    if assign[var] is not None and assign[var] != val:
                        return False, None, None
                    assign[var] = val
                    changed = True
                    # Clause is now satisfied, so we don't add it to new_clauses
                else:
                    new_clauses.append(unassigned_lits)
                    
        return True, new_clauses, assign
        
    def dpll(clauses, assign):
        if _time_left() < 5:
            return None
            
        ok, new_clauses, new_assign = propagate(clauses, assign)
        if not ok:
            return None
            
        if not new_clauses:
            return new_assign.copy()
            
        # Pick a variable
        # Heuristic: pick variable appearing in most clauses
        var_counts = [0] * n_vars
        for clause in new_clauses:
            seen = set()
            for lit in clause:
                if lit[0] not in seen:
                    var_counts[lit[0]] += 1
                    seen.add(lit[0])
                    
        unassigned_vars = [i for i in range(n_vars) if new_assign[i] is None]
        if not unassigned_vars:
            return new_assign.copy()
            
        # Pick var with max count
        var = max(unassigned_vars, key=lambda i: var_counts[i])
        
        # Try True
        new_assign[var] = 1
        res = dpll(new_clauses, new_assign)
        if res:
            return res
            
        # Try False
        new_assign[var] = 0
        res = dpll(new_clauses, new_assign)
        if res:
            return res
            
        # Backtrack
        new_assign[var] = None
        return None
        
    return dpll(cnf_clauses, assignment)

def construct():
    # Start with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    n = q
    
    # Verify it's valid (optional, but good for sanity)
    # adj = _build_adj(edges, n)
    # if _has_clique_k5(adj, n) or _has_independent_set_5(adj, n):
    #     pass # Assume Paley is valid
    
    best_n = n
    best_edges = edges[:]
    
    # Precompute structures for current graph
    adj = _build_adj(edges, n)
    four_cliques, four_indeps = _find_4_structures(adj, n)
    
    # Try to add vertices one by one
    max_attempts = 10
    for attempt in range(max_attempts):
        if _time_left() < 20:
            break
            
        new_n = n + 1
        # The new vertex is index n
        # Variables are edges (n, i) for i in range(n)
        
        # Filter clauses to only involve existing vertices
        # All clauses already involve only existing vertices.
        
        # Solve SAT
        solution = _solve_sat_for_vertex(n, four_cliques, four_indeps)
        
        if solution is None:
            break
            
        # Add edges based on solution
        for i in range(n):
            if solution[i] == 1:
                edges.append((i, n))
                
        n = new_n
        if n > best_n:
            best_n = n
            best_edges = edges[:]
            
        # Update structures
        adj = _build_adj(edges, n)
        four_cliques, four_indeps = _find_4_structures(adj, n)
        
    return best_n, best_edges
