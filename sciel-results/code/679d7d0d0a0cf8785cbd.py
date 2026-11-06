    var_to_clauses = [[] for _ in range(n)]
    var_lit_sign = [[None] * len(clauses) for _ in range(n)] # Or dict? List of lists is faster if dense.
    # Actually, sparse is better.
    # Let's just use a dict of dicts or list of dicts.
    # var_to_clauses[var_idx] is a list of c_idx.
    # We can parallel store signs.
    
    # Simpler: Just keep the loop. 3-SAT clauses are tiny.
