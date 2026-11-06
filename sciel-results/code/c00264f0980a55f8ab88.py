def construct():
    """
    Returns a nonempty subset of [1, 180] with no three distinct elements a, b, c 
    satisfying 1/a = 1/b + 1/c. Maximizes cardinality.
    Uses Integer Linear Programming (ILP) to find the maximum independent set
    in the 3-uniform hypergraph of forbidden triples.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    N = 180
    all_numbers = list(range(1, N + 1))
    
    # 1. Identify all forbidden triples (a, b, c) with distinct elements
    # 1/a = 1/b + 1/c  =>  a = bc / (b+c)
    unique_triples = set()
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            denom = b + c
            num = b * c
            if num % denom == 0:
                a = num // denom
                if 1 <= a <= N and a != b and a != c:
                    unique_triples.add(tuple(sorted((a, b, c))))
    
    triples_list = list(unique_triples)
    num_triples = len(triples_list)
    
    # 2. Formulate ILP
    # Variables: x_i for i in 1..180 (index 0..179 in array)
    # Objective: Maximize sum(x_i) -> Minimize -sum(x_i)
    c = np.full(N, -1.0)
    
    # Constraints:
    # 1. For each triple (a, b, c), x_a + x_b + x_c <= 2
    #    Indices are 0-based: a-1, b-1, c-1
    #    A matrix A where each row corresponds to a triple.
    #    Since N=180 and num_triples is small (~hundreds), we can build a dense matrix
    #    or use sparse. Dense is likely fine and faster for small sizes in scipy.
    
    if num_triples == 0:
        # No constraints, take all
        return all_numbers

    A = np.zeros((num_triples, N))
    for i, (a, b, c_val) in enumerate(triples_list):
        A[i, a - 1] = 1.0
        A[i, b - 1] = 1.0
        A[i, c_val - 1] = 1.0
        
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))
    
    # Linear Constraint: A @ x <= 2
    # scipy.optimize.milp uses LinearConstraint(A, lb, ub)
    # Here lb is -inf (or 0) and ub is 2 for all rows
    linear_constraint = LinearConstraint(A, lb=-np.inf, ub=np.full(num_triples, 2.0))
    
    # Integrities: all variables are integer
    integrality = np.ones(N)
    
    # 3. Solve
    try:
        res = milp(
            c=c,
            constraints=linear_constraint,
            integrality=integrality,
            bounds=bounds
        )
        
        # Check time
        if time.time() > deadline:
             # Fallback if solver took too long (unlikely for this size, but safe)
             pass
             
        if res.success:
            # Extract solution
            x = res.x
            # x should be 0 or 1 (or very close)
            selected = [i + 1 for i in range(N) if x[i] > 0.5]
            
            # Verify validity just in case
            # If invalid, we might need to clean up, but ILP should be exact
            # However, floating point errors in solver might cause issues?
            # milp should handle integrality strictly.
            
            if selected:
                return selected
            else:
                # Should not happen as we can always pick at least 1 element
                return [1]
        else:
            # Solver failed, fallback to greedy
            pass
    except Exception as e:
        # Fallback to greedy if ILP fails
        pass
        
    # 4. Fallback: Greedy construction
    # This is the same logic as before, kept as a safety net
    def greedy_construct():
        S = set()
        # Sort by degree ascending to pick "safe" nodes first
        # Precompute degrees
        degrees = {i: 0 for i in range(1, N + 1)}
        for t in triples_list:
            for v in t:
                degrees[v] += 1
                
        sorted_nodes = sorted(range(1, N + 1), key=lambda x: degrees[x])
        
        for v in sorted_nodes:
            is_safe = True
            # Check if adding v creates a triple with existing S
            # We need to check all triples containing v
            # Optimization: Precompute neighbors?
            # For now, just check all triples.
            # Since N=180, this is fast.
            for t in triples_list:
                if v in t:
                    others = [u for u in t if u != v]
                    if all(u in S for u in others):
                        is_safe = False
                        break
            if is_safe:
                S.add(v)
        return list(S)
        
    greedy_result = greedy_construct()
    if greedy_result:
        return greedy_result
        
    return [1]
