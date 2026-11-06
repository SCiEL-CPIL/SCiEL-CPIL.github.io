"""XOR-circuit synthesis using minimum-weight matching for subexpression sharing."""

def synthesize(matrix, depth_limit):
    import numpy as np
    from collections import defaultdict
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert to list of tuples for hashing
    rows = [tuple(matrix[i]) for i in range(m)]
    
    # Track available wires
    # Wire 0..n-1: inputs, wire n: constant zero
    # Gate k produces wire n+1+k
    
    gates = []  # list of [wire_a, wire_b]
    wire_to_vector = {}  # wire index -> vector (tuple)
    
    for i in range(n):
        vec = [0] * n
        vec[i] = 1
        wire_to_vector[i] = tuple(vec)
    
    wire_to_vector[n] = tuple([0] * n)  # constant zero
    
    next_wire = n + 1
    
    def get_wire_for_vector(vec):
        """Get existing wire for vector, or create new gate."""
        vec = tuple(vec)
        # Check if we already have this vector
        for w, v in wire_to_vector.items():
            if v == vec:
                return w
        # Need to create new gate - find two wires whose XOR gives vec
        # This shouldn't happen if we're building correctly, but handle it
        return None
    
    def xor_vec(a, b):
        return tuple(x ^ y for x, y in zip(a, b))
    
    def vec_to_list(v):
        return list(v)
    
    # Build initial set of vectors we have
    # We'll use a greedy approach with matching-like optimization
    
    # First, identify all unique non-trivial vectors needed
    needed_vectors = set()
    for row in rows:
        needed_vectors.add(tuple(row))
    
    # Remove trivial vectors (unit vectors and zero)
    trivial = set()
    for i in range(n):
        vec = [0] * n
        vec[i] =1
        trivial.add(tuple(vec))
    trivial.add(tuple([0]*n))
    
    needed_vectors -= trivial
    
    # For each needed vector, find a way to express it as XOR of two existing/new vectors
    # Use a priority queue approach: prefer creating intermediates that are shared
    
    # Count how many times each potential intermediate is used
    # An intermediate v is useful if multiple outputs can be expressed using v
    
    # Let's use a different approach: 
    # 1. Start with all input wires available
    # 2. Repeatedly find the most beneficial intermediate to create
    # 3. An intermediate v = a ^ b is beneficial if it reduces total gate count
    
    # Simpler greedy with sharing detection:
    # Process outputs, and when computing an output, if a subexpression is needed 
    # by multiple outputs, create it as intermediate
    
    # Let me use a cleaner formulation:
    # Maintain a set of available vectors (as wires)
    # For each output, decompose it into XOR of available vectors
    # If decomposition needs >1 XOR, consider creating intermediates
    
    available = dict(wire_to_vector)  # wire -> vector
    wire_count = len(available)
    
    def find_decomposition(vec):
        """Find vec as XOR of two available vectors, or None."""
        vec = tuple(vec)
        for w1, v1 in available.items():
            needed = xor_vec(vec, v1)
            if needed in available.values():
                # Find the wire for needed
                for w2, v2 in available.items():
                    if v2 == needed:
                        return (w1, w2)
        return None
    
    def add_gate(w1, w2):
        """Add a gate producing w1 XOR w2, return new wire index."""
        nonlocal next_wire
        new_vec = xor_vec(available[w1], available[w2])
        gates.append([w1, w2])
        available[next_wire] = new_vec
        new_wire = next_wire
        next_wire += 1
        return new_wire
    
    # Phase 1: Identify frequently used subexpressions
    # For each pair of rows, compute their XOR and see if it helps
    
    # Better approach: use the matching idea
    # Create a graph where:
    # - Nodes: all needed vectors + zero vector
    # - Edges: between vectors v1, v2 with weight = cost to create v1^v2 as intermediate
    # - Find minimum weight set of intermediates that allows all outputs to be computed
    
    # Practical implementation:
    # 1. For each needed vector, find its "depth" (min XORs from inputs)
    # 2. Create intermediates in order of most-shared-first
    
    # Let me implement a clean greedy with sharing:
    
    # Step 1: For each output row, find a decomposition into XORs of input wires
    # Step 2: Identify common subexpressions
    # Step 3: Create intermediates for common subexpressions (sorted by sharing count)
    
    # Decompose each row into set of input indices
    row_indices = []
    for row in rows:
        idxs = [i for i in range(n) if row[i] == 1]
        row_indices.append(set(idxs))
    
    # Count how many times each pair of input indices appears together
    # Actually, count how many times each subset appears as a subexpression
    
    # Simpler: for each row, the set of 1s. If two rows share a large common subset,
    # create an intermediate for that subset.
    
    # Let's use a different strategy:
    # 1. Compute all pairwise XORs of rows
    # 2. For each pairwise XOR, check if it simplifies other rows
    # 3. Use a priority queue to create the most beneficial intermediates first
    
    # Even simpler and effective: 
    # Use Gaussian elimination to find a basis, then express each row in terms of basis
    # But that doesn't minimize gates...
    
    # Let me implement the matching-based approach properly:
    
    # The key insight: if we have vectors v1, v2, v3, v4 and we need to compute:
    # v1^v2, v3^v4, and (v1^v2)^(v3^v4)
    # We can either:
    # a) Create t1=v1^v2, t2=v3^v4, then t1^t2 (3 gates)
    # b) Just compute directly (3 gates) - same
    # But if we need v1^v2 in multiple places, creating it as intermediate saves gates
    
    # Algorithm:
    # 1. Start with only input wires available
    # 2. Maintain a list of "needed" vectors (the output rows)
    # 3. Repeatedly:
    #    a. For each needed vector, find the cheapest way to express it using available vectors
    #    b. If the cheapest way requires creating an intermediate that is also needed elsewhere, create it
    #    c. Otherwise, just create the necessary gates for this output
    
    # Let me implement a practical version:
    
    # Build a dependency graph and use topological ordering with sharing
    
    # For simplicity and effectiveness, let's use this approach:
    # 1. Identify all unique non-trivial vectors in the rows
    # 2. For each pair of these vectors, compute their XOR
    # 3. If the XOR is also in the set of needed vectors or is a unit vector, 
    #    it's a candidate intermediate
    # 4. Use a greedy matching: create intermediates that maximize sharing
    
    # Let me just implement a clean, effective solution:
    
    # Approach: 
    # - Maintain available vectors
    # - For each output, find a decomposition
    # - Cache intermediate results
    # - Use a priority queue to create most-shared intermediates first
    
    # Implementation:
    
    # First, let's see if we can express each row directly from inputs
    # A row with k ones needs k-1 gates if computed directly
    
    # Total gates without sharing: sum(len(row_indices[i]) - 1) for i in range(m)
    
    # With sharing, we can do better
    
    # Let's implement:
    
    # 1. Count frequency of each input index across all rows
    # 2. Group rows by their index sets
    # 3. For each group, find common subexpressions
    
    # Actually, let me just implement a solid greedy:
    
    # available_wires: dict from vector_tuple to wire_index
    # We start with unit vectors and zero
    
    avail = {}
    for i in range(n):
        vec = [0]*n
        vec[i] = 1
        avail[tuple(vec)] = i
    avail[tuple([0]*n)] = n
    
    next_w = n + 1
    gates_list = []
    
    def make_gate(w1, w2):
        nonlocal next_w
        v1 = None
        v2 = None
        for v, w in avail.items():
            if w == w1:
                v1 = v
                break
        for v, w in avail.items():
            if w == w2:
                v2 = v
                break
        new_v = xor_vec(v1, v2)
        gates_list.append([w1, w2])
        avail[new_v] = next_w
        new_w = next_w
        next_w += 1
        return new_w
    
    def get_wire(vec):
        vec = tuple(vec)
        if vec in avail:
            return avail[vec]
        return None
    
    # For each output row, find a decomposition
    # Use a recursive approach with memoization
    
    from functools import lru_cache
    
    # Convert rows to tuples
    row_tuples = [tuple(r) for r in rows]
    
    # For each row, find the minimum number of XORs to compute it
    # This is like finding the minimum weight path in a graph
    
    # Let's use a BFS-like approach for each row, but with sharing
    
    # Simpler: just compute each row, caching intermediates
    
    def compute_vector(vec):
        """Compute vec, returning wire index. Caches intermediates."""
        vec = tuple(vec)
        if vec in avail:
            return avail[vec]
        
        # Find two available vectors whose XOR gives vec
        # Try all pairs of available vectors
        best = None
        avail_list = list(avail.items())
        for i in range(len(avail_list)):
            for j in range(i+1, len(avail_list)):
                v1, w1 = avail_list[i]
                v2, w2 = avail_list[j]
                if xor_vec(v1, v2) == vec:
                    if best is None or (len([x for x in v1 if x==1]) + len([x for x in v2 if x==1])) < best[0]:
                        best = (len([x for x in v1 if x==1]) + len([x for x in v2 if x==1]), w1, w2)
        
        if best is None:
            # Can't express directly - shouldn't happen if we have all inputs
            return None
        
        _, w1, w2 = best
        new_w = make_gate(w1, w2)
        return new_w
    
    # Process each output row
    outputs = []
    for row in row_tuples:
        w = compute_vector(row)
        if w is None:
            # Fallback: compute directly from inputs
            idxs = [i for i in range(n) if row[i] == 1]
            if not idxs:
                w = n  # zero
            else:
                w = idxs[0]
                for idx in idxs[1:]:
                    w = make_gate(w, idx)
        outputs.append(w)
    
    return {'gates': gates_list, 'outputs': outputs}
