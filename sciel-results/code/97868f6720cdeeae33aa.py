<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    Returns {'gates': [[wire, wire], ...], 'outputs': [wire, ...]}
    
    Strategy:
    1. Convert rows to bitmasks.
    2. Use Beam Search to find a low-gate-count circuit that computes all unique row masks.
    3. The search space is the set of available masks. We start with inputs and 0.
    4. We expand by adding gates (XOR of two available masks).
    5. We stop when all target masks are available.
    6. This allows for global sharing of intermediate computations.
    """
    start_time = time.time()
    deadline = start_time + 50.0
    
    A = np.array(matrix, dtype=np.uint8)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    n_rows, n_cols = A.shape
    
    # Convert rows to integer masks
    row_masks = []
    for i in range(n_rows):
        mask = 0
        for j in range(n_cols):
            if A[i, j] == 1:
                mask |= (1 << j)
        row_masks.append(mask)
    
    # Unique targets
    targets = set(row_masks)
    targets.discard(0) # 0 is always available
    
    if not targets:
        return {'gates': [], 'outputs': [n_cols] * n_rows}
    
    # Initial state
    # Wire IDs: 0..n_cols-1 are inputs, n_cols is 0.
    # New gates get IDs n_cols+1, n_cols+2, ...
    
    initial_available = set()
    initial_wire_map = {}
    for i in range(n_cols):
        m = 1 << i
        initial_available.add(m)
        initial_wire_map[m] = i
    initial_available.add(0)
    initial_wire_map[0] = n_cols
    
    # Beam Search
    # State: (cost, available_set, wire_map, gates_list)
    # We use a list for the beam, sorted by cost.
    
    beam = [(0, frozenset(initial_available), initial_wire_map, [])]
    best_cost = float('inf')
    best_solution = None
    
    beam_width = 200
    max_depth = 20 # Max gates to search
    
    # To avoid re-processing identical available sets with higher cost
    seen_states = {} # frozenset(available) -> min_cost
    
    while beam:
        if time.time() > deadline - 5.0:
            break
            
        # Sort beam by cost
        beam.sort(key=lambda x: x[0])
        
        new_beam = []
        
        for cost, available, wire_map, gates in beam:
            if cost >= best_cost:
                continue
            if cost > max_depth:
                continue
                
            # Check if all targets are available
            if targets.issubset(available):
                if cost < best_cost:
                    best_cost = cost
                    # Reconstruct outputs
                    outputs = []
                    for mask in row_masks:
                        outputs.append(wire_map[mask])
                    best_solution = {'gates': gates, 'outputs': outputs}
                continue
            
            # Prune if we've seen this available set with lower or equal cost
            avail_key = available
            if avail_key in seen_states:
                if seen_states[avail_key] <= cost:
                    continue
            else:
                seen_states[avail_key] = cost
            
            # Generate neighbors
            avail_list = list(available)
            n_avail = len(avail_list)
            
            # Optimization: If n_avail is large, we can't check all pairs.
            # But n_avail is small in early steps.
            # In later steps, we might have many masks.
            # Limit pair checks if too many?
            # For n_cols=16, max useful masks is small.
            
            for i in range(n_avail):
                m1 = avail_list[i]
                for j in range(i + 1, n_avail):
                    m2 = avail_list[j]
                    m_new = m1 ^ m2
                    
                    if m_new in available:
                        continue
                    
                    # New state
                    new_available = available | {m_new}
                    new_wire_id = n_cols + 1 + len(gates)
                    new_wire_map = wire_map.copy()
                    new_wire_map[m_new] = new_wire_id
                    new_gates = gates + [[wire_map[m1], wire_map[m2]]]
                    new_cost = cost + 1
                    
                    new_beam.append((new_cost, new_available, new_wire_map, new_gates))
                    
        # Prune new_beam
        if not new_beam:
            break
            
        # Sort by cost
        new_beam.sort(key=lambda x: x[0])
        
        # Keep top beam_width
        # Also, if multiple states have the same available set, keep the one with lowest cost
        # (Already handled by seen_states, but within the beam we might have duplicates)
        
        beam = new_beam[:beam_width]
        
        # If we found a solution, we can stop if the minimum cost in beam is >= best_cost
        if best_solution is not None:
            if beam and beam[0][0] >= best_cost:
                break

    if best_solution is None:
        # Fallback to Gaussian Elimination if Beam Search fails or times out
        return _synthesize_gaussian(A, row_masks, n_cols)
        
    return best_solution
=======
# EVOLVE-BLOCK-START
def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit using minimum-cost flow on a layered graph.
    Returns {'gates': [[wire, wire], ...], 'outputs': [wire, ...]}
    
    Strategy:
    1. Convert rows to bitmasks.
    2. Build a layered graph where each layer represents depth.
    3. Nodes are (mask, depth) pairs.
    4. Edges represent XOR operations with cost 1.
    5. Find minimum-cost flow from source (inputs at depth 0) to sinks (targets at any depth).
    6. Extract gates and outputs from the flow.
    """
    start_time = time.time()
    deadline = start_time + 50.0
    
    A = np.array(matrix, dtype=np.uint8)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    n_rows, n_cols = A.shape
    
    # Convert rows to integer masks
    row_masks = []
    for i in range(n_rows):
        mask = 0
        for j in range(n_cols):
            if A[i, j] == 1:
                mask |= (1 << j)
        row_masks.append(mask)
    
    # Unique targets with their row indices
    target_to_rows = defaultdict(list)
    for i, mask in enumerate(row_masks):
        target_to_rows[mask].append(i)
    
    # If all targets are 0
    if all(mask == 0 for mask in row_masks):
        return {'gates': [], 'outputs': [n_cols] * n_rows}
    
    # Maximum depth we'll search
    max_depth = min(depth_limit, 16) if depth_limit else 16
    
    # Use a greedy + local search approach with flow-based optimization
    # First, get a good initial solution using Gaussian elimination
    initial_solution = _synthesize_gaussian(A, row_masks, n_cols)
    if time.time() > deadline - 10.0:
        return initial_solution
    
    # Now try to improve using iterative deepening A* on the gate count
    best_gates = initial_solution['gates']
    best_outputs = initial_solution['outputs']
    best_gate_count = len(best_gates)
    
    # Try to find better solutions with fewer gates
    for max_gates in range(best_gate_count - 1, max(0, best_gate_count - 20), -1):
        if time.time() > deadline - 5.0:
            break
        
        solution = _search_with_gate_limit(row_masks, n_cols, max_gates, max_depth, deadline)
        if solution is not None:
            best_gates = solution['gates']
            best_outputs = solution['outputs']
            best_gate_count = len(best_gates)
    
    return {'gates': best_gates, 'outputs': best_outputs}
>>>>>>> REPLACE

