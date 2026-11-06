<<<<<<< SEARCH
def synthesize(problem):
    global problem_name
    problem_name = problem.get("name", "")
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    # Try known gate detection first
    known = _detect_known_gate(target, qubits, budget)
    if known is not None:
        # Verify it's within budget
        n_cnot = sum(1 for g in known if g[0] == "cx")
        if n_cnot <= budget:
            return known
    
    # Fallback: generic optimization
    # Build a template with up to budget CNOTs and single-qubit rotations
    # Use a structured pattern: alternate between different CNOT pairs
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # For 2 qubits: use CNOT(0,1) and CNOT(1,0) alternating
    # For 3 qubits: use a set of CNOTs that can reach any 2-qubit interaction
    if qubits == 2:
        cnot_pairs = [(0, 1), (1, 0)]
    else:
        cnot_pairs = [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]
    
    template = []
    n_cnot = 0
    for layer in range(budget):
        template += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        template.append(["cx", *cnot_pairs[layer % len(cnot_pairs)]])
        n_cnot += 1
    template += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    result = _optimize(template, target, qubits, budget)
    return result
=======
def synthesize(problem):
    global problem_name
    problem_name = problem.get("name", "")
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    # Try known gate detection first
    known = _detect_known_gate(target, qubits, budget)
    if known is not None:
        n_cnot = sum(1 for g in known if g[0] == "cx")
        if n_cnot <= budget:
            return known
    
    # Try QSD decomposition
    if qubits == 2 and budget >= 3:
        qsd_result = _qsd_decompose_2q(target)
        if qsd_result is not None:
            n_cnot = sum(1 for g in qsd_result if g[0] == "cx")
            if n_cnot <= budget:
                # Verify
                U_check = _build_fast(qsd_result, qubits)
                if _distance(target, U_check) < 1e-6:
                    return qsd_result
    
    # Fallback: generic optimization
    if qubits == 2:
        cnot_pairs = [(0, 1), (1, 0)]
    else:
        cnot_pairs = [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]
    
    template = []
    for layer in range(budget):
        template += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        template.append(["cx", *cnot_pairs[layer % len(cnot_pairs)]])
    template += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    result = _optimize(template, target, qubits, budget)
    return result
>>>>>>> REPLACE

