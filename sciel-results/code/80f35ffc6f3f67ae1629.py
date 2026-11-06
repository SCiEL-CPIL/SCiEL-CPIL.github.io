<<<<<<< SEARCH
def _optimize_generic(target, qubits, budget, rng_seed=42):
    """Optimize a generic unitary using coordinate descent."""
    if _time_left() < 5:
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    import random
    rng = random.Random(rng_seed)
    
    # Try block decomposition for 3 qubits
    if qubits == 3 and budget >= 12:
        block_result = _decompose_3qubit_analytic(target, qubits, budget)
        if block_result is not None:
            cnot_count = sum(1 for g in block_result if g[0] == "cx")
            if cnot_count <= budget:
                d = _distance(target, _build(block_result, qubits))
                if d < 0.01:
                    return block_result
    
    # General coordinate descent
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Build a layout with CNOTs and single-qubit gates
    n_layers = min(budget, 10)
    layout = []
    for layer in range(n_layers):
        layout += [("u3", q) for q in range(qubits)]
        layout.append(("cx", *pairs[layer % len(pairs)]))
    layout += [("u3", q) for q in range(qubits)]
    
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(["cx", gate[1], gate[2]])
        return gates
    
    best = [rng.uniform(-0.5, 0.5) for _ in range(n_params)]
    best_distance = _distance(target, _build(circuit(best), qubits))
    
    sweeps = 4
    if _time_left() < 30:
        sweeps = 2
    if _time_left() < 10:
        sweeps = 1
        
    for sweep in range(sweeps):
        step = 0.5 / (1 + sweep)
        for i in range(n_params):
            if _time_left() < 5:
                break
            for trial in (best[i] + step, best[i] - step,
                         best[i] + rng.uniform(-math.pi, math.pi) * 0.1,
                         best[i] + rng.uniform(-math.pi, math.pi) * 0.01):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build(circuit(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
        if best_distance < 1e-10:
            break
        
    return circuit(best)
=======
def _optimize_generic(target, qubits, budget, rng_seed=42):
    """Optimize a generic unitary using structured coordinate descent."""
    if _time_left() < 5:
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    import random
    rng = random.Random(rng_seed)
    
    # Try block decomposition for 3 qubits
    if qubits == 3 and budget >= 12:
        block_result = _decompose_3qubit_analytic(target, qubits, budget)
        if block_result is not None:
            cnot_count = sum(1 for g in block_result if g[0] == "cx")
            if cnot_count <= budget:
                d = _distance(target, _build(block_result, qubits))
                if d < 0.01:
                    return block_result
    
    # For 2 qubits, use the 2-qubit decomposer
    if qubits == 2:
        return _decompose_2qubit(target)
    
    # General coordinate descent for 3 qubits
    # Use a structured CNOT pattern that covers all pairs
    pairs = [(0, 1), (1, 2), (0, 2)]
    
    # Build a layout with CNOTs and single-qubit gates
    n_layers = min(budget, 12)
    layout = []
    for layer in range(n_layers):
        layout += [("u3", q) for q in range(qubits)]
        layout.append(("cx", *pairs[layer % len(pairs)]))
    layout += [("u3", q) for q in range(qubits)]
    
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(["cx", gate[1], gate[2]])
        return gates
    
    # Multiple initializations
    best = None
    best_distance = float('inf')
    
    for init_seed in range(3):
        rng_init = random.Random(rng_seed + init_seed)
        if init_seed == 0:
            init = [0.0] * n_params
        elif init_seed == 1:
            init = [rng_init.uniform(-0.3, 0.3) for _ in range(n_params)]
        else:
            init = [rng_init.uniform(-math.pi, math.pi) * 0.05 for _ in range(n_params)]
        
        d = _distance(target, _build(circuit(init), qubits))
        if d < best_distance:
            best_distance = d
            best = init[:]
        
        # Quick coordinate descent
        current = init[:]
        current_d = d
        for sweep in range(2):
            if _time_left() < 5:
                break
            step = 0.4 / (1 + sweep * 0.5)
            for i in range(n_params):
                if _time_left() < 3:
                    break
                for delta in (step, -step, step * 0.1, -step * 0.1):
                    values = current[:]
                    values[i] = current[i] + delta
                    d = _distance(target, _build(circuit(values), qubits))
                    if d < current_d:
                        current, current_d = values, d
            if current_d < 1e-12:
                break
        
        if current_d < best_distance:
            best_distance = current_d
            best = current[:]
    
    # Final refinement
    sweeps = 3
    if _time_left() < 30:
        sweeps = 2
    if _time_left() < 10:
        sweeps = 1
        
    for sweep in range(sweeps):
        step = 0.3 / (1 + sweep)
        for i in range(n_params):
            if _time_left() < 5:
                break
            for trial in (best[i] + step, best[i] - step,
                         best[i] + rng.uniform(-math.pi, math.pi) * 0.05,
                         best[i] + rng.uniform(-math.pi, math.pi) * 0.01):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build(circuit(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
        if best_distance < 1e-10:
            break
        
    return circuit(best)
>>>>>>> REPLACE
