<<<<<<< SEARCH
def _optimize_3qubit(target, budget):
    """Optimize a 3-qubit circuit with given CNOT budget."""
    n = 8
    
    # Build a CNOT structure using a standard ladder
    # For 3 qubits, use CNOTs between adjacent qubits
    cnot_pattern = [(0, 1), (1, 2), (0, 1), (1, 2), (0, 1), (1, 2)]
    
    cnots = []
    for i in range(min(budget, 12)):
        cnots.append(cnot_pattern[i % len(cnot_pattern)])
    
    # Build the gate layout
    layout = []
    param_indices = []
    
    # Initial single-qubit gates
    for q in range(3):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        param_indices.append(len(layout) - 1)
    
    for i, (c, t) in enumerate(cnots):
        layout.append(["cx", c, t])
        for q in range(3):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
            param_indices.append(len(layout) - 1)
    
    # Final single-qubit gates
    for q in range(3):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        param_indices.append(len(layout) - 1)
    
    # Check gate count
    total_gates = len(layout)
    if total_gates > 400:
        # Reduce the number of CNOTs
        cnots = cnots[:10]
        layout = []
        param_indices = []
        for q in range(3):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
            param_indices.append(len(layout) - 1)
        for i, (c, t) in enumerate(cnots):
            layout.append(["cx", c, t])
            for q in range(3):
                layout.append(["u3", q, 0.0, 0.0, 0.0])
                param_indices.append(len(layout) - 1)
        for q in range(3):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
            param_indices.append(len(layout) - 1)
    
    n_params = len(param_indices) * 3
    
    def circuit_from_params(params):
        gates = []
        pi = 0
        for gate in layout:
            if gate[0] == "u3":
                q = gate[1]
                gates.append(["u3", q, params[pi], params[pi + 1], params[pi + 2]])
                pi += 3
            else:
                gates.append(list(gate))
        return gates
    
    def objective(params):
        gates = circuit_from_params(params)
        U = _build_unitary(gates, 3)
        return _distance(target, U)
    
    # Initialize with zeros
    best_params = np.zeros(n_params)
    best_obj = objective(best_params)
    
    # Try random initializations
    np.random.seed(42)
    for trial in range(5):
        if _time_remaining() < 30:
            break
        params = np.random.uniform(-math.pi, math.pi, n_params)
        obj = objective(params)
        if obj < best_obj:
            best_obj = obj
            best_params = params.copy()
    
    # Coordinate descent
    params = best_params.copy()
    current_obj = best_obj
    
    for sweep in range(30):
        if _time_remaining() < 20:
            break
        
        step_size = 0.5 / (1 + sweep * 0.3)
        improved = False
        
        for i in range(n_params):
            if _time_remaining() < 20:
                break
            
            best_trial_obj = current_obj
            best_trial_val = params[i]
            
            for factor in (1.0, 0.5, 2.0):
                for direction in (1, -1):
                    trial_val = params[i] + direction * step_size * factor
                    params[i] = trial_val
                    trial_obj = objective(params)
                    if trial_obj < best_trial_obj - 1e-12:
                        best_trial_obj = trial_obj
                        best_trial_val = trial_val
                    params[i] = best_trial_val
            
            if best_trial_obj < current_obj - 1e-12:
                params[i] = best_trial_val
                current_obj = best_trial_obj
                improved = True
        
        if not improved:
            break
        
        if current_obj < best_obj:
            best_obj = current_obj
            best_params = params.copy()
    
    return circuit_from_params(best_params)
=======
def _optimize_3qubit(target, budget):
    """Optimize a 3-qubit circuit with given CNOT budget using SciPy."""
    n = 8
    
    # Build a CNOT structure using a standard ladder
    cnot_pattern = [(0, 1), (1, 2), (0, 1), (1, 2), (0, 1), (1, 2)]
    
    cnots = []
    for i in range(min(budget, 12)):
        cnots.append(cnot_pattern[i % len(cnot_pattern)])
    
    # Build the gate layout
    layout = []
    param_indices = []
    
    # Initial single-qubit gates
    for q in range(3):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        param_indices.append(len(layout) - 1)
    
    for i, (c, t) in enumerate(cnots):
        layout.append(["cx", c, t])
        for q in range(3):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
            param_indices.append(len(layout) - 1)
    
    # Final single-qubit gates
    for q in range(3):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        param_indices.append(len(layout) - 1)
    
    # Check gate count
    total_gates = len(layout)
    if total_gates > 400:
        cnots = cnots[:10]
        layout = []
        param_indices = []
        for q in range(3):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
            param_indices.append(len(layout) - 1)
        for i, (c, t) in enumerate(cnots):
            layout.append(["cx", c, t])
            for q in range(3):
                layout.append(["u3", q, 0.0, 0.0, 0.0])
                param_indices.append(len(layout) - 1)
        for q in range(3):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
            param_indices.append(len(layout) - 1)
    
    n_params = len(param_indices) * 3
    
    def circuit_from_params(params):
        gates = []
        pi = 0
        for gate in layout:
            if gate[0] == "u3":
                q = gate[1]
                gates.append(["u3", q, params[pi], params[pi + 1], params[pi + 2]])
                pi += 3
            else:
                gates.append(list(gate))
        return gates
    
    def objective(params):
        gates = circuit_from_params(params)
        U = _build_unitary(gates, 3)
        return _distance(target, U)
    
    best_obj = float('inf')
    best_params = np.zeros(n_params)
    
    np.random.seed(42)
    for trial in range(3):
        if _time_remaining() < 30:
            break
        x0 = np.random.uniform(-math.pi, math.pi, n_params)
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x
    
    return circuit_from_params(best_params)
>>>>>>> REPLACE
