<<<<<<< SEARCH
def _optimize_generic_2q(target, qubits, budget, rng_seed=42):
    """Specialized optimizer for 2-qubit unitaries using Euler angle decomposition."""
    # For 2 qubits, any unitary can be decomposed as:
    # U = (A ⊗ B) · CX · (C ⊗ D) · CX · (E ⊗ F) · CX · (G ⊗ H)
    # with 3 CNOTs (for general 2-qubit unitaries).
    # But we might have fewer CNOTs in budget.
    
    # Actually, let's use a more direct approach:
    # Parameterize with the layout and do coordinate descent, but with better init.
    
    # For 2-qubit, we can use the fact that:
    # U = U_single_left · CX · U_single_mid · CX · U_single_right
    # with 2 CNOTs, or with 3 CNOTs for general case.
    
    rng = random.Random(rng_seed)
    
    if qubits == 2:
        # Try 2-CNOT decomposition first if budget >= 2
        # U = (A⊗B) CX (C⊗D) CX (E⊗F)
        # This works for unitaries with certain properties. For general 2-qubit, need 3 CNOTs.
        
        # Let's just use the general layout optimizer but with smarter init
        # Actually, let's try to use scipy or numpy to get a better initial guess.
        
        # Initialize with identity and optimize
        layout = []
        for q in range(2):
            layout.append(("u3", q))
        
        # CNOT pattern for 2 qubits
        cnot_seq = []
        for i in range(min(budget, 6)):
            if i % 2 == 0:
                cnot_seq.append((0, 1))
            else:
                cnot_seq.append((1, 0))
        
        for c, t in cnot_seq:
            layout.append(("cx", c, t))
            for q in range(2):
                layout.append(("u3", q))
        
        for q in range(2):
            layout.append(("u3", q))
        
        n_u3 = sum(1 for e in layout if e[0] == "u3")
        n_params = 3 * n_u3
        
        # Initialize with zeros
        best_params = [0.0] * n_params
        gates, matrix = _build_from_params(layout, best_params, qubits)
        best_dist = _distance(target, matrix)
        
        # Coordinate descent
        max_sweeps = 15
        base_step = 1.5
        
        for sweep in range(max_sweeps):
            if _time_left() < 15.0:
                break
            step = base_step / (1.0 + 0.3 * sweep)
            
            for pi in range(n_params):
                if _time_left() < 15.0:
                    break
                best_d = best_dist
                best_val = best_params[pi]
                # Golden-section-like search around current value
                for frac in [step, -step, step*0.5, -step*0.5, step*0.25, -step*0.25]:
                    trial_val = best_params[pi] + frac
                    new_params = best_params[:]
                    new_params[pi] = trial_val
                    _, mat = _build_from_params(layout, new_params, qubits)
                    d = _distance(target, mat)
                    if d < best_d:
                        best_d = d
                        best_val = trial_val
                        best_params = new_params
                best_params[pi] = best_val
                best_dist = best_d
            
            if best_dist < 1e-12:
                break
        
        gates, _ = _build_from_params(layout, best_params, qubits)
        return gates
    
    else:
        return _optimize_generic(target, qubits, budget, rng_seed)
=======
def _peel_decomposition(target, qubits, budget):
    """Analytic layer-by-layer peeling to decompose target into gates."""
    U = target.copy()
    gates = []
    
    # Determine CNOT pattern
    if qubits == 2:
        cnot_pattern = [(0, 1), (1, 0), (0, 1)]
    else:
        cnot_pattern = [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]
    
    # We'll peel from the left: U = G1 G2 ... Gn
    # At each step, peel off a single-qubit gate or a CNOT.
    # Strategy: alternate between peeling single-qubit gates and CNOTs.
    
    # Phase 1: Peel single-qubit gates to reduce U
    # Phase 2: Peel CNOTs and single-qubit gates alternately
    
    # First, peel single-qubit gates on all qubits
    for q in range(qubits):
        if _time_left() < 10.0:
            break
        params, U = _peel_single_qubit(U, q, qubits)
        if params is not None:
            gates.append(_u3(q, params[0], params[1], params[2]))
    
    # Now peel CNOTs and single-qubit gates
    cnot_used = 0
    for i in range(budget):
        if _time_left() < 10.0:
            break
        
        # Peel a CNOT
        c, t = cnot_pattern[i % len(cnot_pattern)]
        U = _peel_cnot(U, c, t, qubits)
        gates.append(_cx(c, t))
        cnot_used += 1
        
        # Peel single-qubit gates
        for q in range(qubits):
            if _time_left() < 10.0:
                break
            params, U = _peel_single_qubit(U, q, qubits)
            if params is not None:
                gates.append(_u3(q, params[0], params[1], params[2]))
    
    # Final single-qubit gates
    for q in range(qubits):
        if _time_left() < 10.0:
            break
        params, U = _peel_single_qubit(U, q, qubits)
        if params is not None:
            gates.append(_u3(q, params[0], params[1], params[2]))
    
    return gates, U
>>>>>>> REPLACE

