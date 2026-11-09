import scipy.optimize as opt

def _optimize_2qubit(target, budget, time_budget):
    start = time.time()
    num_cnots = min(budget, 3)
    if num_cnots == 0:
        return [[], 1.0]
    
    # CNOT pattern: (0,1), (1,0), (0,1)
    cnot_pairs = [(0, 1), (1, 0), (0, 1)][:num_cnots]
    num_layers = num_cnots + 1
    num_u3 = 2 * num_layers
    n_params = 3 * num_u3
    
    def build_circuit(params):
        gates = []
        k = 0
        for i in range(num_layers):
            gates.append(["u3", 0, params[k], params[k+1], params[k+2]])
            k += 3
            gates.append(["u3", 1, params[k], params[k+1], params[k+2]])
            k += 3
            if i < num_cnots:
                gates.append(["cx", cnot_pairs[i][0], cnot_pairs[i][1]])
        return gates
    
    def dist_fn(params):
        gates = build_circuit(params)
        U = _build(gates, 2)
        return _distance(target, U)
        
    def neg_fidelity(params):
        # Minimize 1 - fidelity = distance
        return dist_fn(params)

    best_params = None
    best_dist = 1.0
    
    # Multiple restarts
    for restart in range(5):
        if time.time() - start > time_budget:
            break
        
        if restart == 0:
            x0 = [0.0] * n_params
        else:
            import random
            rng = random.Random(42 + restart)
            x0 = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
            
        try:
            res = opt.minimize(neg_fidelity, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_dist:
                best_dist = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        best_params = [0.0] * n_params
        
    gates = build_circuit(best_params)
    return gates, best_dist
