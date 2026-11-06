<<<<<<< SEARCH
    # Generic unitary: use optimization
    # Allocate remaining time for optimization
    time_remaining = _DEADLINE - _elapsed()
    max_opt_time = max(5.0, time_remaining - 10.0)
    
    gates, dist = _optimize_generic(target, qubits, budget, max_time=max_opt_time)
    return gates
=======
    # Generic unitary: use SVD peeling
    gates, dist = _svd_peel_generic(target, qubits, budget)
    return gates
>>>>>>> REPLACE
