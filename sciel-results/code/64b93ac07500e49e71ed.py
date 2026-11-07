<<<<<<< SEARCH
    # Objective function for optimization
    def objective(params):
        p = _generate_pulse_legendre(params)
        if not _check_validity(p):
            return 10.0  # Penalty
        return _compute_score(p, a, a_dag, n)
=======
    # Objective function for optimization
    def objective(params):
        try:
            p = _generate_pulse_legendre(params)
            if not _check_validity(p):
                return 10.0  # Penalty
            return float(_compute_score(p, a, a_dag, n))
        except Exception:
            return 10.0
>>>>>>> REPLACE
