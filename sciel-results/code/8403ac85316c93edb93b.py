    z_arr = np.array([c[0] for c in test_cases])
    w0_arr = np.array([c[1] for c in test_cases])
    wa_arr = np.array([c[2] for c in test_cases])
    N_cases = len(test_cases)

    def _quadrature_vec_batch(nodes, weights, z_arr, w0_arr, wa_arr):
        # nodes: (8,), weights: (8,)
        # z_arr: (N,)
        # a_vals: (N, 8)
        a_vals = 1.0 / (1.0 + z_arr[:, np.newaxis] * nodes[np.newaxis, :])
        a_vals = np.clip(a_vals, 1e-12, 1.0)
        log_term = 3.0 * (1.0 + w0_arr[:, np.newaxis] + wa_arr[:, np.newaxis]) * np.log(a_vals) - 3.0 * wa_arr[:, np.newaxis] * (a_vals - 1.0)
        log_term = np.clip(log_term, -700, 700)
        term_L = Omega_L * np.exp(log_term)
        term_m = Omega_m * (a_vals ** -3)
        E2 = term_m + term_L
        E2 = np.maximum(E2, 1e-300)
        E = np.sqrt(E2)
        inv_E = 1.0 / E
        # weights: (8,) -> (1, 8)
        return np.sum(weights[np.newaxis, :] * inv_E, axis=1)

    def _minmax_objective(params):
        nodes, weights = _param_to_nodes_weights(params)
        q_vals = _quadrature_vec_batch(nodes, weights, z_arr, w0_arr, wa_arr)
        errs = np.abs(q_vals - ref_vals) / ref_vals
        return np.max(errs)
