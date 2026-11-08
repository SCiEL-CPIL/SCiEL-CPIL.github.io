<<<<<<< SEARCH
    def simulate_unitary_fast(I_Q, delta, scale):
        """Simulate pulse using unitary evolution - vectorized over states."""
        U = np.eye(3, dtype=complex)
        for i in range(N):
            H = get_H(I_Q[i, 0], I_Q[i, 1], delta, scale)
            U = expm(-1j * H * dt) @ U
        return U

    def score_case_fast(I_Q, delta, scale):
        """Score a single case using vectorized simulation."""
        U = simulate_unitary_fast(I_Q, delta, scale)
        # Fidelity for each state: |<target|psi_final>|^2
        # psi_final = U @ states, target = targets
        # fidelity_i = |vdot(target_i, U @ state_i)|^2
        evolved = U @ states  # (3, 6)
        fids = np.abs(np.sum(targets.conj() * evolved, axis=0))**2
        return 1.0 - np.mean(fids)

    def score(I_Q):
        """Full score: mean + max over 9 cases."""
        if not check_constraints(I_Q):
            return 1e6
        infidelities = []
        for d in deltas:
            for s in scales:
                infidelities.append(score_case_fast(I_Q, d, s))
        infidelities = np.array(infidelities)
        return np.mean(infidelities) + np.max(infidelities)
=======
    def simulate_unitary_fast(I_Q, delta, scale):
        """Simulate pulse using unitary evolution - vectorized over states."""
        U = np.eye(3, dtype=complex)
        for i in range(N):
            H = get_H(I_Q[i, 0], I_Q[i, 1], delta, scale)
            U = expm(-1j * H * dt) @ U
        return U

    def score_case_fast(I_Q, delta, scale):
        """Score a single case using vectorized simulation."""
        U = simulate_unitary_fast(I_Q, delta, scale)
        # Fidelity for each state: |<target|psi_final>|^2
        # psi_final = U @ states, target = targets
        # fidelity_i = |vdot(target_i, U @ state_i)|^2
        evolved = U @ states  # (3, 6)
        fids = np.abs(np.sum(targets.conj() * evolved, axis=0))**2
        return 1.0 - np.mean(fids)

    def score(I_Q, smooth_beta=0.0):
        """Full score: mean + max over 9 cases. If smooth_beta>0, use soft-max."""
        if not check_constraints(I_Q):
            return 1e6
        infidelities = []
        for d in deltas:
            for s in scales:
                infidelities.append(score_case_fast(I_Q, d, s))
        infidelities = np.array(infidelities)
        mean_inf = np.mean(infidelities)
        if smooth_beta > 0:
            # Soft-max: (1/beta) * log(sum(exp(beta * x_i)))
            shifted = beta * infidelities - beta * np.max(infidelities)
            soft_max = np.max(infidelities) + np.log(np.sum(np.exp(shifted))) / beta
            return mean_inf + soft_max
        else:
            return mean_inf + np.max(infidelities)
>>>>>>> REPLACE

