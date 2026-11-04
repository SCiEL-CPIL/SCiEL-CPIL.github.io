def coordinate_descent(C, A, deadline, n_sweeps=50):
    """Fine-grained coordinate descent with golden-section-like steps."""
    step = 0.5
    
    def get_flat(C_cur, A_cur):
        f = np.empty(3 * N)
        f[0::3] = C_cur[:, 0]
        f[1::3] = A_cur
        f[2::3] = C_cur[:, 1]
        return f

    for sweep in range(n_sweeps):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(N):
            # Optimize x[i]
            best_x = C[i, 0]
            best_val = objective_flat(get_flat(C, A))
            for dx in [-step, step]:
                C[i, 0] += dx
                val = objective_flat(get_flat(C, A))
                if val < best_val:
                    best_val = val
                    best_x = C[i, 0]
                C[i, 0] -= dx
            C[i, 0] = best_x

            # Optimize y[i]
            best_y = C[i, 1]
            best_val = objective_flat(get_flat(C, A))
            for dy in [-step, step]:
                C[i, 1] += dy
                val = objective_flat(get_flat(C, A))
                if val < best_val:
                    best_val = val
                    best_y = C[i, 1]
                C[i, 1] -= dy
            C[i, 1] = best_y

            # Optimize angle[i]
            best_a = A[i]
            best_val = objective_flat(get_flat(C, A))
            for da in [-step, step]:
                A[i] += da
                val = objective_flat(get_flat(C, A))
                if val < best_val:
                    best_val = val
                    best_a = A[i]
                A[i] -= da
            A[i] = best_a

            if C[i, 0] != best_x or C[i, 1] != best_y or A[i] != best_a:
                improved = True

        if not improved:
            break
        step *= 0.5
    return C, A
