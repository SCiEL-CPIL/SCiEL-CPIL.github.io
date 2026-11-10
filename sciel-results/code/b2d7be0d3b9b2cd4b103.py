<<<<<<< SEARCH
    # Final high-precision polish: one more pass with tighter tolerance
    if _elapsed() < T_LIMIT - 2.0:
        for i in range(N):
            max_p_i = x[i] * V[i, 0] + y[i] * V[i, 1]
            new_max_p_i = max_p_i
            for j in range(N):
                if i == j:
                    continue
                upper_j = p[j] + x[i] * V[i, 0] + y[i] * V[i, 1] - x[j] * V[i, 0] - y[j] * V[i, 1]
                new_max_p_i = min(new_max_p_i, upper_j)
            new_p_i = max(0.0, min(new_max_p_i, max_p_i))
            if new_p_i > p[i] + 1e-16:
                p[i] = new_p_i
=======
    # Final high-precision polish: iterative passes until fully converged
    if _elapsed() < T_LIMIT - 2.0:
        for _pass in range(100):
            changed = False
            for i in range(N):
                max_p_i = x[i] * V[i, 0] + y[i] * V[i, 1]
                new_max_p_i = max_p_i
                for j in range(N):
                    if i == j:
                        continue
                    upper_j = p[j] + x[i] * V[i, 0] + y[i] * V[i, 1] - x[j] * V[i, 0] - y[j] * V[i, 1]
                    new_max_p_i = min(new_max_p_i, upper_j)
                new_p_i = max(0.0, min(new_max_p_i, max_p_i))
                if new_p_i > p[i] + 1e-17:
                    p[i] = new_p_i
                    changed = True
            if not changed:
                break
>>>>>>> REPLACE

