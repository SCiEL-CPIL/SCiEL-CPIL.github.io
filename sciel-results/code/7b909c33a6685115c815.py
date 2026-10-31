def _dynamics_simulate(z, time_budget, dt=0.002, max_iter=5000):
    x = z[0::3][:N].copy().astype(float)
    y = z[1::3][:N].copy().astype(float)
    r = z[2::3][:N].copy().astype(float)
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2:
        return None

    vx = np.zeros(N)
    vy = np.zeros(N)
    mass = r * r

    x = np.clip(x, r, w - r)
    y = np.clip(y, r, h - r)

    damping = 0.995
    start_time = time.time()

    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break

        fx = np.zeros(N)
        fy = np.zeros(N)

        # Wall forces
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 500.0
        fx += right_viol * 500.0
        fy -= bottom_viol * 500.0
        fy += top_viol * 500.0

        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            f_pairs = overlap * 200.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs
            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]

        vx += dt * fx / mass
        vy += dt * fy / mass
        vx *= damping
        vy *= damping
        x += dt * vx
        y += dt * vy

        # Multiple collision resolution passes
        for _pass in range(5):
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)
            
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            r_sum = r[:, None] + r[None, :]
            iu = np.triu_indices(N, k=1)
            d_sq_pairs = d_sq[iu]
            r_sum_pairs = r_sum[iu]
            d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
            overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

            if np.max(overlap) > 1e-9:
                dx_pairs = dx[iu]
                dy_pairs = dy[iu]
                ux = dx_pairs / d_pairs
                uy = dy_pairs / d_pairs
                for idx in range(len(iu[0])):
                    i_idx = iu[0][idx]
                    j_idx = iu[1][idx]
                    m_i = mass[i_idx]
                    m_j = mass[j_idx]
                    total_m = m_i + m_j
                    correction = overlap[idx] * 0.5
                    x[i_idx] += ux[idx] * correction * (m_j / total_m)
                    y[i_idx] += uy[idx] * correction * (m_j / total_m)
                    x[j_idx] -= ux[idx] * correction * (m_i / total_m)
                    y[j_idx] -= uy[idx] * correction * (m_i / total_m)
            else:
                break

    if not _is_valid_fast(x, y, r, w, h):
        return None

    z_new = np.empty(3 * N + 1)
    z_new[0::3][:N] = x
    z_new[1::3][:N] = y
    z_new[2::3][:N] = r
    z_new[-1] = w
    return z_new
