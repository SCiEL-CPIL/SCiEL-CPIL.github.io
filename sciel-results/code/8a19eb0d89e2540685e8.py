def _objective_and_grad(x, tau):
    """Return (f, g) where f = -soft_min, g = gradient."""
    c = x.reshape(-1, 2)

    # Wall clearances (separate x and y for correct gradient)
    wx = 0.5 - np.abs(c[:, 0])
    wy = 0.5 - np.abs(c[:, 1])

    # Pairwise clearances (upper triangle)
    iu, ju = np.triu_indices(N, k=1)
    diff = c[iu] - c[ju]
    pd = np.sqrt(np.sum(diff**2, axis=1)) / 2.0

    # Combine all clearances
    all_d = np.concatenate([wx, wy, pd])
    min_d = np.min(all_d)
    shifted = all_d - min_d
    exp_vals = np.exp(-shifted / tau)
    sum_exp = np.sum(exp_vals)
    w = exp_vals / sum_exp
    soft_min = min_d - tau * np.log(sum_exp)

    grad = np.zeros_like(x)
    g = grad.reshape(N, 2)

    # Weights for wx, wy, pd
    w_wx = w[:N]
    w_wy = w[N:2*N]
    w_pd = w[2*N:]

    # Gradients for wx: d(wx)/dx = -sign(x)
    # d(-soft_min)/dx = - sum w_k d(d_k)/dx
    # Term for wx: - w_wx * (-sign(x)) = w_wx * sign(x)
    g[:, 0] += w_wx * np.sign(c[:, 0])
    
    # Term for wy: - w_wy * (-sign(y)) = w_wy * sign(y)
    g[:, 1] += w_wy * np.sign(c[:, 1])

    # Gradients for pd
    # d(pd)/dx_i = 0.5 * (x_i - x_j) / dist
    # d(-soft_min)/dx_i = - sum w_pd * d(pd)/dx_i
    # = - sum w_pd * 0.5 * (x_i - x_j) / dist
    # = - 0.5 * sum (w_pd / dist) * (x_i - x_j)
    
    for k in range(len(iu)):
        wk = w_pd[k]
        i, j = iu[k], ju[k]
        dvec = c[i] - c[j]
        d = np.sqrt(np.dot(dvec, dvec))
        if d > 1e-12:
            f = 0.5 * wk / d
            g[i] -= f * dvec
            g[j] += f * dvec

    return -soft_min, grad
