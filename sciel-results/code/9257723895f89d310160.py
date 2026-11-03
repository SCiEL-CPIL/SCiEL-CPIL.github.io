<<<<<<< SEARCH
_DEADLINE = 110.0  # safety margin before the 120s (and 150s kill) limit


def _penalty_and_objective(z, n):
    """z = [x_1..x_n, y_1..y_n, r_1..r_n]. Return (value, grad).
    value = sum(r) - K_overlap*overlap - K_boundary*boundary.
    Higher value is better; we minimize -value."""
    x = z[:n]
    y = z[n:2*n]
    r = z[2*n:3*n]

    # pairwise overlap (positive when circles overlap)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    rr = r[:, None] + r[None, :]
    d = np.sqrt(dx*dx + dy*dy)
    # overlap amount per pair (0 if separated)
    ov = np.maximum(rr - d, 0.0)
    # use only upper triangle
    iu = np.triu_indices(n, k=1)
    overlap = np.sum(ov[iu])

    # boundary violation: circle must be inside [0,1]^2
    b = np.maximum(0.0, r - x) + np.maximum(0.0, r - y) \
      + np.maximum(0.0, r - (1.0 - x)) + np.maximum(0.0, r - (1.0 - y))
    boundary = np.sum(b)

    K_ov = 50.0
    K_b = 50.0
    value = np.sum(r) - K_ov * overlap - K_b * boundary

    # gradients
    gx = np.zeros(n); gy = np.zeros(n); gr = np.zeros(n)

    # overlap gradient
    for i, j in zip(iu[0], iu[1]):
        if d[i, j] > 1e-12:
            s = ov[i, j]
            if s > 0:
                ux = dx[i, j] / d[i, j]
                uy = dy[i, j] / d[i, j]
                gx[i] -= K_ov * ux
                gy[i] -= K_ov * uy
                gx[j] += K_ov * ux
                gy[j] += K_ov * uy
                gr[i] -= K_ov
                gr[j] -= K_ov
        else:
            # degenerate: push apart along x
            gx[i] -= K_ov
            gx[j] += K_ov
            gr[i] -= K_ov
            gr[j] -= K_ov

    # boundary gradient
    for i in range(n):
        if r[i] > x[i]:
            gx[i] += K_b
            gr[i] -= K_b
        if r[i] > y[i]:
            gy[i] += K_b
            gr[i] -= K_b
        if r[i] > (1.0 - x[i]):
            gx[i] -= K_b
            gr[i] -= K_b
        if r[i] > (1.0 - y[i]):
            gy[i] -= K_b
            gr[i] -= K_b

    # objective gradient (minimize -value)
    gr += 1.0  # d(sum r)/dr = 1, so d(-value)/dr = -1; we return grad of -value
    grad = np.concatenate([-gx, -gy, -gr])
    return -value, grad
=======
_DEADLINE = 110.0  # safety margin before the 120s (and 150s kill) limit


def _penalty_and_objective(z, n):
    """z = [x_1..x_n, y_1..y_n, r_1..r_n]. Return (value, grad).
    value = sum(r) - K_overlap*overlap - K_boundary*boundary.
    Higher value is better; we minimize -value."""
    x = z[:n]
    y = z[n:2*n]
    r = z[2*n:3*n]

    # pairwise overlap (positive when circles overlap)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    rr = r[:, None] + r[None, :]
    d = np.sqrt(dx*dx + dy*dy)
    # overlap amount per pair (0 if separated)
    ov = np.maximum(rr - d, 0.0)
    # use only upper triangle
    iu = np.triu_indices(n, k=1)
    overlap = np.sum(ov[iu])

    # boundary violation: circle must be inside [0,1]^2
    b = np.maximum(0.0, r - x) + np.maximum(0.0, r - y) \
      + np.maximum(0.0, r - (1.0 - x)) + np.maximum(0.0, r - (1.0 - y))
    boundary = np.sum(b)

    K_ov = 50.0
    K_b = 50.0
    value = np.sum(r) - K_ov * overlap - K_b * boundary

    # gradients
    gx = np.zeros(n); gy = np.zeros(n); gr = np.zeros(n)

    # overlap gradient
    for i, j in zip(iu[0], iu[1]):
        if d[i, j] > 1e-12:
            s = ov[i, j]
            if s > 0:
                ux = dx[i, j] / d[i, j]
                uy = dy[i, j] / d[i, j]
                gx[i] -= K_ov * ux
                gy[i] -= K_ov * uy
                gx[j] += K_ov * ux
                gy[j] += K_ov * uy
                gr[i] -= K_ov
                gr[j] -= K_ov
        else:
            # degenerate: push apart along x
            gx[i] -= K_ov
            gx[j] += K_ov
            gr[i] -= K_ov
            gr[j] -= K_ov

    # boundary gradient
    for i in range(n):
        if r[i] > x[i]:
            gx[i] += K_b
            gr[i] -= K_b
        if r[i] > y[i]:
            gy[i] += K_b
            gr[i] -= K_b
        if r[i] > (1.0 - x[i]):
            gx[i] -= K_b
            gr[i] -= K_b
        if r[i] > (1.0 - y[i]):
            gy[i] -= K_b
            gr[i] -= K_b

    # objective gradient (minimize -value)
    # We want to minimize -value. 
    # value = sum(r) - K_ov*ov - K_b*bd
    # -value = -sum(r) + K_ov*ov + K_b*bd
    # d(-value)/dr = -1 + K_ov*(d ov/dr) + K_b*(d bd/dr)
    # In the code above, gr accumulates -K_ov and -K_b terms?
    # Let's re-verify the gradient accumulation in the code above.
    # In overlap loop: gr[i] -= K_ov. This is d(K_ov*ov)/dr? 
    # ov = max(rr-d, 0). d(ov)/dr_i = 1 if overlapping.
    # So d(K_ov*ov)/dr_i = K_ov.
    # We are minimizing -value = -sum(r) + K_ov*ov + K_b*bd.
    # So grad wrt r should be -1 + K_ov*(indicator) + K_b*(indicator).
    # The code adds -K_ov to gr. Then later we do grad = concat(-gx, -gy, -gr).
    # So final grad_r = -gr = -(-K_ov + ... ) = K_ov ...
    # Wait, the code has:
    # gr[i] -= K_ov
    # ...
    # gr += 1.0
    # grad = np.concatenate([-gx, -gy, -gr])
    # So grad_r = - ( -K_ov + 1 ) = K_ov - 1.
    # This matches d(-value)/dr = -1 + K_ov.
    # So the logic is correct.
    gr += 1.0
    grad = np.concatenate([-gx, -gy, -gr])
    return -value, grad
>>>>>>> REPLACE
