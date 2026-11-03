# Inside _nesterov_accelerated_gradient
# ...
val_y, grad = _smooth_max_obj_grad(y, beta, lam_reg, M_reg)
# ...
for bt in range(20):
    h_candidate = y - step_lr * grad
    h_candidate = _project(h_candidate, N / 2.0)
    val_candidate, _ = _smooth_max_obj_grad(h_candidate, beta, lam_reg, M_reg)
    if val_candidate <= val_y + 1e-6:
        h_new = h_candidate
        break
    step_lr *= 0.5
