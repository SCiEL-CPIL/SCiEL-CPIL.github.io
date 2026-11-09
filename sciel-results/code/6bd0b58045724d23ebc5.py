for i, j in net:
    min_val = np.minimum(state[:, i], state[:, j])
    max_val = np.maximum(state[:, i], state[:, j])
    state[:, i] = min_val
    state[:, j] = max_val
