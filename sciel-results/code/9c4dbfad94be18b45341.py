def is_valid(net, inputs):
    states = inputs.copy()
    for i, j in net:
        left = states[:, i]
        right = states[:, j]
        states[:, i] = np.minimum(left, right)
        states[:, j] = np.maximum(left, right)
    return np.all(np.diff(states, axis=1) >= 0)
