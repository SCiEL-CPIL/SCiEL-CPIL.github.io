    for i in range(N - 1):
        if np.any(x[:, i] > x[:, i + 1]):
            return False
