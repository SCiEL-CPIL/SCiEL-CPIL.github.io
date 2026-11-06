def _verify_net_numpy(net, N=16):
    if not net:
        return N == 1
    # Generate inputs
    inputs = np.zeros((1 << N, N), dtype=np.int8)
    for j in range(N):
        inputs[:, j] = (np.arange(1 << N) >> j) & 1
        
    for i, j in net:
        min_val = np.minimum(inputs[:, i], inputs[:, j])
        max_val = np.maximum(inputs[:, i], inputs[:, j])
        inputs[:, i] = min_val
        inputs[:, j] = max_val
        
    diffs = np.diff(inputs, axis=1)
    return not np.any(diffs < 0)
