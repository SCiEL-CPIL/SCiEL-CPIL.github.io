import numpy as np

N = 16

def construct():
    # ... generation code ...
    
    # Pre-compute full inputs
    bits = np.arange(1 << N)[:, None]
    full_inputs = (bits >> np.arange(N - 1, -1, -1)) & 1
    
    rng = np.random.default_rng(42)
    fast_inputs = rng.integers(0, 2, size=(1000, N))

    def verify_fast(net):
        state = fast_inputs.copy()
        for i, j in net:
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = state[:, i] + state[:, j] - min_val
            state[:, i] = min_val
            state[:, j] = max_val
        return not np.any(np.diff(state, axis=1) < 0)

    def verify_full(net):
        state = full_inputs.copy()
        for i, j in net:
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = state[:, i] + state[:, j] - min_val
            state[:, i] = min_val
            state[:, j] = max_val
        return not np.any(np.diff(state, axis=1) < 0)

    # ... pruning loop ...
    
    return network
