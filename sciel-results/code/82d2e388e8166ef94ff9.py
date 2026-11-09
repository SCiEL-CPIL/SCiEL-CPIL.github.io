<<<<<<< SEARCH
        x = np.arange(n_inputs, dtype=np.uint32)
        # state[wire, input_index] = bit 'wire' of input 'input_index'
        state = ((x[None, :] >> np.arange(N, dtype=np.uint32)) & 1).astype(np.uint8)
=======
        x = np.arange(n_inputs, dtype=np.uint32)
        # state[wire, input_index] = bit 'wire' of input 'input_index'
        # Reshape arange to (N, 1) to broadcast with (1, n_inputs)
        bits = np.arange(N, dtype=np.uint32).reshape(N, 1)
        state = ((x[None, :] >> bits) & 1).astype(np.uint8)
>>>>>>> REPLACE
