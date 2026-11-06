        def _verify_network(network):
            if len(network) == 0:
                return True
            masks = np.arange(1 << N, dtype=np.uint32)
            state = np.zeros((1 << N, N), dtype=np.uint8)
            for i in range(N):
                state[:, i] = (masks >> i) & 1
            
            for i, j in network:
                vi = state[:, i]
                vj = state[:, j]
                swap_mask = (vi > vj)
                if np.any(swap_mask):
                    state[swap_mask, i] = 0
                    state[swap_mask, j] = 1
            
            # Check sorted
            # Efficient check: sum of first k elements should be 0?
            # Or: for each row, is it sorted?
            # state[:, 1:] < state[:, :-1] should be all False.
            # np.any(state[:, 1:] < state[:, :-1])
            # Wait, sorted means 0...0 1...1.
            # So state[i] <= state[i+1].
            # Violation: state[i] > state[i+1].
            if np.any(state[:, :-1] > state[:, 1:]):
                return False
            return True
        