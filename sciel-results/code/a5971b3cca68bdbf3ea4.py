<<<<<<< SEARCH
    def is_valid(net):
        test = all_inputs.copy()
        for (i, j) in net:
            a = test[:, i]
            b = test[:, j]
            test[:, i] = np.minimum(a, b)
            test[:, j] = np.maximum(a, b)
        return np.all(test[:, :-1] <= test[:, 1:])
    
    # Check initial validity
    if not is_valid(current_network):
        return current_network
    
    best = current_count
    for _ in range(max_iterations):
        # Try random modifications
        action = np.random.random()
        
        if action < 0.3:
            # Try removing a comparator
            if len(current_network) > 1:
                idx = np.random.randint(len(current_network))
                new_net = current_network[:idx] + current_network[idx+1:]
                if is_valid(new_net):
                    current_network = new_net
                    best = min(best, len(current_network))
        elif action < 0.6:
            # Try moving a comparator
            if len(current_network) > 1:
                idx = np.random.randint(len(current_network))
                new_idx = np.random.randint(len(current_network))
                if idx != new_idx:
                    comp = current_network[idx]
                    new_net = current_network[:idx] + current_network[idx+1:new_idx] + [comp] + current_network[new_idx:]
                    if is_valid(new_net):
                        current_network = new_net
        else:
            # Try adding a random comparator
            if len(current_network) < MAX_COMPARATORS:
                i = np.random.randint(N)
                j = np.random.randint(i+1, N)
                pos = np.random.randint(len(current_network) + 1)
                new_net = current_network[:pos] + [(i, j)] + current_network[pos:]
                if is_valid(new_net):
                    current_network = new_net
                    best = min(best, len(current_network))
    
    return current_network
=======
    def is_valid(net):
        """Fast vectorized validity check for all 2^N boolean inputs."""
        if not net:
            return False
        test = all_inputs.copy()
        # Vectorize the application of comparators
        # Note: We can't easily vectorize the loop over comparators in pure numpy without 
        # precomputing indices or using a more complex structure, but 262k * 150 ops is manageable 
        # if we do it in chunks or if numpy overhead is low. 
        # However, a pure python loop over 150 comparators taking slices of 262k arrays is ~150 * 262k * 2 ops.
        # Let's try to optimize this by using a list of indices if possible, or just accept the numpy loop.
        # Actually, for N=18, 2^18 = 262144. 
        # A numpy operation on a column of size 262144 is fast.
        # 150 iterations * (2 min/max + 2 assignments) is acceptable.
        for (i, j) in net:
            a = test[:, i]
            b = test[:, j]
            new_a = np.minimum(a, b)
            new_b = np.maximum(a, b)
            test[:, i] = new_a
            test[:, j] = new_b
        return np.all(test[:, :-1] <= test[:, 1:])
    
    # Check initial validity
    if not is_valid(current_network):
        return current_network
    
    best = current_count
    for _ in range(max_iterations):
        # Try random modifications
        action = np.random.random()
        
        if action < 0.3:
            # Try removing a comparator
            if len(current_network) > 1:
                idx = np.random.randint(len(current_network))
                new_net = current_network[:idx] + current_network[idx+1:]
                if is_valid(new_net):
                    current_network = new_net
                    best = min(best, len(current_network))
        elif action < 0.6:
            # Try moving a comparator
            if len(current_network) > 1:
                idx = np.random.randint(len(current_network))
                new_idx = np.random.randint(len(current_network))
                if idx != new_idx:
                    comp = current_network[idx]
                    new_net = current_network[:idx] + current_network[idx+1:new_idx] + [comp] + current_network[new_idx:]
                    if is_valid(new_net):
                        current_network = new_net
        else:
            # Try adding a random comparator
            if len(current_network) < MAX_COMPARATORS:
                i = np.random.randint(N - 1) # Ensure i < N-1 so i+1 < N
                j = np.random.randint(i+1, N)
                pos = np.random.randint(len(current_network) + 1)
                new_net = current_network[:pos] + [(i, j)] + current_network[pos:]
                if is_valid(new_net):
                    current_network = new_net
                    best = min(best, len(current_network))
    
    return current_network
>>>>>>> REPLACE

