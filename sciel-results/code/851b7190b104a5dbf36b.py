    if len(network) == 0:
        # Should not happen, but if it does, return the initial network
        network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    return network
