    def _objective_hard(h):
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = np.convolve(h, h, mode='full')
        return abs(2.0 * N * np.max(conv) / (s * s))
    