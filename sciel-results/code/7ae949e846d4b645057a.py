    def rect_4x4x1():
        rank = 4
        U = np.zeros((16, rank))
        V = np.zeros((20, rank))
        W = np.zeros((20, rank))
        for col in range(4):
            for i in range(4):
                U[i * 4 + col, col] = 1.0
                W[i, col] = 1.0
            V[col * 5, col] = 1.0
        return U, V, W
