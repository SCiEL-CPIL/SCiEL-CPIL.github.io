    def outer_212():
        U = np.zeros((2, 4))
        V = np.zeros((2, 4))
        W = np.zeros((4, 4))
        
        # r=0: A0 B0 -> C00 (idx 0)
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        
        # r=1: A0 B1 -> C01 (idx 2)
        U[0, 1] = 1; V[1, 1] = 1; W[2, 1] = 1
        
        # r=2: A1 B0 -> C10 (idx 1)
        U[1, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        
        # r=3: A1 B1 -> C11 (idx 3)
        U[1, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
        
        return U, V, W
    