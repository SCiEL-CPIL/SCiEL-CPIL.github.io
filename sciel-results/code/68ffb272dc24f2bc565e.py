def schoolbook_2x2x1():
    # 2x2x1: T[i,j,0]=1 for i,j in {0,1}
    # Rank 2: U=[1,1], V=[1,1], W=[1,1]
    U = np.zeros((4,2)); V = np.zeros((2,2)); W = np.zeros((1,2))
    U[0,0]=1; U[1,0]=1; V[0,0]=1; V[1,0]=1; W[0,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; V[1,1]=1; W[0,1]=1
    return U, V, W
