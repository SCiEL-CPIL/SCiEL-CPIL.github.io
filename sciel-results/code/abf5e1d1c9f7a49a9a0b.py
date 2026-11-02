def _find_2x2x3_rank10():
    U = np.zeros((4, 10))
    V = np.zeros((6, 10))
    W = np.zeros((6, 10))
    
    # A known rank-10 construction
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[3,1]=1; W[0,1]=1
    U[2,2]=1; V[0,2]=1; W[1,2]=1
    U[3,3]=1; V[3,3]=1; W[1,3]=1
    U[0,4]=1; V[1,4]=1; W[2,4]=1
    U[1,5]=1; V[4,5]=1; W[2,5]=1
    U[2,6]=1; V[1,6]=1; W[3,6]=1
    U[3,7]=1; V[4,7]=1; W[3,7]=1
    U[0,8]=1; V[2,8]=1; W[4,8]=1
    U[1,9]=1; V[5,9]=1; W[4,9]=1
    # This misses (1,0,2) and (1,1,2)
    
    # Verify
    for i in range(2):
        for j in range(2):
            for k in range(3):
                s = sum(U[i*2+j,r]*V[j*3+k,r]*W[k*2+i,r] for r in range(10))
                if abs(s-1) > 1e-9:
                    return None
    return U.tolist(), V.tolist(), W.tolist()
