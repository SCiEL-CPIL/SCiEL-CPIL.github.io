def construct():
    import numpy as np
    
    n, m, p = 4, 4, 5
    
    def get_strassen_2x2x2():
        u = np.zeros((4, 7))
        v = np.zeros((4, 7))
        w = np.zeros((4, 7))
        
        u[0,0]=1; u[3,0]=1; v[0,0]=1; v[3,0]=1; w[0,0]=1; w[3,0]=1
        u[2,1]=1; u[3,1]=1; v[0,1]=1; w[2,1]=1
        u[0,2]=1; v[1,2]=1; v[3,2]=-1; w[1,2]=1
        u[3,3]=1; v[2,3]=1; v[0,3]=-1; w[2,3]=1; w[3,3]=1
        u[0,4]=1; u[1,4]=1; v[3,4]=1; w[3,4]=1
        u[2,5]=1; u[0,5]=-1; v[0,5]=1; v[1,5]=1; w[3,5]=1
        u[1,6]=1; u[3,6]=-1; v[2,6]=1; v[3,6]=1; w[1,6]=1; w[3,6]=1
        
        return u, v, w
    
    u2, v2, w2 = get_strassen_2x2x2()
    
    # Build 4x4x4 Strassen
    rank_444 = 49
    U44 = np.zeros((16, rank_444))
    V44 = np.zeros((16, rank_444))
    W44 = np.zeros((16, rank_444))
    
    col = 0
    for t in range(7):
        for s in range(7):
            for bi in range(2):
                for bj in range(2):
                    for bk in range(2):
                        for ei in range(2):
                            for ej in range(2):
                                for ek in range(2):
                                    u_idx = (bi*2+ei)*4 + (bj*2+ej)
                                    v_idx = (bj*2+ej)*4 + (bk*2+ek)
                                    w_idx = (bk*2+ek)*4 + (bi*2+ei)
                                    
                                    U44[u_idx, col] += u2[bi*2+bj, t] * u2[ei*2+ej, s]
                                    V44[v_idx, col] += v2[bj*2+bk, t] * v2[ej*2+ek, s]
                                    W44[w_idx, col] += w2[bk*2+bi, t] * w2[ek*2+ei, s]
            col += 1
    
    # Verify 4x4x4
    valid_444 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                expected = 1
                actual = 0
                for t in range(49):
                    actual += U44[i*4+j, t] * V44[j*4+k, t] * W44[k*4+i, t]
                if abs(actual - expected) > 1e-10:
                    valid_444 = False
                    break
            if not valid_444:
                break
        if not valid_444:
            break
    
    # 4x4x1 schoolbook
    rank_441 = 16
    U41 = np.zeros((16, rank_441))
    V41 = np.zeros((20, rank_441))
    W41 = np.zeros((20, rank_441))
    
    col = 0
    for i in range(4):
        for j in range(4):
            U41[i*4+j, col] = 1
            V41[j*5+4, col] = 1
            W41[4*4+i, col] = 1
            col += 1
    
    if valid_444:
        total_rank = rank_444 + rank_441
        U = np.zeros((16, total_rank))
        V = np.zeros((20, total_rank))
        W = np.zeros((20, total_rank))
        
        U[:, :rank_444] = U44
        V[:16, :rank_444] = V44
        W[:16, :rank_444] = W44
        
        U[:, rank_444:] = U41
        V[:, rank_444:] = V41
        W[:, rank_444:] = W41
        
        # Final verification
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    expected = 1
                    actual = 0
                    for t in range(total_rank):
                        actual += U[i*4+j, t] * V[j*5+k, t] * W[k*4+i, t]
                    if abs(actual - expected) > 1e-10:
                        print(f"Final error at ({i},{j},{k}): {actual} != {expected}")
                        valid_444 = False
                        break
                if not valid_444:
                    break
            if not valid_444:
                break
        
        if valid_444:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 80
    rank = 80
    U = np.zeros((16, rank))
    V = np.zeros((20, rank))
    W = np.zeros((20, rank))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(5):
                U[i*4+j, col] = 1
                V[j*5+k, col] = 1
                W[k*4+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
