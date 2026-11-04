<<<<<<< SEARCH
    # Block 4: A(0:2, 2:3) x B(2:3, 0:5) -> 2x1x5
    # 10 terms: A(i,2) * B(2,k) -> C(i,k)
    for i in range(2):
        for k in range(5):
            col = 18 + i * 5 + k
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            
            U[u_idx][col] += 1
            V[v_idx][col] += 1
            W[w_idx][col] += 1

    return {"U": U, "V": V, "W": W}
=======
    # Block 4: A(0:2, 2:3) x B(2:3, 0:5) -> 2x1x5
    # Can be done with rank 1: U[i*3+2, 18]=1, V[2*5+k, 18]=1, W[k*2+i, 18]=1 for all i,k
    for i in range(2):
        u_idx = i * m + 2
        U[u_idx][18] += 1
    for k in range(5):
        v_idx = 2 * p + k
        V[v_idx][18] += 1
    for k in range(5):
        for i in range(2):
            w_idx = k * n + i
            W[w_idx][18] += 1

    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
