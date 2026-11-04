    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank_total):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                err += (val - T[i,j,k])**2
    
    if err < 1e-8:
        return {"U": U, "V": V, "W": W}
