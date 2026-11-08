def check(U,V,W):
    U=np.array(U);V=np.array(V);W=np.array(W)
    T = np.einsum('ir,kr,jr->ijk', U, W, V)
    n,m,p=4,4,8
    ref = np.zeros((n,m,p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ref[i,j,k]=1
    return np.allclose(T,ref), np.max(np.abs(T-ref))

U,V,W = construct()['U'],construct()['V'],construct()['W']
print(check(U,V,W))
