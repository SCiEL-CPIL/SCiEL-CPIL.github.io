cd /tmp && python3 - <<'PY'
import numpy as np, time
def try_decompose(n, m, p, R, iters=300, seed=0):
    rng = np.random.RandomState(seed)
    U = rng.randn(n*m, R); V = rng.randn(m*p, R); W = rng.randn(p*n, R)
    U/=U.std(); V/=V.std(); W/=W.std()
    def err():
        e=0.0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    s=0.0
                    for r in range(R):
                        s+=U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                    t=1.0 if i==k else 0.0
                    e+=(s-t)*(s-t)
        return e
    for it in range(iters):
        for i in range(n):
            for j in range(m):
                M=np.zeros((p,R))
                for k in range(p): M[k,:]=V[j*p+k,:]*W[k*n+i,:]
                b=np.array([1.0 if i==k else 0.0 for k in range(p)])
                U[i*m+j,:]=np.linalg.lstsq(M,b,rcond=None)[0]
        for j in range(m):
            for k in range(p):
                M=np.zeros((n,R))
                for i in range(n): M[i,:]=U[i*m+j,:]*W[k*n+i,:]
                b=np.array([1.0 if i==k else 0.0 for i in range(n)])
                V[j*p+k,:]=np.linalg.lstsq(M,b,rcond=None)[0]
        for k in range(p):
            for i in range(n):
                M=np.zeros((m,R))
                for j in range(m): M[j,:]=U[i*m+j,:]*V[j*p+k,:]
                b=np.array([1.0 if i==k else 0.0 for j in range(m)])
                W[k*n+i,:]=np.linalg.lstsq(M,b,rcond=None)[0]
    return U,V,W,err()
for R in [88,84,80,76,72]:
    t=time.time()
    U,V,W,e = try_decompose(3,4,8,R,iters=300,seed=0)
    print(R, "err=%.3e"%e, "%.1fs"%(time.time()-t))
PY
