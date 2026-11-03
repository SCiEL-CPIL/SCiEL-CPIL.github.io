"""Matrix multiplication 3x4x7 via block embedding of Strassen + schoolbook."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_222():
    """Rank-7 Strassen decomposition for 2x2x2. Returns U(4,7), V(4,7), W(4,7).
    U indexed by (i,k) with i in [0,2), k in [0,2) -> row i*2+k.
    V indexed by (k,j) -> row k*2+j.
    W indexed by (j,i) -> row j*2+i.  (matches W[k*n+i] style: first index is the
     'intermediate-then-output' ordering; here for a 2x2 block n=2 so W[j*2+i].)
    """
    # Standard Strassen: C = A*B for 2x2.
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[p,q],[r,s]]
    # M1=(b-c)(e+h); M2=(a+b)h; M3=a(e-f); M4=d(g-h); M5=(a+d)(e+h); M6=(c-d)f; M7=(b-c)(g+h)
    # C = [[M5+M4-M7+M6, M1+M7],[M1+M3, M5+M2-M4-M3]]
    # We build U (rows a,b,c,d indexed by (i,k): row0=a(i0k0),row1=b(i0k1),row2=c(i1k0),row3=d(i1k1))
    # V (rows e,f,g,h indexed by (k,j): row0=e(k0j0),row1=f(k0j1),row2=g(k1j0),row3=h(k1j1))
    # W (rows p,q,r,s indexed by (j,i): row0=p(j0i0),row1=q(j1i0),row2=r(j0i1),row3=s(j1i1))
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # column: (Ucoeffs, Vcoeffs, Wcoeffs)
    # M1 = (b-c)(e+h) -> U: b=+1,c=-1 ; V: e=+1,h=+1 ; W: q=+1,r=+1
    U[1,0]=1; U[2,0]=-1; V[0,0]=1; V[3,0]=1; W[1,0]=1; W[2,0]=1
    # M2 = (a+b)h -> U: a=+1,b=+1; V: h=+1; W: q=+1
    U[0,1]=1; U[1,1]=1; V[3,1]=1; W[1,1]=1
    # M3 = a(e-f) -> U: a=+1; V: e=+1,f=-1; W: r=+1
    U[0,2]=1; V[0,2]=1; V[1,2]=-1; W[2,2]=1
    # M4 = d(g-h) -> U: d=+1; V: g=+1,h=-1; W: p=+1,s=+1
    U[3,3]=1; V[2,3]=1; V[3,3]=-1; W[0,3]=1; W[3,3]=1
    # M5 = (a+d)(e+h) -> U: a=+1,d=+1; V: e=+1,h=+1; W: p=+1,s=+1
    U[0,4]=1; U[3,4]=1; V[0,4]=1; V[3,4]=1; W[0,4]=1; W[3,4]=1
    # M6 = (c-d)f -> U: c=+1,d=-1; V: f=+1; W: p=+1
    U[2,5]=1; U[3,5]=-1; V[1,5]=1; W[0,5]=1
    # M7 = (b-c)(g+h) -> U: b=+1,c=-1; V: g=+1,h=+1; W: p=+1,r=+1
    U[1,6]=1; U[2,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1; W[2,6]=1
    return U, V, W

def _schoolbook(n, m, p):
    """Rank n*m*p schoolbook for n x m times m x p.
    U (n*m, r) indexed (i,k)-> i*m+k ; V (m*p, r) indexed (k,j)-> k*p+j ; W (p*n, r) indexed (j,i)-> j*n+i.
    """
    r = n*m*p
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    c = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i*m + k, c] = 1
                V[k*p + j, c] = 1
                W[j*n + i, c] = 1
                c += 1
    return U, V, W

def _combine(blocks, n, m, p):
    """Combine a list of (i0,j0,k0, subU, subV, subW) blocks into full (n,m,p) factors.
    Each block multiplies A[i0:i0+ni, k0:k0+mk] * B[k0:k0+mk, j0:j0+mj].
    subU indexed (i,k) local -> global row (i0+i)*m + (k0+k)?? 
    Wait: U global row is (i, k) with i in [0,n), k in [0,m). For block at output-row i0, col k0:
      global U row = (i0+i)*m + (k0+k).
    subV indexed (k,j) local -> global V row = (k0+k)*p + (j0+j).
    subW indexed (j,i) local -> global W row = (j0+j)*n + (i0+i).
    """
    total_rank = sum(b[3].shape[1] for b in blocks)
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    off = 0
    for (i0, j0, k0, sU, sV, sW) in blocks:
        ni, mk, mj = sU.shape[0], sV.shape[0], sW.shape[0]
        # sU shape (ni*mk, r) local (i,k); sV (mk*mj, r) local (k,j); sW (mj*ni, r) local (j,i)
        r = sU.shape[1]
        for a in range(ni):
            for b in range(mk):
                # local U row a*mk+b -> global (i0+a)*m + (k0+b)
                for cc in range(r):
                    U[(i0+a)*m + (k0+b), off+cc] = sU[a*mk+b, cc]
        for b in range(mk):
            for c in range(mj):
                # local V row b*mj+c -> global (k0+b)*p + (j0+c)
                for cc in range(r):
                    V[(k0+b)*p + (j0+c), off+cc] = sV[b*mj+c, cc]
        for c in range(mj):
            for a in range(ni):
                # local W row c*ni+a -> global (j0+c)*n + (i0+a)
                for cc in range(r):
                    W[(j0+c)*n + (i0+a), off+cc] = sW[c*ni+a, cc]
        off += r
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Check sum_c U[i*m+k,c]*V[k*p+j,c]*W[j*n+i,c] == 1 for all i,k,j."""
    U = np.asarray(U, dtype=float); V = np.asarray(V, dtype=float); W = np.asarray(W, dtype=float)
    # Build tensor T[i,k,j] = sum_c
    # U row (i,k) = i*m+k ; V row (k,j)=k*p+j ; W row (j,i)=j*n+i
    # T[i,k,j] = sum_c U[i*m+k,c] V[k*p+j,c] W[j*n+i,c]
    # Efficient: for each c, outer product reshaped.
    T = np.zeros((n, m, p))
    for c in range(U.shape[1]):
        uc = U[:, c]; vc = V[:, c]; wc = W[:, c]
        # uc[i*m+k], vc[k*p+j], wc[j*n+i]
        Uc = uc.reshape(n, m)   # Uc[i,k]
        Vc = vc.reshape(m, p)   # Vc[k,j]
        Wc = wc.reshape(p, n)   # Wc[j,i]
        # T[i,k,j] += Uc[i,k]*Vc[k,j]*Wc[j,i]
        # = sum over j of (Uc[i,k]*Vc[k,j]) * Wc[j,i]
        # Build temp[i,k,j] = Uc[i,k]*Vc[k,j]
        tmp = np.einsum('ik,kj->ijk', Uc, Vc)
        T += tmp * Wc.T  # Wc.T[i,j] = Wc[j,i]
    return np.allclose(T, 1.0, atol=1e-9)

def construct():
    t0 = time.time()
    deadline = t0 + 110.0
    n, m, p = DIMENSIONS  # 3, 4, 7
    S = _strassen_222()
    blocks = []
    # Block (i0=0,j0=0,k0=0): 2x2x3 = Strassen(2x2x2) + schoolbook(2x2x1)
    # We need a 2x2x3 block: split k into 2+1.
    # Strassen part: A[0:2,0:2]*B[0:2,0:2] -> covers i in[0,2), k in[0,2), j in[0,2)
    blocks.append((0, 0, 0, S[0], S[1], S[2]))
    # schoolbook 2x2x1: A[0:2,2:3]*B[2:3,0:2] -> i in[0,2), k in[2,3), j in[0,2)
    blocks.append((0, 0, 2, _schoolbook(2,1,2)[0], _schoolbook(2,1,2)[1], _schoolbook(2,1,2)[2]))
    # Block (i0=0,j0=2,k0=0): 2x2x4 = Strassen(2x2x2) at k0=0 + Strassen(2x2x2) at k0=2
    blocks.append((0, 2, 0, S[0], S[1], S[2]))
    blocks.append((0, 2, 2, S[0], S[1], S[2]))
    # Block (i0=2,j0=0,k0=0): 1x2x3 schoolbook
    blocks.append((2, 0, 0, _schoolbook(1,3,2)[0], _schoolbook(1,3,2)[1], _schoolbook(1,3,2)[2]))
    # Block (i0=2,j0=2,k0=0): 1x2x4 schoolbook
    blocks.append((2, 2, 0, _schoolbook(1,4,2)[0], _schoolbook(1,4,2)[1], _schoolbook(1,4,2)[2]))
    U, V, W = _combine(blocks, n, m, p)
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook (rank 84)
        U, V, W = _schoolbook(n, m, p)
    # Convert to list of lists of int/float
    def to_list(M):
        M = np.asarray(M)
        # round near-integers
        M = np.round(M, 12)
        return M.tolist()
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
