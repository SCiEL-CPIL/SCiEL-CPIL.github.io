def construct():
    import numpy as np
    n, m, p = 2, 4, 6
    nm, mp, pn = n*m, m*p, p*n
    rank = 42
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    def add_strassen(i0, j0, k0, off):
        a00, a01 = i0*m+j0, i0*m+j0+1
        a10, a11 = (i0+1)*m+j0, (i0+1)*m+j0+1
        b00, b01 = j0*p+k0, j0*p+k0+1
        b10, b11 = (j0+1)*p+k0, (j0+1)*p+k0+1
        w00, w01 = k0*n+i0, (k0+1)*n+i0
        w10, w11 = k0*n+i0+1, (k0+1)*n+i0+1
        
        U[a00,off]+=1;U[a11,off]+=1;V[b00,off]+=1;V[b11,off]+=1;W[w00,off]+=1;W[w11,off]+=1
        U[a10,off+1]+=1;U[a11,off+1]+=1;V[b00,off+1]+=1;W[w10,off+1]+=1;W[w11,off+1]-=1
        U[a00,off+2]+=1;V[b01,off+2]+=1;V[b11,off+2]-=1;W[w01,off+2]+=1;W[w11,off+2]+=1
        U[a11,off+3]+=1;V[b10,off+3]+=1;V[b00,off+3]-=1;W[w00,off+3]+=1;W[w10,off+3]+=1
        U[a00,off+4]+=1;U[a01,off+4]+=1;V[b11,off+4]+=1;W[w00,off+4]-=1;W[w01,off+4]+=1
        U[a10,off+5]+=1;U[a00,off+5]-=1;V[b00,off+5]+=1;V[b01,off+5]+=1;W[w11,off+5]+=1
        U[a01,off+6]+=1;U[a11,off+6]-=1;V[b10,off+6]+=1;V[b11,off+6]+=1;W[w00,off+6]+=1
    
    off=0
    for i0 in range(0,n,2):
        for j0 in range(0,m,2):
            for k0 in range(0,p,2):
                add_strassen(i0,j0,k0,off)
                off+=7
    
    return {"U":U.tolist(),"V":V.tolist(),"W":W.tolist()}
