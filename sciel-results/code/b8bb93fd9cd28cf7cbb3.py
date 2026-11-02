import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    rank = 66
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            raise OverflowError("Rank exceeded")
        for idx, coeff in u_coeffs.items():
            U[idx, col] += coeff
        for idx, coeff in v_coeffs.items():
            V[idx, col] += coeff
        for idx, coeff in w_coeffs.items():
            W[idx, col] += coeff
        col += 1
    
    def strassen_2x2x2(i0, j0, jb, k0, ci0, ck0):
        terms = [
            ({(i0,j0):1,(i0+1,j0+1):1}, {(jb,k0):1,(jb+1,k0+1):1}, {(ci0,ck0):1,(ci0+1,ck0+1):1}),
            ({(i0+1,j0):1,(i0+1,j0+1):1}, {(jb,k0):1}, {(ci0+1,ck0):1,(ci0+1,ck0+1):-1}),
            ({(i0,j0):1}, {(jb,k0+1):1,(jb+1,k0+1):-1}, {(ci0,ck0+1):1,(ci0+1,ck0+1):1}),
            ({(i0+1,j0+1):1}, {(jb+1,k0):1,(jb,k0):-1}, {(ci0,ck0):1,(ci0+1,ck0):1}),
            ({(i0,j0):1,(i0,j0+1):1}, {(jb+1,k0+1):1}, {(ci0,ck0):-1,(ci0,ck0+1):1}),
            ({(i0+1,j0):1,(i0,j0):-1}, {(jb,k0):1,(jb,k0+1):1}, {(ci0+1,ck0+1):1}),
            ({(i0,j0+1):1,(i0+1,j0+1):-1}, {(jb+1,k0):1,(jb+1,k0+1):1}, {(ci0,ck0):1}),
        ]
        for a, b, c in terms:
            add_term(
                {i*m+j: v for (i,j),v in a.items()},
                {j*p+k: v for (j,k),v in b.items()},
                {k*n+i: v for (i,k),v in c.items()}
            )
    
    def schoolbook_1x2x2(i0, j0, jb, k0, ci0, ck0):
        terms = [
            ({(i0,j0):1}, {(jb,k0):1}, {(ci0,ck0):1}),
            ({(i0,j0+1):1}, {(jb+1,k0):1}, {(ci0,ck0):1}),
            ({(i0,j0):1}, {(jb,k0+1):1}, {(ci0,ck0+1):1}),
            ({(i0,j0+1):1}, {(jb+1,k0+1):1}, {(ci0,ck0+1):1}),
        ]
        for a, b, c in terms:
            add_term(
                {i*m+j: v for (i,j),v in a.items()},
                {j*p+k: v for (j,k),v in b.items()},
                {k*n+i: v for (i,k),v in c.items()}
            )
    
    for b in range(3):
        ks = b * 2
        strassen_2x2x2(0, 0, 0, ks, 0, ks)
        strassen_2x2x2(0, 2, 2, ks, 0, ks)
        schoolbook_1x2x2(2, 0, 0, ks, 2, ks)
        schoolbook_1x2x2(2, 2, 2, ks, 2, ks)
    
    if col < rank:
        U, V, W = U[:,:col], V[:,:col], W[:,:col]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
