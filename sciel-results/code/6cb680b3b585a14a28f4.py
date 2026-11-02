import numpy as np
import time

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=7
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (14, rank)
    # W: (p*n, rank) = (14, rank)
    
    def strassen_block(c0, c1, offset):
        """Return 7 rank-1 terms for Strassen on B columns c0, c1."""
        terms = []
        # M1 = (A00+A11)(B0c0+B1c1) -> C[0,c0]+, C[1,c1]+
        u = [0,0,0,0]; u[0]=1; u[3]=1
        v = [0]*14; v[c0]=1; v[7+c1]=1
        w = [0]*14; w[2*c0]=1; w[2*c1+1]=1
        terms.append((u,v,w))
        
        # M2 = (A10+A11)B0c0 -> C[1,c0]+, C[1,c1]-
        u = [0,0,0,0]; u[2]=1; u[3]=1
        v = [0]*14; v[c0]=1
        w = [0]*14; w[2*c0]=1; w[2*c1+1]=-1
        terms.append((u,v,w))
        
        # M3 = A00(B1c0-B1c1) -> C[0,c1]+, C[1,c1]+
        u = [0,0,0,0]; u[0]=1
        v = [0]*14; v[7+c0]=1; v[7+c1]=-1
        w = [0]*14; w[2*c1]=1; w[2*c1+1]=1
        terms.append((u,v,w))
        
        # M4 = A11(B0c1-B0c0) -> C[0,c0]+, C[1,c0]+
        u = [0,0,0,0]; u[3]=1
        v = [0]*14; v[c1]=1; v[c0]=-1
        w = [0]*14; w[2*c0]=1; w[2*c0+1]=1
        terms.append((u,v,w))
        
        # M5 = (A00+A01)B1c1 -> C[0,c0]-, C[0,c1]+
        u = [0,0,0,0]; u[0]=1; u[1]=1
        v = [0]*14; v[7+c1]=1
        w = [0]*14; w[2*c0]=-1; w[2*c1]=1
        terms.append((u,v,w))
        
        # M6 = (A10-A00)(B0c0+B1c0) -> C[1,c1]+
        u = [0,0,0,0]; u[2]=1; u[0]=-1
        v = [0]*14; v[c0]=1; v[7+c0]=1
        w = [0]*14; w[2*c1+1]=1
        terms.append((u,v,w))
        
        # M7 = (A01-A11)(B0c1+B1c1) -> C[0,c0]+
        u = [0,0,0,0]; u[1]=1; u[3]=-1
        v = [0]*14; v[c1]=1; v[7+c1]=1
        w = [0]*14; w[2*c0]=1
        terms.append((u,v,w))
        
        return terms
    
    def schoolbook_single(c, offset):
        """Return 4 rank-1 terms for schoolbook on B column c."""
        terms = []
        # A00*B0c -> C[0,c]
        u = [0,0,0,0]; u[0]=1
        v = [0]*14; v[c]=1
        w = [0]*14; w[2*c]=1
        terms.append((u,v,w))
        
        # A01*B1c -> C[0,c]
        u = [0,0,0,0]; u[1]=1
        v = [0]*14; v[7+c]=1
        w = [0]*14; w[2*c]=1
        terms.append((u,v,w))
        
        # A10*B0c -> C[1,c]
        u = [0,0,0,0]; u[2]=1
        v = [0]*14; v[c]=1
        w = [0]*14; w[2*c+1]=1
        terms.append((u,v,w))
        
        # A11*B1c -> C[1,c]
        u = [0,0,0,0]; u[3]=1
        v = [0]*14; v[7+c]=1
        w = [0]*14; w[2*c+1]=1
        terms.append((u,v,w))
        
        return terms
    
    all_terms = []
    all_terms.extend(strassen_block(0, 1, 0))
    all_terms.extend(strassen_block(2, 3, 0))
    all_terms.extend(strassen_block(4, 5, 0))
    all_terms.extend(schoolbook_single(6, 0))
    
    rank = len(all_terms)
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(14)]
    W = [[0]*rank for _ in range(14)]
    
    for r, (u, v, w) in enumerate(all_terms):
        for i in range(4):
            U[i][r] = u[i]
        for j in range(14):
            V[j][r] = v[j]
        for k in range(14):
            W[k][r] = w[k]
    
    return {"U": U, "V": V, "W": W}
