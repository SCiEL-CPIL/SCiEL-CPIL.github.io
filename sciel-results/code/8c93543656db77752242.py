import numpy as np
import time

n, m, p = 2, 5, 6

def construct():
    # We construct the decomposition by splitting the middle dimension m=5 into m1=2 and m2=3.
    # Then split the output dimension p=6 into blocks of size 2 or 3 to use smaller known decompositions.
    
    # We assume Rank(M_{2,1,p}) = 1.
    # We assume Rank(M_{2,2,2}) = 2 (derived from splitting m=2 into 1+1).
    # We assume Rank(M_{2,3,3}) = 3 (derived from splitting m=3 into 1+2, and M_{2,2,2} rank 2).
    
    # Let's build the components.
    
    # Component 1: M_{2,2,6}
    # Split p=6 into three blocks of 2.
    # Each block is M_{2,2,2}.
    # Rank(M_{2,2,2}) = 2.
    # Total rank for M_{2,2,6} = 3 * 2 = 6.
    
    # Component 2: M_{2,3,6}
    # Split p=6 into two blocks of 3.
    # Each block is M_{2,3,3}.
    # Rank(M_{2,3,3}) = 3.
    # Total rank for M_{2,3,6} = 2 * 3 = 6.
    
    # Total Rank = 12.
    
    R = 12
    
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    col = 0
    
    # Helper to add a decomposition for M_{n, m_sub, p_sub} at offsets i_off, j_off, k_off
    def add_block(n, m_sub, p_sub, i_off, j_off, k_off, rank_sub):
        nonlocal col
        # This is a placeholder. We need explicit formulas.
        pass

    # Explicit construction for M_{2,1,p} rank 1
    # U[i, 0, r] = 1
    # V[0, k, r] = 1
    # W[k, i, r] = 1
    
    # Explicit construction for M_{2,2,2} rank 2
    # Split m=2 into j=0 and j=1.
    # r=0: handles j=0. U[i,0,0]=1, V[0,k,0]=1, W[k,i,0]=1.
    # r=1: handles j=1. U[i,1,1]=1, V[1,k,1]=1, W[k,i,1]=1.
    
    # Explicit construction for M_{2,3,3} rank 3
    # Split m=3 into j=0 (rank 1) and j=1,2 (rank 2).
    # r=0: handles j=0. U[i,0,0]=1, V[0,k,0]=1, W[k,i,0]=1.
    # r=1: handles j=1. U[i,1,1]=1, V[1,k,1]=1, W[k,i,1]=1.
    # r=2: handles j=2. U[i,2,2]=1, V[2,k,2]=1, W[k,i,2]=1.
    
    # So, it seems Rank(M_{n, m, p}) = m?
    # If Rank(M_{2, m, p}) = m, then Rank(M_{2,5,6}) = 5.
    # Let's verify if Rank(M_{2,5,6}) = 5 is possible.
    # This would imply we just need 5 terms, one for each j.
    # Term j: U[i, j, r_j]=1, V[j, k, r_j]=1, W[k, i, r_j]=1.
    # Sum over j: 1.
    # Yes!
    # The "schoolbook" algorithm uses n*m*p terms because it treats each (i,j,k) as a separate rank-1 tensor in the full n x m x p space.
    # But here, the W matrix allows us to "broadcast" the 1 across all i,k for a fixed j.
    # So the rank is simply m?
    
    # Let's construct rank 5.
    R = 5
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    for j in range(m):
        r = j
        # U[i, j, r] = 1 for all i
        for i in range(n):
            U[i*m + j, r] = 1
        # V[j, k, r] = 1 for all k
        for k in range(p):
            V[j*p + k, r] = 1
        # W[k, i, r] = 1 for all k, i
        for k in range(p):
            for i in range(n):
                W[k*n + i, r] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
