import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    # Trivial rank-60 solution
    rank_trivial = n * m * p  # 60
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Build the rank-55 solution using Strassen for the 2x2 blocks in the middle
    def build_rank55():
        rank = 55
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        c = 0
        # Part A: j=0
        for i in range(n):
            for k in range(p):
                U[i*m+0, c] = 1
                V[0*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
        
        # Part B: A1*B1 + A2*B2
        for b in range(5):
            k0 = 2*b
            k1 = 2*b+1
            
            u11, u12 = 0*m+1, 0*m+2
            u21, u22 = 1*m+1, 1*m+2
            
            v11, v12 = 1*p+k0, 1*p+k1
            v21, v22 = 2*p+k0, 2*p+k1
            
            w00, w01 = k0*n+0, k1*n+0
            w10, w11 = k0*n+1, k1*n+1
            
            # Strassen for 2x2 * 2x2
            U[u11,c]+=1; U[u22,c]+=1; V[v11,c]+=1; V[v22,c]+=1; W[w00,c]+=1; W[w11,c]+=1; c+=1
            U[u21,c]+=1; U[u22,c]+=1; V[v11,c]+=1; W[w10,c]+=1; W[w11,c]+=-1; c+=1
            U[u11,c]+=1; V[v12,c]+=1; V[v22,c]+=-1; W[w01,c]+=1; W[w11,c]+=1; c+=1
            U[u22,c]+=1; V[v21,c]+=1; V[v11,c]+=-1; W[w00,c]+=1; W[w10,c]+=1; c+=1
            U[u11,c]+=1; U[u12,c]+=1; V[v22,c]+=1; W[w00,c]+=-1; W[w01,c]+=1; c+=1
            U[u21,c]+=1; U[u11,c]+=-1; V[v11,c]+=1; V[v12,c]+=1; W[w11,c]+=1; c+=1
            U[u12,c]+=1; U[u22,c]+=-1; V[v21,c]+=1; V[v22,c]+=1; W[w00,c]+=1; c+=1
        
        return U, V, W, rank
    
    # Try to find a rank-10 decomposition for M_{2,2,3}
    # If found, we can build a rank-54 solution for M_{2,3,10}
    # M_{2,2,10} = 3 * M_{2,2,3} + 1 * M_{2,2,1}
    # Rank(M_{2,2,1}) = 4
    # If Rank(M_{2,2,3}) = 10, then Rank(M_{2,2,10}) = 3*10 + 4 = 34
    # Total Rank = 20 (for j=0) + 34 = 54
    
    def find_m223_rank10():
        # M_{2,2,3} tensor
        # A: 2x2 (4 indices)
        # B: 2x3 (6 indices)
        # C: 3x2 (6 indices)
        # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i)
        # i in 0..1, j in 0..1, k in 0..2
        
        T_223 = np.zeros((4, 6, 6))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    a_idx = i*2 + j
                    b_idx = j*3 + k
                    c_idx = k*2 + i
                    T_223[a_idx, b_idx, c_idx] = 1.0
                    
        # Start with rank 11 solution (Strassen for k=0,1 + Trivial for k=2)
        # Strassen for k=0,1 uses columns 0,1 of B and rows 0,1 of C
        # Trivial for k=2 uses column 2 of B and row 2 of C
        
        U_223 = np.zeros((4, 11))
        V_223 = np.zeros((6, 11))
        W_223 = np.zeros((6, 11))
        
        c = 0
        # Strassen part for k=0,1
        # A indices: 0,1,2,3 (i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1)
        # B indices: 0,1 (j=0,k=0; j=0,k=1), 2,3 (j=1,k=0; j=1,k=1)
        # C indices: 0,1 (k=0,i=0; k=1,i=0), 2,3 (k=0,i=1; k=1,i=1)
        
        u00, u01, u10, u11 = 0, 1, 2, 3
        v00, v01, v10, v11 = 0, 1, 2, 3
        w00, w01, w10, w11 = 0, 1, 2, 3
        
        # M1
        U_223[u00,c]+=1; U_223[u11,c]+=1; V_223[v00,c]+=1; V_223[v11,c]+=1; W_223[w00,c]+=1; W_223[w11,c]+=1; c+=1
        # M2
        U_223[u10,c]+=1; U_223[u11,c]+=1; V_223[v00,c]+=1; W_223[w10,c]+=1; W_223[w11,c]+=-1; c+=1
        # M3
        U_223[u00,c]+=1; V_223[v01,c]+=1; V_223[v11,c]+=-1; W_223[w01,c]+=1; W_223[w11,c]+=1; c+=1
        # M4
        U_223[u11,c]+=1; V_223[v10,c]+=1; V_223[v00,c]+=-1; W_223[w00,c]+=1; W_223[w10,c]+=1; c+=1
        # M5
        U_223[u00,c]+=1; U_223[u01,c]+=1; V_223[v11,c]+=1; W_223[w00,c]+=-1; W_223[w01,c]+=1; c+=1
        # M6
        U_223[u10,c]+=1; U_223[u00,c]+=-1; V_223[v00,c]+=1; V_223[v01,c]+=1; W_223[w11,c]+=1; c+=1
        # M7
        U_223[u01,c]+=1; U_223[u11,c]+=-1; V_223[v10,c]+=1; V_223[v11,c]+=1; W_223[w00,c]+=1; c+=1
        
        # Trivial part for k=2
        # B indices: 4 (j=0,k=2), 5 (j=1,k=2)
        # C indices: 4 (k=2,i=0), 5 (k=2,i=1)
        
        # i=0, j=0, k=2
        U_223[u00,c]=1; V_223[v00+3,c]=1; W_223[w00+3,c]=1; c+=1
        # i=0, j=1, k=2
        U_223[u01,c]=1; V_223[v10+3,c]=1; W_223[w00+3,c]=1; c+=1
        # i=1, j=0, k=2
        U_223[u10,c]=1; V_223[v00+3,c]=1; W_223[w10+3,c]=1; c+=1
        # i=1, j=1, k=2
        U_223[u11,c]=1; V_223[v10+3,c]=1; W_223[w10+3,c]=1; c+=1
        
        def verify_223(U, V, W, r):
            for a in range(4):
                for b in range(6):
                    for cc in range(6):
                        val = np.sum(U[a,:r] * V[b,:r] * W[cc,:r])
                        if abs(val - T_223[a,b,cc]) > 1e-6:
                            return False
            return True
            
        if not verify_223(U_223, V_223, W_223, 11):
            return None
            
        # Try to reduce rank to 10
        # This is a hard problem. We will try a few random perturbations.
        # But given time, we might just return None and fallback to 55.
        # For now, assume we don't find it quickly.
        return None
        
    U, V, W, rank = build_rank55()
    
    # Verify
    def verify(U, V, W):
        r = U.shape[1]
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    val = np.sum(U[a,:r] * V[b,:r] * W[c,:r])
                    if abs(val - T[a,b,c]) > 1e-6:
                        return False
        return True
        
    if not verify(U, V, W):
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
                    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
