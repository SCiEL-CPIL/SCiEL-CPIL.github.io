import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Cache for computed decompositions
    cache = {}
    
    def get_decomp(nn, mm, pp):
        if (nn, mm, pp) in cache:
            return cache[(nn, mm, pp)]
            
        # Base cases
        if nn == 1 and mm == 1 and pp == 1:
            U = np.array([[1.0]])
            V = np.array([[1.0]])
            W = np.array([[1.0]])
            cache[(nn, mm, pp)] = (U, V, W)
            return U, V, W
            
        if nn == 1:
            # Rank mm * pp
            r = mm * pp
            U = np.zeros((nn * mm, r))
            V = np.zeros((mm * pp, r))
            W = np.zeros((pp * nn, r))
            col = 0
            for j in range(mm):
                for k in range(pp):
                    U[j, col] = 1
                    V[j * pp + k, col] = 1
                    W[k, col] = 1
                    col += 1
            cache[(nn, mm, pp)] = (U, V, W)
            return U, V, W
            
        if mm == 1:
            # Rank nn * pp
            r = nn * pp
            U = np.zeros((nn * mm, r))
            V = np.zeros((mm * pp, r))
            W = np.zeros((pp * nn, r))
            col = 0
            for i in range(nn):
                for k in range(pp):
                    U[i, col] = 1
                    V[k, col] = 1
                    W[k * nn + i, col] = 1
                    col += 1
            cache[(nn, mm, pp)] = (U, V, W)
            return U, V, W
            
        if pp == 1:
            # Rank nn * mm
            r = nn * mm
            U = np.zeros((nn * mm, r))
            V = np.zeros((mm * pp, r))
            W = np.zeros((pp * nn, r))
            col = 0
            for i in range(nn):
                for j in range(mm):
                    U[i * mm + j, col] = 1
                    V[j, col] = 1
                    W[i, col] = 1
                    col += 1
            cache[(nn, mm, pp)] = (U, V, W)
            return U, V, W
            
        if nn == 2 and mm == 2 and pp == 2:
            # Strassen
            # U, V, W for 2x2x2
            # Indices:
            # U: (4, 7) -> i in 0,1; j in 0,1
            # V: (4, 7) -> j in 0,1; k in 0,1
            # W: (4, 7) -> k in 0,1; i in 0,1
            U = np.zeros((4, 7))
            V = np.zeros((4, 7))
            W = np.zeros((4, 7))
            
            # Strassen multiplications:
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B21-B22)
            # M4 = A22(B11-B21)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            
            # A11 is U[0, :], A12 is U[1, :], A21 is U[2, :], A22 is U[3, :]
            # B11 is V[0, :], B12 is V[1, :], B21 is V[2, :], B22 is V[3, :]
            # C11 is W[1, :] (k=0, i=0 -> idx 0*2+0=0? No. W[k*n+i]. n=2.
            # W[0] -> k=0, i=0 (C11)
            # W[1] -> k=0, i=1 (C21)
            # W[2] -> k=1, i=0 (C12)
            # W[3] -> k=1, i=1 (C22)
            
            # M1
            U[0, 0] = 1; U[3, 0] = 1
            V[0, 0] = 1; V[3, 0] = 1
            W[0, 0] = 1; W[3, 0] = 1
            
            # M2
            U[2, 1] = 1; U[3, 1] = 1
            V[0, 1] = 1
            W[1, 1] = 1; W[3, 1] = 1
            
            # M3
            U[0, 2] = 1
            V[2, 2] = 1; V[3, 2] = -1
            W[2, 2] = 1; W[3, 2] = 1
            
            # M4
            U[3, 3] = 1
            V[0, 3] = 1; V[2, 3] = -1
            W[1, 3] = 1; W[3, 3] = 1
            
            # M5
            U[0, 4] = 1; U[1, 4] = 1
            V[3, 4] = 1
            W[0, 4] = 1; W[2, 4] = 1
            
            # M6
            U[2, 5] = 1; U[0, 5] = -1
            V[0, 5] = 1; V[1, 5] = 1
            W[1, 5] = 1; W[2, 5] = 1
            
            # M7
            U[1, 6] = 1; U[3, 6] = -1
            V[2, 6] = 1; V[3, 6] = 1
            W[0, 6] = 1; W[2, 6] = 1
            
            cache[(nn, mm, pp)] = (U, V, W)
            return U, V, W
            
        # Recursive splits
        best_rank = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # Split n
        for n1 in range(1, nn):
            n2 = nn - n1
            U1, V1, W1 = get_decomp(n1, mm, pp)
            U2, V2, W2 = get_decomp(n2, mm, pp)
            r = U1.shape[1] + U2.shape[1]
            if r < best_rank:
                # Pad and concatenate
                # U1 is (n1*mm, r1). Need (nn*mm, r1).
                # Rows 0..n1*mm-1 are U1. Rest 0.
                U1_pad = np.zeros((nn * mm, U1.shape[1]))
                U1_pad[:n1 * mm, :] = U1
                # V1 is (mm*pp, r1). No change in size.
                # W1 is (pp*n1, r1). Need (pp*nn, r1).
                # W1 rows correspond to k, i in 0..n1-1.
                # In W_pad, rows k*nn + i.
                # For i < n1, row is k*nn + i.
                # W1 row idx is k*n1 + i.
                # We need to map.
                W1_pad = np.zeros((pp * nn, W1.shape[1]))
                for k in range(pp):
                    for i in range(n1):
                        W1_pad[k * nn + i, :] = W1[k * n1 + i, :]
                
                U2_pad = np.zeros((nn * mm, U2.shape[1]))
                # U2 corresponds to i in n1..nn-1.
                # U2 row idx j' + i'*mm? No, U is (i, j).
                # U2 is for i' in 0..n2-1 (local). Global i = n1 + i'.
                # So U2 rows (i', j) map to (n1+i', j).
                for i2 in range(n2):
                    for j in range(mm):
                        U2_pad[(n1 + i2) * mm + j, :] = U2[i2 * mm + j, :]
                        
                V2_pad = V2 # Same size
                
                W2_pad = np.zeros((pp * nn, W2.shape[1]))
                for k in range(pp):
                    for i2 in range(n2):
                        W2_pad[k * nn + (n1 + i2), :] = W2[k * n2 + i2, :]
                        
                U_new = np.hstack([U1_pad, U2_pad])
                V_new = np.hstack([V1_pad, V2_pad])
                W_new = np.hstack([W1_pad, W2_pad])
                
                best_rank = r
                best_U, best_V, best_W = U_new, V_new, W_new

        # Split m
        for m1 in range(1, mm):
            m2 = mm - m1
            U1, V1, W1 = get_decomp(nn, m1, pp)
            U2, V2, W2 = get_decomp(nn, m2, pp)
            r = U1.shape[1] + U2.shape[1]
            if r < best_rank:
                # U1: (nn*m1, r1). Pad to (nn*mm, r1).
                # U1 rows (i, j) for j < m1.
                U1_pad = np.zeros((nn * mm, U1.shape[1]))
                for i in range(nn):
                    for j in range(m1):
                        U1_pad[i * mm + j, :] = U1[i * m1 + j, :]
                        
                # V1: (m1*pp, r1). Pad to (mm*pp, r1).
                # V1 rows (j, k) for j < m1.
                V1_pad = np.zeros((mm * pp, V1.shape[1]))
                for j in range(m1):
                    for k in range(pp):
                        V1_pad[j * pp + k, :] = V1[j * pp + k, :]
                        
                # W1: (pp*nn, r1). No change in size.
                
                U2_pad = np.zeros((nn * mm, U2.shape[1]))
                for i in range(nn):
                    for j2 in range(m2):
                        U2_pad[i * mm + (m1 + j2), :] = U2[i * m2 + j2, :]
                        
                V2_pad = np.zeros((mm * pp, V2.shape[1]))
                for j2 in range(m2):
                    for k in range(pp):
                        V2_pad[(m1 + j2) * pp + k, :] = V2[j2 * pp + k, :]
                        
                W2_pad = W2
                
                U_new = np.hstack([U1_pad, U2_pad])
                V_new = np.hstack([V1_pad, V2_pad])
                W_new = np.hstack([W1_pad, W2_pad])
                
                best_rank = r
                best_U, best_V, best_W = U_new, V_new, W_new

        # Split p
        for p1 in range(1, pp):
            p2 = pp - p1
            U1, V1, W1 = get_decomp(nn, mm, p1)
            U2, V2, W2 = get_decomp(nn, mm, p2)
            r = U1.shape[1] + U2.shape[1]
            if r < best_rank:
                # U1: (nn*mm, r1). No change.
                
                # V1: (mm*p1, r1). Pad to (mm*pp, r1).
                V1_pad = np.zeros((mm * pp, V1.shape[1]))
                for j in range(mm):
                    for k in range(p1):
                        V1_pad[j * pp + k, :] = V1[j * p1 + k, :]
                        
                # W1: (p1*nn, r1). Pad to (pp*nn, r1).
                W1_pad = np.zeros((pp * nn, W1.shape[1]))
                for k in range(p1):
                    for i in range(nn):
                        W1_pad[k * nn + i, :] = W1[k * nn + i, :]
                        
                U2_pad = U2
                
                V2_pad = np.zeros((mm * pp, V2.shape[1]))
                for j in range(mm):
                    for k2 in range(p2):
                        V2_pad[j * pp + (p1 + k2), :] = V2[j * p2 + k2, :]
                        
                W2_pad = np.zeros((pp * nn, W2.shape[1]))
                for k2 in range(p2):
                    for i in range(nn):
                        W2_pad[(p1 + k2) * nn + i, :] = W2[k2 * nn + i, :]
                        
                U_new = np.hstack([U1_pad, U2_pad])
                V_new = np.hstack([V1_pad, V2_pad])
                W_new = np.hstack([W1_pad, W2_pad])
                
                best_rank = r
                best_U, best_V, best_W = U_new, V_new, W_new

        cache[(nn, mm, pp)] = (best_U, best_V, best_W)
        return best_U, best_V, best_W

    U, V, W = get_decomp(n, m, p)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

