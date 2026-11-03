import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal rank
        # Local indices for 2x2 blocks
        u00 = a_rows[0]*m + a_cols[0]
        u01 = a_rows[0]*m + a_cols[1]
        u10 = a_rows[1]*m + a_cols[0]
        u11 = a_rows[1]*m + a_cols[1]
        
        v00 = b_rows[0]*p + b_cols[0]
        v01 = b_rows[0]*p + b_cols[1]
        v10 = b_rows[1]*p + b_cols[0]
        v11 = b_rows[1]*p + b_cols[1]
        
        w00 = c_cols[0]*n + c_rows[0]
        w01 = c_cols[1]*n + c_rows[0]
        w10 = c_cols[0]*n + c_rows[1]
        w11 = c_cols[1]*n + c_rows[1]
        
        # M1 = (P00+P11)*(Q00+Q11)
        U[u00, rank] += 1; U[u11, rank] += 1
        V[v00, rank] += 1; V[v11, rank] += 1
        W[w00, rank] += 1; W[w11, rank] += 1
        rank += 1
        
        # M2 = (P10+P11)*Q00
        U[u10, rank] += 1; U[u11, rank] += 1
        V[v00, rank] += 1
        W[w10, rank] += 1; W[w11, rank] -= 1
        rank += 1
        
        # M3 = P00*(Q01-Q11)
        U[u00, rank] += 1
        V[v01, rank] += 1; V[v11, rank] -= 1
        W[w01, rank] += 1; W[w11, rank] += 1
        rank += 1
        
        # M4 = P11*(Q10-Q00)
        U[u11, rank] += 1
        V[v10, rank] += 1; V[v00, rank] -= 1
        W[w00, rank] += 1; W[w10, rank] += 1
        rank += 1
        
        # M5 = (P00+P01)*Q11
        U[u00, rank] += 1; U[u01, rank] += 1
        V[v11, rank] += 1
        W[w01, rank] += 1; W[w00, rank] -= 1
        rank += 1
        
        # M6 = (P10-P00)*(Q00+Q01)
        U[u10, rank] += 1; U[u00, rank] -= 1
        V[v00, rank] += 1; V[v01, rank] += 1
        W[w11, rank] += 1
        rank += 1
        
        # M7 = (P01-P11)*(Q10+Q11)
        U[u01, rank] += 1; U[u11, rank] -= 1
        V[v10, rank] += 1; V[v11, rank] += 1
        W[w00, rank] += 1
        rank += 1
    
    # Four 2x2x2 blocks
    add_strassen([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    add_strassen([0,1], [2,3], [2,3], [0,1], [0,1], [0,1])
    add_strassen([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    add_strassen([0,1], [2,3], [2,3], [2,3], [0,1], [2,3])
    
    assert rank == 28
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
