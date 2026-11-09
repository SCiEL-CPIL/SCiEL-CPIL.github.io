import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _schoolbook_decomp(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i*m+j]
                v_row = V_arr[j*p+k]
                w_row = W_arr[k*n+i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_decomp_2x2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}

def _combine_decompositions(decs, n, m, p):
    total_rank = sum(len(d['U'][0]) for d in decs)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0
    for d in decs:
        n_i, m_i, p_i = d['n'], d['m'], d['p']
        oi, oj, ok = d['offset_i'], d['offset_j'], d['offset_k']
        r_i = len(d['U'][0])
        
        for ii in range(n_i):
            for jj in range(m_i):
                for r in range(r_i):
                    U[(oi+ii)*m + (oj+jj)][rank_offset + r] = d['U'][ii*m_i+jj][r]
        
        for jj in range(m_i):
            for kk in range(p_i):
                for r in range(r_i):
                    V[(oj+jj)*p + (ok+kk)][rank_offset + r] = d['V'][jj*p_i+kk][r]
        
        for kk in range(p_i):
            for ii in range(n_i):
                for r in range(r_i):
                    W[(ok+kk)*n + (oi+ii)][rank_offset + r] = d['W'][kk*n_i+ii][r]
        
        rank_offset += r_i
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    
    decs = []
    
    s = _strassen_decomp_2x2x2()
    decs.append({**s, 'n': 2, 'm': 2, 'p': 2, 'offset_i': 0, 'offset_j': 0, 'offset_k': 0})
    
    s = _strassen_decomp_2x2x2()
    decs.append({**s, 'n': 2, 'm': 2, 'p': 2, 'offset_i': 0, 'offset_j': 2, 'offset_k': 0})
    
    s = _strassen_decomp_2x2x2()
    decs.append({**s, 'n': 2, 'm': 2, 'p': 2, 'offset_i': 0, 'offset_j': 2, 'offset_k': 2})
    
    sb = _schoolbook_decomp(2, 1, 2)
    decs.append({**sb, 'n': 2, 'm': 1, 'p': 2, 'offset_i': 0, 'offset_j': 4, 'offset_k': 0})
    
    sb = _schoolbook_decomp(2, 1, 2)
    decs.append({**sb, 'n': 2, 'm': 1, 'p': 2, 'offset_i': 0, 'offset_j': 4, 'offset_k': 2})
    
    sb = _schoolbook_decomp(1, 2, 2)
    decs.append({**sb, 'n': 1, 'm': 2, 'p': 2, 'offset_i': 2, 'offset_j': 0, 'offset_k': 0})
    
    sb = _schoolbook_decomp(1, 2, 2)
    decs.append({**sb, 'n': 1, 'm': 2, 'p': 2, 'offset_i': 2, 'offset_j': 2, 'offset_k': 0})
    
    sb = _schoolbook_decomp(1, 2, 2)
    decs.append({**sb, 'n': 1, 'm': 2, 'p': 2, 'offset_i': 2, 'offset_j': 2, 'offset_k': 2})
    
    sb = _schoolbook_decomp(1, 1, 2)
    decs.append({**sb, 'n': 1, 'm': 1, 'p': 2, 'offset_i': 2, 'offset_j': 4, 'offset_k': 0})
    
    sb = _schoolbook_decomp(1, 1, 2)
    decs.append({**sb, 'n': 1, 'm': 1, 'p': 2, 'offset_i': 2, 'offset_j': 4, 'offset_k': 2})
    
    result = _combine_decompositions(decs, n, m, p)
    
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        result = _schoolbook_decomp(n, m, p)
    
    return result
