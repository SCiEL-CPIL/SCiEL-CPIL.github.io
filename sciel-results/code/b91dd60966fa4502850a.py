import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 4, 6
    
    def build_strassen_42():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen_terms = [
            {'A': [(0,0,1), (1,1,1)], 'B': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
            {'A': [(1,0,1), (1,1,1)], 'B': [(0,0,1)], 'W': [(1,0,1), (1,1,-1)]},
            {'A': [(0,0,1)], 'B': [(0,1,1), (1,1,-1)], 'W': [(0,1,1), (1,1,1)]},
            {'A': [(1,1,1)], 'B': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (1,0,1)]},
            {'A': [(0,0,1), (0,1,1)], 'B': [(1,1,1)], 'W': [(0,0,-1), (0,1,1)]},
            {'A': [(1,0,1), (0,0,-1)], 'B': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
            {'A': [(0,1,1), (1,1,-1)], 'B': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]},
        ]
        
        col = 0
        for j_block in range(2):
            for k_block in range(3):
                for term in strassen_terms:
                    r = col
                    for (ar, ac, sign) in term['A']:
                        u_idx = ar * m + (j_block * 2 + ac)
                        U[u_idx][r] += sign
                    for (br, bc, sign) in term['B']:
                        v_idx = (j_block * 2 + br) * p + (k_block * 2 + bc)
                        V[v_idx][r] += sign
                    for (or_, oc, sign) in term['W']:
                        w_idx = (k_block * 2 + oc) * n + or_
                        W[w_idx][r] += sign
                    col += 1
        
        assert col == rank, f"Expected {rank} columns, got {col}"
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(r):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if s != 1:
                        return False
        return True
    
    fallback = build_strassen_42()
    assert verify(fallback["U"], fallback["V"], fallback["W"]), "Strassen-42 verification failed"
    
    return fallback

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
