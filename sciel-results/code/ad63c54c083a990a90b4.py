import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 10.0
    
    n, m, p = 2, 3, 5
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1
        U[i_off * m + j_off][col] = 1
        U[(i_off + 1) * m + (j_off + 1)][col] = 1
        V[j_off * p + k_off][col] = 1
        V[(j_off + 1) * p + (k_off + 1)][col] = 1
        W[k_off * n + i_off][col] = 1
        W[(k_off + 1) * n + (i_off + 1)][col] = 1
        col += 1
        
        # M2
        U[(i_off + 1) * m + j_off][col] = 1
        U[(i_off + 1) * m + (j_off + 1)][col] = 1
        V[j_off * p + k_off][col] = 1
        W[k_off * n + (i_off + 1)][col] = 1
        W[(k_off + 1) * n + (i_off + 1)][col] = -1
        col += 1
        
        # M3
        U[i_off * m + j_off][col] = 1
        V[j_off * p + (k_off + 1)][col] = 1
        V[(j_off + 1) * p + (k_off + 1)][col] = -1
        W[(k_off + 1) * n + i_off][col] = 1
        W[(k_off + 1) * n + (i_off + 1)][col] = 1
        col += 1
        
        # M4
        U[(i_off + 1) * m + (j_off + 1)][col] = 1
        V[(j_off + 1) * p + k_off][col] = 1
        V[j_off * p + k_off][col] = -1
        W[k_off * n + i_off][col] = 1
        W[k_off * n + (i_off + 1)][col] = 1
        col += 1
        
        # M5
        U[i_off * m + j_off][col] = 1
        U[i_off * m + (j_off + 1)][col] = 1
        V[(j_off + 1) * p + (k_off + 1)][col] = 1
        W[k_off * n + i_off][col] = -1
        W[(k_off + 1) * n + i_off][col] = 1
        col += 1
        
        # M6
        U[(i_off + 1) * m + j_off][col] = 1
        U[i_off * m + j_off][col] = -1
        V[j_off * p + k_off][col] = 1
        V[j_off * p + (k_off + 1)][col] = 1
        W[(k_off + 1) * n + (i_off + 1)][col] = 1
        col += 1
        
        # M7
        U[i_off * m + (j_off + 1)][col] = 1
        U[(i_off + 1) * m + (j_off + 1)][col] = -1
        V[(j_off + 1) * p + k_off][col] = 1
        V[(j_off + 1) * p + (k_off + 1)][col] = 1
        W[k_off * n + i_off][col] = 1
        col += 1

    # Block 1: j=0
    for i in range(n):
        for k in range(p):
            U[i * m][col] = 1
            V[k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 2: j=1,2
    add_strassen(0, 1, 0)
    add_strassen(0, 1, 2)
    
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    assert col == rank
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
