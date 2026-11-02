"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_block_strassen(i_off, j_off, k_off):
        nonlocal col
        # 7 rank-1 tensors for 2x2x2
        # U indices: i*2+j (local)
        # V indices: j*2+k (local)
        # W indices: k*2+i (local)
        
        # M1
        u = [1, 0, 0, 1]
        v = [1, 0, 0, 1]
        w = [1, 0, 0, 1]
        for i in range(2):
            for j in range(2):
                if u[i*2+j] != 0:
                    U[i_off*2 + i*2 + j][col] += u[i*2+j]
        for j in range(2):
            for k in range(2):
                if v[j*2+k] != 0:
                    V[(j_off+j)*2 + k][col] += v[j*2+k]
        for k in range(2):
            for i in range(2):
                if w[k*2+i] != 0:
                    W[(k_off+k)*2 + i][col] += w[k*2+i]
        col += 1
        
        # M2
        u = [0, 0, 1, 1]
        v = [1, 0, 0, 0]
        w = [0, 1, 0, -1]
        for i in range(2):
            for j in range(2):
                if u[i*2+j] != 0:
                    U[i_off*2 + i*2 + j][col] += u[i*2+j]
        for j in range(2):
            for k in range(2):
                if v[j*2+k] != 0:
                    V[(j_off+j)*2 + k][col] += v[j*2+k]
        for k in range(2):
            for i in range(2):
                if w[k*2+i] != 0:
                    W[(k_off+k)*2 + i][col] += w[k*2+i]
        col += 1
        
        # M3
        u = [1, 0, 0, 0]
        v = [0, 1, 0, -1]
        w = [0, 0, 1, 1]
        for i in range(2):
            for j in range(2):
                if u[i*2+j] != 0:
                    U[i_off*2 + i*2 + j][col] += u[i*2+j]
        for j in range(2):
            for k in range(2):
                if v[j*2+k] != 0:
                    V[(j_off+j)*2 + k][col] += v[j*2+k]
        for k in range(2):
            for i in range(2):
                if w[k*2+i] != 0:
                    W[(k_off+k)*2 + i][col] += w[k*2+i]
        col += 1
        
        # M4
        u = [0, 0, 0, 1]
        v = [0, -1, 1, 0]
        w = [1, 1, 0, 0]
        for i in range(2):
            for j in range(2):
                if u[i*2+j] != 0:
                    U[i_off*2 + i*2 + j][col] += u[i*2+j]
        for j in range(2):
            for k in range(2):
                if v[j*2+k] != 0:
                    V[(j_off+j)*2 + k][col] += v[j*2+k]
        for k in range(2):
            for i in range(2):
                if w[k*2+i] != 0:
                    W[(k_off+k)*2 + i][col] += w[k*2+i]
        col += 1
        
        # M5
        u = [1, 1, 0, 0]
        v = [0, 0, 0, 1]
        w = [-1, 0, 1, 0]
        for i in range(2):
            for j in range(2):
                if u[i*2+j] != 0:
                    U[i_off*2 + i*2 + j][col] += u[i*2+j]
        for j in range(2):
            for k in range(2):
                if v[j*2+k] != 0:
                    V[(j_off+j)*2 + k][col] += v[j*2+k]
        for k in range(2):
            for i in range(2):
                if w[k*2+i] != 0:
                    W[(k_off+k)*2 + i][col] += w[k*2+i]
        col += 1
        
        # M6
        u = [0, 0, 1, -1]
        v = [1, 1, 0, 0]
        w = [1, 0, 0, 0]
        for i in range(2):
            for j in range(2):
                if u[i*2+j] != 0:
                    U[i_off*2 + i*2 + j][col] += u[i*2+j]
        for j in range(2):
            for k in range(2):
                if v[j*2+k] != 0:
                    V[(j_off+j)*2 + k][col] += v[j*2+k]
        for k in range(2):
            for i in range(2):
                if w[k*2+i] != 0:
                    W[(k_off+k)*2 + i][col] += w[k*2+i]
        col += 1
        
        # M7
        u = [0, 1, 0, -1]
        v = [0, 0, 1, 1]
        w = [0, 0, 0, 1]
        for i in range(2):
            for j in range(2):
                if u[i*2+j] != 0:
                    U[i_off*2 + i*2 + j][col] += u[i*2+j]
        for j in range(2):
            for k in range(2):
                if v[j*2+k] != 0:
                    V[(j_off+j)*2 + k][col] += v[j*2+k]
        for k in range(2):
            for i in range(2):
                if w[k*2+i] != 0:
                    W[(k_off+k)*2 + i][col] += w[k*2+i]
        col += 1

    def add_block_221(i_off, j_off, k_off):
        nonlocal col
        # 4 rank-1 tensors for 2x2x1
        for i in range(2):
            for j in range(2):
                u = [0]*4
                u[i*2+j] = 1
                v = [0]*2
                v[j] = 1
                w = [0]*2
                w[0] = 1 # k=0, i=i? No. W index is k*2+i. k=0. So index i.
                
                # Wait, W shape is p*n. For 2x2x1 block, p=1, n=2.
                # W index is k*n + i. k=0. Index i.
                # But global W index is (k_off+k)*n + i.
                # Here n=2.
                
                U[i_off*2 + i*2 + j][col] = 1
                V[(j_off+j)*1 + 0][col] = 1 # k=0
                W[(k_off+0)*2 + i][col] = 1
                col += 1

    # j blocks: 0, 2
    # k blocks: 0, 2, 4 (size 2), 6 (size 1)
    
    for j_off in [0, 2]:
        for k_off in [0, 2, 4]:
            add_block_strassen(0, j_off, k_off)
        add_block_221(0, j_off, 6)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
