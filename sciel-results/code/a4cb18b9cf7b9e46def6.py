import time

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (32, rank)
    
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set blocks in U, V, W
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # Define Strassen terms for one 4x4x4 block
    # A blocks:
    # A11: i in {0,1}, j in {0,1}
    # A12: i in {0,1}, j in {2,3}
    # A21: i in {2,3}, j in {0,1}
    # A22: i in {2,3}, j in {2,3}
    
    # B blocks (for first half, k in 0..3):
    # B11: j in {0,1}, k in {0,1}
    # B12: j in {0,1}, k in {2,3}
    # B21: j in {2,3}, k in {0,1}
    # B22: j in {2,3}, k in {2,3}
    
    # C blocks:
    # C11: i in {0,1}, k in {0,1}
    # C12: i in {0,1}, k in {2,3}
    # C21: i in {2,3}, k in {0,1}
    # C22: i in {2,3}, k in {2,3}
    
    def add_block_to_U(block_name, col, val):
        if block_name == 'A11':
            for i in range(2):
                for j in range(2):
                    U[i*m + j][col] += val
        elif block_name == 'A12':
            for i in range(2):
                for j in range(2, 4):
                    U[i*m + j][col] += val
        elif block_name == 'A21':
            for i in range(2, 4):
                for j in range(2):
                    U[i*m + j][col] += val
        elif block_name == 'A22':
            for i in range(2, 4):
                for j in range(2, 4):
                    U[i*m + j][col] += val

    def add_block_to_V(block_name, col, val, k_offset=0):
        # k_offset is 0 for first half, 4 for second half
        if block_name == 'B11':
            for j in range(2):
                for k in range(2):
                    V[j*p + (k + k_offset)][col] += val
        elif block_name == 'B12':
            for j in range(2):
                for k in range(2, 4):
                    V[j*p + (k + k_offset)][col] += val
        elif block_name == 'B21':
            for j in range(2, 4):
                for k in range(2):
                    V[j*p + (k + k_offset)][col] += val
        elif block_name == 'B22':
            for j in range(2, 4):
                for k in range(2, 4):
                    V[j*p + (k + k_offset)][col] += val

    def add_block_to_W(block_name, col, val):
        if block_name == 'C11':
            for i in range(2):
                for k in range(2):
                    W[k*n + i][col] += val
        elif block_name == 'C12':
            for i in range(2):
                for k in range(2, 4):
                    W[k*n + i][col] += val
        elif block_name == 'C21':
            for i in range(2, 4):
                for k in range(2):
                    W[k*n + i][col] += val
        elif block_name == 'C22':
            for i in range(2, 4):
                for k in range(2, 4):
                    W[k*n + i][col] += val

    # Term 1: M1 = (A11 + A22)(B11 + B22)
    # U: A11, A22
    # V: B11, B22
    # W: C11, C22
    col = 0
    add_block_to_U('A11', col, 1)
    add_block_to_U('A22', col, 1)
    add_block_to_V('B11', col, 1, 0)
    add_block_to_V('B22', col, 1, 0)
    add_block_to_W('C11', col, 1)
    add_block_to_W('C22', col, 1)
    
    # Term 2: M2 = (A21 + A22)B11
    # U: A21, A22
    # V: B11
    # W: C21, C22
    col = 1
    add_block_to_U('A21', col, 1)
    add_block_to_U('A22', col, 1)
    add_block_to_V('B11', col, 1, 0)
    add_block_to_W('C21', col, 1)
    add_block_to_W('C22', col, 1)
    
    # Term 3: M3 = A11(B12 - B22)
    # U: A11
    # V: B12, -B22
    # W: C12, C22
    col = 2
    add_block_to_U('A11', col, 1)
    add_block_to_V('B12', col, 1, 0)
    add_block_to_V('B22', col, -1, 0)
    add_block_to_W('C12', col, 1)
    add_block_to_W('C22', col, 1)
    
    # Term 4: M4 = A22(B21 - B11)
    # U: A22
    # V: B21, -B11
    # W: C11, C21
    col = 3
    add_block_to_U('A22', col, 1)
    add_block_to_V('B21', col, 1, 0)
    add_block_to_V('B11', col, -1, 0)
    add_block_to_W('C11', col, 1)
    add_block_to_W('C21', col, 1)
    
    # Term 5: M5 = (A11 + A12)B22
    # U: A11, A12
    # V: B22
    # W: -C11, C12
    col = 4
    add_block_to_U('A11', col, 1)
    add_block_to_U('A12', col, 1)
    add_block_to_V('B22', col, 1, 0)
    add_block_to_W('C11', col, -1)
    add_block_to_W('C12', col, 1)
    
    # Term 6: M6 = (A21 - A11)(B11 + B12)
    # U: A21, -A11
    # V: B11, B12
    # W: C22
    col = 5
    add_block_to_U('A21', col, 1)
    add_block_to_U('A11', col, -1)
    add_block_to_V('B11', col, 1, 0)
    add_block_to_V('B12', col, 1, 0)
    add_block_to_W('C22', col, 1)
    
    # Term 7: M7 = (A12 - A22)(B21 + B22)
    # U: A12, -A22
    # V: B21, B22
    # W: C11, C22
    col = 6
    add_block_to_U('A12', col, 1)
    add_block_to_U('A22', col, -1)
    add_block_to_V('B21', col, 1, 0)
    add_block_to_V('B22', col, 1, 0)
    add_block_to_W('C11', col, 1)
    add_block_to_W('C22', col, 1)
    
    # Now repeat for second half of B (k_offset = 4)
    # Same U and W, different V
    
    # Term 8: M1'
    col = 7
    add_block_to_U('A11', col, 1)
    add_block_to_U('A22', col, 1)
    add_block_to_V('B11', col, 1, 4)
    add_block_to_V('B22', col, 1, 4)
    add_block_to_W('C11', col, 1)
    add_block_to_W('C22', col, 1)
    
    # Term 9: M2'
    col = 8
    add_block_to_U('A21', col, 1)
    add_block_to_U('A22', col, 1)
    add_block_to_V('B11', col, 1, 4)
    add_block_to_W('C21', col, 1)
    add_block_to_W('C22', col, 1)
    
    # Term 10: M3'
    col = 9
    add_block_to_U('A11', col, 1)
    add_block_to_V('B12', col, 1, 4)
    add_block_to_V('B22', col, -1, 4)
    add_block_to_W('C12', col, 1)
    add_block_to_W('C22', col, 1)
    
    # Term 11: M4'
    col = 10
    add_block_to_U('A22', col, 1)
    add_block_to_V('B21', col, 1, 4)
    add_block_to_V('B11', col, -1, 4)
    add_block_to_W('C11', col, 1)
    add_block_to_W('C21', col, 1)
    
    # Term 12: M5'
    col = 11
    add_block_to_U('A11', col, 1)
    add_block_to_U('A12', col, 1)
    add_block_to_V('B22', col, 1, 4)
    add_block_to_W('C11', col, -1)
    add_block_to_W('C12', col, 1)
    
    # Term 13: M6'
    col = 12
    add_block_to_U('A21', col, 1)
    add_block_to_U('A11', col, -1)
    add_block_to_V('B11', col, 1, 4)
    add_block_to_V('B12', col, 1, 4)
    add_block_to_W('C22', col, 1)
    
    # Term 14: M7'
    col = 13
    add_block_to_U('A12', col, 1)
    add_block_to_U('A22', col, -1)
    add_block_to_V('B21', col, 1, 4)
    add_block_to_V('B22', col, 1, 4)
    add_block_to_W('C11', col, 1)
    add_block_to_W('C22', col, 1)
    
    return {"U": U, "V": V, "W": W}
