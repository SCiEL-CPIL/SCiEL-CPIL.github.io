import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 60
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a standard multiplication term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        U[u_idx, col] += coeff
        V[v_idx, col] += coeff
        W[w_idx, col] += coeff
        col += 1

    # Helper to add a Strassen 2x2 block
    # a_i, a_j: start indices for A block (row, col)
    # b_i, b_j: start indices for B block (row, col)
    # c_i, c_j: start indices for C block (row, col)
    def add_strassen(a_i, a_j, b_i, b_j, c_i, c_j):
        nonlocal col
        # m1 = (A00+A11)(B00+B11)
        U[a_i*m + a_j, col] = 1
        U[(a_i+1)*m + (a_j+1), col] = 1
        V[b_i*p + b_j, col] = 1
        V[(b_i+1)*p + (b_j+1), col] = 1
        W[c_j*n + c_i, col] = 1
        W[(c_j+1)*n + (c_i+1), col] = 1
        col += 1
        
        # m2 = (A10+A11)B00
        U[(a_i+1)*m + a_j, col] = 1
        U[(a_i+1)*m + (a_j+1), col] = 1
        V[b_i*p + b_j, col] = 1
        W[c_j*n + (c_i+1), col] = 1
        col += 1
        
        # m3 = A00(B01-B11)
        U[a_i*m + a_j, col] = 1
        V[b_i*p + (b_j+1), col] = 1
        V[(b_i+1)*p + (b_j+1), col] = -1
        W[(c_j+1)*n + c_i, col] = 1
        W[(c_j+1)*n + (c_i+1), col] = 1
        col += 1
        
        # m4 = A11(B10-B00)
        U[(a_i+1)*m + (a_j+1), col] = 1
        V[(b_i+1)*p + b_j, col] = 1
        V[b_i*p + b_j, col] = -1
        W[c_j*n + c_i, col] = 1
        W[c_j*n + (c_i+1), col] = 1
        col += 1
        
        # m5 = (A00+A01)B11
        U[a_i*m + a_j, col] = 1
        U[a_i*m + (a_j+1), col] = 1
        V[(b_i+1)*p + (b_j+1), col] = 1
        W[c_j*n + c_i, col] = -1
        W[(c_j+1)*n + c_i, col] = 1
        col += 1
        
        # m6 = (A10-A00)(B00+B01)
        U[(a_i+1)*m + a_j, col] = 1
        U[a_i*m + a_j, col] = -1
        V[b_i*p + b_j, col] = 1
        V[b_i*p + (b_j+1), col] = 1
        W[(c_j+1)*n + (c_i+1), col] = 1
        col += 1
        
        # m7 = (A01-A11)(B10+B11)
        U[a_i*m + (a_j+1), col] = 1
        U[(a_i+1)*m + (a_j+1), col] = -1
        V[(b_i+1)*p + b_j, col] = 1
        V[(b_i+1)*p + (b_j+1), col] = 1
        W[c_j*n + c_i, col] = 1
        W[c_j*n + (c_i+1), col] = 1
        col += 1

    # Helper to add 2x2 by 2x1 block
    def add_2x2_2x1(a_i, a_j, b_i, b_j, c_i, c_j):
        nonlocal col
        # C0 = A00*B0 + A01*B1
        add_term(a_i*m + a_j, b_i*p + b_j, c_j*n + c_i)
        add_term(a_i*m + (a_j+1), (b_i+1)*p + b_j, c_j*n + c_i)
        # C1 = A10*B0 + A11*B1
        add_term((a_i+1)*m + a_j, b_i*p + b_j, c_j*n + (c_i+1))
        add_term((a_i+1)*m + (a_j+1), (b_i+1)*p + b_j, c_j*n + (c_i+1))

    # Helper to add 1x2 by 2xP block
    def add_1x2_2xP(a_i, a_j, b_i, b_j, c_i, c_j, P):
        nonlocal col
        for k in range(P):
            add_term(a_i*m + a_j, b_i*p + (b_j+k), (c_j+k)*n + c_i)
            add_term(a_i*m + (a_j+1), (b_i+1)*p + (b_j+k), (c_j+k)*n + c_i)

    # Helper to add 2x1 by 1xP block
    def add_2x1_1xP(a_i, a_j, b_i, b_j, c_i, c_j, P):
        nonlocal col
        for i_off in range(2):
            for k in range(P):
                add_term((a_i+i_off)*m + a_j, b_i*p + (b_j+k), (c_j+k)*n + (c_i+i_off))

    # Helper to add 1x1 by 1xP block
    def add_1x1_1xP(a_i, a_j, b_i, b_j, c_i, c_j, P):
        nonlocal col
        for k in range(P):
            add_term(a_i*m + a_j, b_i*p + (b_j+k), (c_j+k)*n + c_i)

    # 1. A11 (0,0) * B11a (0,0) -> C (0,0) 2x2 by 2x2
    add_strassen(0, 0, 0, 0, 0, 0)
    
    # 2. A11 (0,0) * B11b (0,2) -> C (0,2) 2x2 by 2x2
    add_strassen(0, 0, 0, 2, 0, 2)
    
    # 3. A11 (0,0) * B12a (0,4) -> C (0,4) 2x2 by 2x2
    add_strassen(0, 0, 0, 4, 0, 4)
    
    # 4. A11 (0,0) * B12b (0,6) -> C (0,6) 2x2 by 2x1
    add_2x2_2x1(0, 0, 0, 6, 0, 6)
    
    # 5. A21 (2,0) * B11a (0,0) -> C (2,0) 1x2 by 2x2
    add_1x2_2xP(2, 0, 0, 0, 2, 0, 2)
    
    # 6. A21 (2,0) * B11b (0,2) -> C (2,2) 1x2 by 2x2
    add_1x2_2xP(2, 0, 0, 2, 2, 2, 2)
    
    # 7. A21 (2,0) * B12a (0,4) -> C (2,4) 1x2 by 2x2
    add_1x2_2xP(2, 0, 0, 4, 2, 4, 2)
    
    # 8. A21 (2,0) * B12b (0,6) -> C (2,6) 1x2 by 2x1
    add_1x2_2xP(2, 0, 0, 6, 2, 6, 1)
    
    # 9. A12 (0,2) * B21a (2,0) -> C (0,0) 2x1 by 1x2
    add_2x1_1xP(0, 2, 2, 0, 0, 0, 2)
    
    # 10. A12 (0,2) * B21b (2,2) -> C (0,2) 2x1 by 1x2
    add_2x1_1xP(0, 2, 2, 2, 0, 2, 2)
    
    # 11. A12 (0,2) * B22a (2,4) -> C (0,4) 2x1 by 1x2
    add_2x1_1xP(0, 2, 2, 4, 0, 4, 2)
    
    # 12. A12 (0,2) * B22b (2,6) -> C (0,6) 2x1 by 1x1
    add_2x1_1xP(0, 2, 2, 6, 0, 6, 1)
    
    # 13. A22 (2,2) * B21a (2,0) -> C (2,0) 1x1 by 1x2
    add_1x1_1xP(2, 2, 2, 0, 2, 0, 2)
    
    # 14. A22 (2,2) * B21b (2,2) -> C (2,2) 1x1 by 1x2
    add_1x1_1xP(2, 2, 2, 2, 2, 2, 2)
    
    # 15. A22 (2,2) * B22a (2,4) -> C (2,4) 1x1 by 1x2
    add_1x1_1xP(2, 2, 2, 4, 2, 4, 2)
    
    # 16. A22 (2,2) * B22b (2,6) -> C (2,6) 1x1 by 1x1
    add_1x1_1xP(2, 2, 2, 6, 2, 6, 1)

    return {"U": U, "V": V, "W": W}
