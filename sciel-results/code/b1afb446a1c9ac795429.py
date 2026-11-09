import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 * (1 * 2 * 4) = 56
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to get local alpha for A[i, j]
    def a_alpha(i, j):
        if i == 0:
            if j < 2: return j
            else: return j - 2
        else:
            if j < 2: return j
            else: return j - 2
            
    # Helper to get local alpha, beta for B[j, k]
    def b_alpha_beta(j, k):
        alpha = j if j < 2 else j - 2
        beta = k if k < 4 else k - 4
        return alpha, beta
        
    # Helper to get local beta for C[i_c, k_c]
    def c_beta(i_c, k_c):
        return k_c if k_c < 4 else k_c - 4

    # Strassen coefficients
    # u_coeffs[t][i][j]
    # v_coeffs[t][j][k]
    # w_coeffs[t][i_c][k_c]
    
    u_coeffs = [{} for _ in range(7)]
    v_coeffs = [{} for _ in range(7)]
    w_coeffs = [{} for _ in range(7)]
    
    # M1: (A11 + A22)(B11 + B22)
    u_coeffs[0][(0,0)] = 1; u_coeffs[0][(0,1)] = 1
    u_coeffs[0][(1,2)] = 1; u_coeffs[0][(1,3)] = 1
    for j in range(2):
        for k in range(4):
            v_coeffs[0][(j,k)] = 1
            v_coeffs[0][(j+2, k+4)] = 1
    for k_c in range(4):
        w_coeffs[0][(0, k_c)] = 1
        w_coeffs[0][(1, k_c+4)] = 1
        
    # M2: (A21 + A22)B11
    u_coeffs[1][(1,0)] = 1; u_coeffs[1][(1,1)] = 1
    u_coeffs[1][(1,2)] = 1; u_coeffs[1][(1,3)] = 1
    for j in range(2):
        for k in range(4):
            v_coeffs[1][(j,k)] = 1
    for k_c in range(4):
        w_coeffs[1][(1, k_c)] = 1
        
    # M3: A11(B12 - B22)
    u_coeffs[2][(0,0)] = 1; u_coeffs[2][(0,1)] = 1
    for j in range(2):
        for k in range(4):
            v_coeffs[2][(j, k+4)] = 1
            v_coeffs[2][(j+2, k+4)] = -1
    for k_c in range(4):
        w_coeffs[2][(0, k_c+4)] = 1
        
    # M4: A22(B21 - B11)
    u_coeffs[3][(1,2)] = 1; u_coeffs[3][(1,3)] = 1
    for j in range(2):
        for k in range(4):
            v_coeffs[3][(j+2, k)] = 1
            v_coeffs[3][(j, k)] = -1
    for k_c in range(4):
        w_coeffs[3][(0, k_c)] = 1
        w_coeffs[3][(1, k_c)] = 1
        
    # M5: (A11 + A12)B22
    u_coeffs[4][(0,0)] = 1; u_coeffs[4][(0,1)] = 1
    u_coeffs[4][(0,2)] = 1; u_coeffs[4][(0,3)] = 1
    for j in range(2):
        for k in range(4):
            v_coeffs[4][(j+2, k+4)] = 1
    for k_c in range(4):
        w_coeffs[4][(0, k_c)] = -1
        w_coeffs[4][(0, k_c+4)] = 1
        
    # M6: (A21 - A11)(B11 + B12)
    u_coeffs[5][(1,0)] = 1; u_coeffs[5][(1,1)] = 1
    u_coeffs[5][(0,0)] = -1; u_coeffs[5][(0,1)] = -1
    for j in range(2):
        for k in range(4):
            v_coeffs[5][(j, k)] = 1
            v_coeffs[5][(j, k+4)] = 1
    for k_c in range(4):
        w_coeffs[5][(1, k_c+4)] = 1
        
    # M7: (A12 - A22)(B21 + B22)
    u_coeffs[6][(0,2)] = 1; u_coeffs[6][(0,3)] = 1
    u_coeffs[6][(1,2)] = -1; u_coeffs[6][(1,3)] = -1
    for j in range(2):
        for k in range(4):
            v_coeffs[6][(j+2, k)] = 1
            v_coeffs[6][(j+2, k+4)] = 1
    for k_c in range(4):
        w_coeffs[6][(0, k_c)] = 1
        w_coeffs[6][(1, k_c+4)] = 1

    # Construct U, V, W
    for t in range(7):
        for alpha in range(2):
            for beta in range(4):
                # Get all (i,j) for this t, alpha
                u_items = []
                for (i,j), val in u_coeffs[t].items():
                    if a_alpha(i,j) == alpha:
                        u_items.append((i,j,val))
                        
                # Get all (j',k') for this t, alpha, beta
                v_items = []
                for (j,k), val in v_coeffs[t].items():
                    a, b = b_alpha_beta(j,k)
                    if a == alpha and b == beta:
                        v_items.append((j,k,val))
                        
                # Get all (i_c,k_c) for this t, beta
                w_items = []
                for (i_c,k_c), val in w_coeffs[t].items():
                    if c_beta(i_c,k_c) == beta:
                        w_items.append((i_c,k_c,val))
                        
                for i,j,u_val in u_items:
                    for j_prime,k_prime,v_val in v_items:
                        for i_c,k_c,w_val in w_items:
                            if col >= rank:
                                break
                            U[i*m + j][col] = u_val
                            V[j_prime*p + k_prime][col] = v_val
                            W[k_c*n + i_c][col] = w_val
                            col += 1
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
