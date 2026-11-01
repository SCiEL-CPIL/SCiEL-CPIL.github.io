"""FTW-inspired decomposition for 2x4x6 matrix multiplication.

Uses Kronecker product of Strassen (2x2x2, rank 7) and a rank-1 (1x2x3) core.
Total rank = 7 * 1 = 7.
"""
import numpy as np

def construct():
    n, m, p = 2, 4, 6
    
    # Strassen decomposition for 2x2x2
    # Standard Strassen 7 products
    # X (2x2) * Y (2x2) = Z (2x2)
    # U1[i*2+j, s], V1[j*2+k, s], W1[k*2+i, s]
    
    # Strassen products:
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y01-Y11)
    # M4 = X11*(Y10-Y00)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X01-X11)*(Y10+Y11)
    #
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    rank1 = 7
    U1 = np.zeros((4, rank1))
    V1 = np.zeros((4, rank1))
    W1 = np.zeros((4, rank1))
    
    # Product 1: (X00+X11)*(Y00+Y11)
    U1[0, 0] = 1  # X00: i=0,j=0
    U1[3, 0] = 1  # X11: i=1,j=1
    V1[0, 0] = 1  # Y00: j=0,k=0
    V1[3, 0] = 1  # Y11: j=1,k=1
    W1[0, 0] = 1  # Z00: k=0,i=0
    W1[3, 0] = 1  # Z11: k=1,i=1
    
    # Product 2: (X10+X11)*Y00
    U1[2, 1] = 1  # X10: i=1,j=0
    U1[3, 1] = 1  # X11: i=1,j=1
    V1[0, 1] = 1  # Y00: j=0,k=0
    W1[2, 1] = 1  # Z10: k=0,i=1
    W1[3, 1] = -1 # Z11: k=1,i=1
    
    # Product 3: X00*(Y01-Y11)
    U1[0, 2] = 1  # X00: i=0,j=0
    V1[1, 2] = 1  # Y01: j=0,k=1
    V1[3, 2] = -1 # Y11: j=1,k=1
    W1[1, 2] = 1  # Z01: k=1,i=0
    W1[3, 2] = 1  # Z11: k=1,i=1
    
    # Product 4: X11*(Y10-Y00)
    U1[3, 3] = 1  # X11: i=1,j=1
    V1[2, 3] = 1  # Y10: j=1,k=0
    V1[0, 3] = -1 # Y00: j=0,k=0
    W1[0, 3] = 1  # Z00: k=0,i=0
    W1[2, 3] = 1  # Z10: k=0,i=1
    
    # Product 5: (X00+X01)*Y11
    U1[0, 4] = 1  # X00: i=0,j=0
    U1[1, 4] = 1  # X01: i=0,j=1
    V1[3, 4] = 1  # Y11: j=1,k=1
    W1[0, 4] = -1 # Z00: k=0,i=0
    W1[1, 4] = 1  # Z01: k=1,i=0
    
    # Product 6: (X10-X00)*(Y00+Y01)
    U1[2, 5] = 1  # X10: i=1,j=0
    U1[0, 5] = -1 # X00: i=0,j=0
    V1[0, 5] = 1  # Y00: j=0,k=0
    V1[1, 5] = 1  # Y01: j=0,k=1
    W1[3, 5] = 1  # Z11: k=1,i=1
    
    # Product 7: (X01-X11)*(Y10+Y11)
    U1[1, 6] = 1  # X01: i=0,j=1
    U1[3, 6] = -1 # X11: i=1,j=1
    V1[2, 6] = 1  # Y10: j=1,k=0
    V1[3, 6] = 1  # Y11: j=1,k=1
    W1[0, 6] = 1  # Z00: k=0,i=0
    
    # Rank-1 decomposition for 1x2x3 tensor
    # T[i,j,k] = 1 for i=0, j in {0,1}, k in {0,1,2}
    # U2 has shape (1*2, 1) = (2, 1)
    # V2 has shape (2*3, 1) = (6, 1)
    # W2 has shape (3*1, 1) = (3, 1)
    
    rank2 = 1
    U2 = np.ones((2, rank2))
    V2 = np.ones((6, rank2))
    W2 = np.ones((3, rank2))
    
    # Kronecker product: total rank = rank1 * rank2 = 7
    total_rank = rank1 * rank2
    
    # Combined dimensions:
    # n_total = n1 * n2 = 2 * 1 = 2
    # m_total = m1 * m2 = 2 * 2 = 4
    # p_total = p1 * p2 = 2 * 3 = 6
    
    # U_combined shape: (n_total * m_total, total_rank) = (8, 7)
    # Index: (i1*n2+i2) * m_total + (j1*m2+j2) = i1*4 + j1*2 + j2
    # Since n2=1, i2=0 always, so i = i1
    # j = j1*2 + j2
    
    # V_combined shape: (m_total * p_total, total_rank) = (24, 7)
    # Index: (j1*m2+j2) * p_total + (k1*p2+k2) = (j1*2+j2)*6 + (k1*3+k2)
    
    # W_combined shape: (p_total * n_total, total_rank) = (12, 7)
    # Index: (k1*p2+k2) * n_total + (i1*n2+i2) = (k1*3+k2)*2 + i1
    
    U_combined = np.zeros((n * m, total_rank))
    V_combined = np.zeros((m * p, total_rank))
    W_combined = np.zeros((p * n, total_rank))
    
    for s1 in range(rank1):
        for s2 in range(rank2):
            s = s1 * rank2 + s2
            
            # U: for each (i1, j1, j2)
            for i1 in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        i = i1  # i2 = 0
                        j = j1 * 2 + j2
                        U_combined[i * m + j, s] += U1[i1 * 2 + j1, s1] * U2[j2, s2]
            
            # V: for each (j1, j2, k1, k2)
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            j = j1 * 2 + j2
                            k = k1 * 3 + k2
                            V_combined[j * p + k, s] += V1[j1 * 2 + j2, s1] * V2[j2 * 3 + k2, s2]
            
            # W: for each (k1, k2, i1)
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        k = k1 * 3 + k2
                        i = i1
                        W_combined[k * n + i, s] += W1[k1 * 2 + i1, s1] * W2[k2, s2]
    
    # Convert to lists of lists with float entries
    U_list = [[float(x) for x in row] for row in U_combined]
    V_list = [[float(x) for x in row] for row in V_combined]
    W_list = [[float(x) for x in row] for row in W_combined]
    
    assert len(U_list) == 8 and len(U_list[0]) == 7
    assert len(V_list) == 24 and len(V_list[0]) == 7
    assert len(W_list) == 12 and len(W_list[0]) == 7
    
    return {"U": U_list, "V": V_list, "W": W_list}
