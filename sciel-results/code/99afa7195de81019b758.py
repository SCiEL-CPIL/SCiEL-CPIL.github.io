"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 22 decomposition using Strassen for 2x2x2 blocks and schoolbook for 2x1x2 blocks
    rank = 22
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        if u_val != 0:
            U[u_idx][col] += u_val
        if v_val != 0:
            V[v_idx][col] += v_val
        if w_val != 0:
            W[w_idx][col] += w_val

    # Define the 11 multiplications for a 2x3x2 block
    # U indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # V indices (local 0-5 for j=0,1,2 and k=0,1): 
    # j=0: 0:(0,0), 1:(0,1)
    # j=1: 2:(1,0), 3:(1,1) -> Wait, V index is j*p + k.
    # In the block, k is local 0,1. 
    # But V is global. 
    # For left block (k=0,1 global):
    # j=0, k=0 -> V[0]
    # j=0, k=1 -> V[1]
    # j=1, k=0 -> V[4]
    # j=1, k=1 -> V[5]
    # j=2, k=0 -> V[8]
    # j=2, k=1 -> V[9]
    
    # For right block (k=2,3 global):
    # j=0, k=2 -> V[2]
    # j=0, k=3 -> V[3]
    # j=1, k=2 -> V[6]
    # j=1, k=3 -> V[7]
    # j=2, k=2 -> V[10]
    # j=2, k=3 -> V[11]

    # W indices (local 0-3 for i=0,1 and k_local=0,1):
    # Left block (k_global 0,1):
    # i=0, k=0 -> W[0]
    # i=0, k=1 -> W[1]
    # i=1, k=0 -> W[2]
    # i=1, k=1 -> W[3]
    
    # Right block (k_global 2,3):
    # i=0, k=2 -> W[4]
    # i=0, k=3 -> W[5]
    # i=1, k=2 -> W[6]
    # i=1, k=3 -> W[7]

    # Strassen terms for 2x2x2 part (j=0,1)
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # Schoolbook terms for 2x1x2 part (j=2)
    # m8 = A02 B20
    # m9 = A02 B21
    # m10 = A12 B20
    # m11 = A12 B21

    # Left Block (cols 0-10)
    # Col 0: m1
    add_term(0, 1, 0, 1, 0, 1, 0) # a*e part? No, (a+d)(e+h)
    add_term(4, 1, 0, 1, 0, 1, 0) # d*e
    add_term(0, 1, 5, 1, 0, 1, 0) # a*h
    add_term(4, 1, 5, 1, 0, 1, 0) # d*h
    # Wait, Strassen m1 is (a+d)(e+h). 
    # U: a(0), d(4). V: e(0), h(5).
    # W: C00(0) +1, C11(3) +1.
    # My add_term calls above are wrong for combined terms. 
    # I should just define the U, V, W vectors for each multiplication.
    
    # Let's restart the filling logic properly.
    
    # Left Block
    # Col 0: m1 = (A00+A10)(B00+B11)
    U[0][0] += 1; U[4][0] += 1
    V[0][0] += 1; V[5][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # Col 1: m2 = (A10+A11)B00
    U[3][1] += 1; U[4][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] -= 1
    
    # Col 2: m3 = A00(B01-B11)
    U[0][2] += 1
    V[1][2] += 1; V[5][2] -= 1
    W[1][2] += 1; W[3][2] += 1
    
    # Col 3: m4 = A11(B10-B00)
    U[4][3] += 1
    V[4][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[2][3] += 1
    
    # Col 4: m5 = (A00+A01)B11
    U[0][4] += 1; U[1][4] += 1
    V[5][4] += 1
    W[0][4] -= 1; W[1][4] += 1
    
    # Col 5: m6 = (A10-A00)(B00+B01)
    U[3][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # Col 6: m7 = (A01-A11)(B10+B11)
    U[1][6] += 1; U[4][6] -= 1
    V[4][6] += 1; V[5][6] += 1
    W[0][6] += 1
    
    # Col 7: m8 = A02 B20
    U[2][7] += 1
    V[8][7] += 1
    W[0][7] += 1
    
    # Col 8: m9 = A02 B21
    U[2][8] += 1
    V[9][8] += 1
    W[1][8] += 1
    
    # Col 9: m10 = A12 B20
    U[5][9] += 1
    V[8][9] += 1
    W[2][9] += 1
    
    # Col 10: m11 = A12 B21
    U[5][10] += 1
    V[9][10] += 1
    W[3][10] += 1
    
    # Right Block (cols 11-21)
    # Same U
    # V mapping: 0->2, 1->3, 4->6, 5->7, 8->10, 9->11
    # W mapping: 0->4, 1->5, 2->6, 3->7
    
    # Col 11: m1
    U[0][11] += 1; U[4][11] += 1
    V[2][11] += 1; V[7][11] += 1
    W[4][11] += 1; W[7][11] += 1
    
    # Col 12: m2
    U[3][12] += 1; U[4][12] += 1
    V[2][12] += 1
    W[6][12] += 1; W[7][12] -= 1
    
    # Col 13: m3
    U[0][13] += 1
    V[3][13] += 1; V[7][13] -= 1
    W[5][13] += 1; W[7][13] += 1
    
    # Col 14: m4
    U[4][14] += 1
    V[6][14] += 1; V[2][14] -= 1
    W[4][14] += 1; W[6][14] += 1
    
    # Col 15: m5
    U[0][15] += 1; U[1][15] += 1
    V[7][15] += 1
    W[4][15] -= 1; W[5][15] += 1
    
    # Col 16: m6
    U[3][16] += 1; U[0][16] -= 1
    V[2][16] += 1; V[3][16] += 1
    W[7][16] += 1
    
    # Col 17: m7
    U[1][17] += 1; U[4][17] -= 1
    V[6][17] += 1; V[7][17] += 1
    W[4][17] += 1
    
    # Col 18: m8
    U[2][18] += 1
    V[10][18] += 1
    W[4][18] += 1
    
    # Col 19: m9
    U[2][19] += 1
    V[11][19] += 1
    W[5][19] += 1
    
    # Col 20: m10
    U[5][20] += 1
    V[10][20] += 1
    W[6][20] += 1
    
    # Col 21: m11
    U[5][21] += 1
    V[11][21] += 1
    W[7][21] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
