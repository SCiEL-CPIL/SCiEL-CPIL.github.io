"""Matrix multiplication 2,4,8 - Strassen-based rank-56 decomposition."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x4 matrix by a 4x8 matrix.
    
    Strategy: Use Strassen's algorithm structure.
    - Split the 2x4 * 4x8 multiplication into 8 independent 2x2 * 2x2 blocks.
    - Each 2x2 * 2x2 can be done with rank 7 (Strassen).
    - Total rank = 8 * 7 = 56, vs schoolbook 64.
    
    A is 2x4, B is 4x8.
    Split A columns into two 2x2 blocks: A = [A0 | A1]
    Split B rows into two 2x8 blocks: B = [B0; B1]
    C = A0*B0 + A1*B1
    
    Each 2x2 * 2x8 is 4 independent 2x2 * 2x2 multiplications.
    Total: 8 sub-multiplications, each rank 7 via Strassen.
    Total rank = 56.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # 2x2 Strassen terms for 2x2 * 2x2:
    # U_local indexed by (i_l*2+j_l): [x00, x01, x10, x11]
    # V_local indexed by (j_l*2+k_l): [y00, y01, y10, y11]
    # W_local indexed by (k_l*2+i_l): [z(k=0,i=0), z(k=0,i=1), z(k=1,i=0), z(k=1,i=1)]
    # i.e., W[0]=z00, W[1]=z10, W[2]=z01, W[3]=z11
    strassen_terms = [
        # M1 = (x00+x11)*(y00+y11) -> z00+=1, z11+=1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (x10+x11)*y00 -> z10+=1, z11-=1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        # M3 = x00*(y01-y11) -> z01+=1, z11+=1
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        # M4 = x11*(y10-y00) -> z00+=1, z10+=1
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        # M5 = (x00+x01)*y11 -> z00-=1, z01+=1
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        # M6 = (x10-x00)*(y00+y01) -> z11+=1
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (x01-x11)*(y10+y11) -> z00+=1
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    col = 0
    for ba in range(2):       # which A/B row block (0 or 1)
        for bb in range(4):   # which column block (0,1,2,3)
            for t in range(7):
                U_loc, V_loc, W_loc = strassen_terms[t]
                for i_l in range(2):
                    for j_l in range(2):
                        u_idx = i_l * m + (2 * ba + j_l)
                        U[u_idx][col] = U_loc[i_l * 2 + j_l]
                for j_l in range(2):
                    for k_l in range(2):
                        v_idx = (2 * ba + j_l) * p + (2 * bb + k_l)
                        V[v_idx][col] = V_loc[j_l * 2 + k_l]
                for k_l in range(2):
                    for i_l in range(2):
                        w_idx = (2 * bb + k_l) * n + i_l
                        W[w_idx][col] = W_loc[k_l * 2 + i_l]
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
