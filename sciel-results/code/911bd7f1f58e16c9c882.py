import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen 2x2 factors
    # Indices: 00, 01, 10, 11
    U_S = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, -1, -1]
    ]
    # Check M1: (A00+A11)(B00+B11). U=[1,0,0,1], V=[1,0,0,1].
    # My U_S above: Col 0: [1,0,0,1]. Correct.
    # Col 1: M2=(A10+A11)B00. U=[0,0,1,1]. V=[1,0,0,0].
    # My U_S Col 1: [0,0,1,1]. Correct.
    # Col 2: M3=A00(B10-B11). U=[1,0,0,0]. V=[0,0,1,-1].
    # My U_S Col 2: [1,0,0,0]. Correct.
    # Col 3: M4=A11(B01-B00). U=[0,0,0,1]. V=[0,1,-1,0].
    # My U_S Col 3: [0,0,0,1]. Correct.
    # Col 4: M5=(A00+A01)B11. U=[1,1,0,0]. V=[0,0,0,1].
    # My U_S Col 4: [1,1,0,0]. Correct.
    # Col 5: M6=(A10-A00)(B00+B10). U=[0,0,1,-1]. V=[1,0,1,0].
    # My U_S Col 5: [0,0,1,-1]. Correct.
    # Col 6: M7=(A01-A11)(B01+B11). U=[0,1,0,-1]. V=[0,1,0,1].
    # My U_S Col 6: [0,1,0,-1]. Correct.
    
    V_S = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, -1, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    # Check V_S against definitions:
    # M1: V=[1,0,0,1]. Col 0: [1,0,0,1]. Correct.
    # M2: V=[1,0,0,0]. Col 1: [1,0,0,0]. Correct.
    # M3: V=[0,0,1,-1]. Col 2: [0,0,1,-1]. Correct.
    # M4: V=[0,1,-1,0]. Col 3: [0,1,-1,0]. Correct.
    # M5: V=[0,0,0,1]. Col 4: [0,0,0,1]. Correct.
    # M6: V=[1,0,1,0]. Col 5: [1,0,1,0]. Correct.
    # M7: V=[0,1,0,1]. Col 6: [0,1,0,1]. Correct.

    W_S = [
        [1, 0, 0, -1, 1, 0, -1],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, -1, 1, -1]
    ]
    # Check W_S:
    # M1: W=[1,0,0,1]. Col 0: [1,0,0,1]. Correct.
    # M2: W=[0,0,1,0]. Col 1: [0,0,1,0]. Correct.
    # M3: W=[0,1,0,0]. Col 2: [0,1,0,0]. Correct.
    # M4: W=[-1,0,1,0]. Col 3: [-1,0,1,0]. Correct.
    # M5: W=[1,0,0,-1]. Col 4: [1,0,0,-1]. Correct.
    # M6: W=[0,0,0,1]. Col 5: [0,0,0,1]. Correct.
    # M7: W=[-1,1,0,-1]. Col 6: [-1,1,0,-1]. Correct.

    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for k in range(7):
        for l in range(7):
            c = k * 7 + l
            for I in range(2):
                for i in range(2):
                    for J in range(2):
                        for j in range(2):
                            r = (2 * I + i) * 4 + (2 * J + j)
                            U[r][c] = U_S[I][k] * U_S[i][l]
                            V[r][c] = V_S[I][k] * V_S[i][l]
                            W[r][c] = W_S[I][k] * W_S[i][l]
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
