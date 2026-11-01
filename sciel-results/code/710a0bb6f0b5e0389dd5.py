import numpy as np
import time

DIMENSIONS = [2, 2, 3]

def _build_rank_11_decomposition():
    """
    Constructs the rank-11 decomposition for 2x2x3 matrix multiplication.
    This combines Strassen's algorithm for the 2x2x2 block (rank 7)
    and the schoolbook method for the remaining 2x2x1 block (rank 4).
    """
    n, m, p = 2, 2, 3
    rank = 11
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Part 1: Strassen for C[:, 0:2] (2x2x2 multiplication) ---
    # Indices in U correspond to A elements: a(0), b(1), c(2), d(3)
    # Indices in V correspond to B elements: b00(0), b01(1), b10(2), b11(3)
    # Indices in W correspond to C elements: c00(0), c10(1), c01(2), c11(3)
    
    # M1 = (a+d)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (c+d)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = a(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M4 = d(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M5 = (a+b)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (c-a)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (b-d)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # Combine Strassen products into C[:, 0:2]
    # C00 = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    # C10 = M2 + M4
    W[1, 1] = 1; W[1, 3] = 1
    # C01 = M3 + M5
    W[2, 2] = 1; W[2, 4] = 1
    # C11 = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1

    # --- Part 2: Schoolbook for C[:, 2] (2x2x1 multiplication) ---
    # Indices in V for col 2: b02(4), b12(5)
    # Indices in W for col 2: c02(4), c12(5)
    
    # M8 = a * b02
    U[0, 7] = 1
    V[4, 7] = 1
    
    # M9 = b * b12
    U[1, 8] = 1
    V[5, 8] = 1
    
    # M10 = c * b02
    U[2, 9] = 1
    V[4, 9] = 1
    
    # M11 = d * b12
    U[3, 10] = 1
    V[5, 10] = 1
    
    # Combine Schoolbook products into C[:, 2]
    # C02 = M8 + M9
    W[4, 7] = 1; W[4, 8] = 1
    # C12 = M10 + M11
    W[5, 9] = 1; W[5, 10] = 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The minimal rank for 2x2x3 matrix multiplication is 11.
    # This is a proven lower bound in algebraic complexity theory for this specific dimension.
    # Any decomposition with rank < 11 is impossible.
    
    # Construct the optimal rank-11 decomposition
    result = _build_rank_11_decomposition()
    
    # Convert numpy arrays to lists as required by the interface specification
    # (though numpy arrays are often accepted, lists are safer for "integer/half-integer entries" checks)
    U_list = result["U"].tolist()
    V_list = result["V"].tolist()
    W_list = result["W"].tolist()
    
    # Ensure entries are clean integers (remove floating point noise if any, though we used exact ints)
    for i in range(len(U_list)):
        for j in range(len(U_list[i])):
            U_list[i][j] = int(round(U_list[i][j]))
    for i in range(len(V_list)):
        for j in range(len(V_list[i])):
            V_list[i][j] = int(round(V_list[i][j]))
    for i in range(len(W_list)):
        for j in range(len(W_list[i])):
            W_list[i][j] = int(round(W_list[i][j]))
            
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
