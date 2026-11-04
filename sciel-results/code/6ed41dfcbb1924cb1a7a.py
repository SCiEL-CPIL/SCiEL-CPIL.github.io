import time
import numpy as np

def construct():
    # Deadline guard
    deadline = time.time() + 120.0
    
    n, m, p = 2, 3, 5
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx, val in u_indices:
            U[idx, col] += val
        for idx, val in v_indices:
            V[idx, col] += val
        for idx, val in w_indices:
            W[idx, col] += val
        col += 1
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")

    # Indices for A (2x3): i*m + j
    # A00=0, A01=1, A02=2
    # A10=3, A11=4, A12=5
    
    # Indices for B (3x5): j*p + k
    # B00=0, B01=1, B02=2, B03=3, B04=4
    # B10=5, B11=6, B12=7, B13=8, B14=9
    # B20=10, B21=11, B22=12, B23=13, B24=14
    
    # Indices for W (5x2): k*n + i
    # C00 (k=0,i=0)=0, C10 (k=0,i=1)=1
    # C01 (k=1,i=0)=2, C11 (k=1,i=1)=3
    # C02 (k=2,i=0)=4, C12 (k=2,i=1)=5
    # C03 (k=3,i=0)=6, C13 (k=3,i=1)=7
    # C04 (k=4,i=0)=8, C14 (k=4,i=1)=9

    # Part 1: Strassen for j in {0,1}, k in {0,1}
    # A indices: 0,1,3,4
    # B indices: 0,1,5,6
    # W indices: 0,1,2,3
    
    # P1 = (A00+A11)(B00+B11) -> C00 + C11
    add_term([(0,1), (4,1)], [(0,1), (6,1)], [(0,1), (3,1)])
    
    # P2 = (A10+A11)B00 -> C10 - C11
    add_term([(3,1), (4,1)], [(0,1)], [(1,1), (3,-1)])
    
    # P3 = A00(B01-B11) -> C01 + C11
    add_term([(0,1)], [(1,1), (6,-1)], [(2,1), (3,1)])
    
    # P4 = A11(B10-B00) -> C00 + C10
    add_term([(4,1)], [(5,1), (0,-1)], [(0,1), (1,1)])
    
    # P5 = (A00+A01)B11 -> -C00 + C01
    add_term([(0,1), (1,1)], [(6,1)], [(0,-1), (2,1)])
    
    # P6 = (A10-A00)(B00+B01) -> C11
    add_term([(3,1), (0,-1)], [(0,1), (1,1)], [(3,1)])
    
    # P7 = (A01-A11)(B10+B11) -> C00 - C11
    add_term([(1,1), (4,-1)], [(5,1), (6,1)], [(0,1), (3,-1)])

    # Part 2: Strassen for j in {0,1}, k in {2,3}
    # A indices: 0,1,3,4
    # B indices: 2,3,7,8
    # W indices: 4,5,6,7
    
    # P1' = (A00+A11)(B02+B13) -> C02 + C13
    add_term([(0,1), (4,1)], [(2,1), (8,1)], [(4,1), (7,1)])
    
    # P2' = (A10+A11)B02 -> C12 - C13
    add_term([(3,1), (4,1)], [(2,1)], [(5,1), (7,-1)])
    
    # P3' = A00(B03-B13) -> C03 + C13
    add_term([(0,1)], [(3,1), (8,-1)], [(6,1), (7,1)])
    
    # P4' = A11(B12-B02) -> C02 + C12
    add_term([(4,1)], [(7,1), (2,-1)], [(4,1), (5,1)])
    
    # P5' = (A00+A01)B13 -> -C02 + C03
    add_term([(0,1), (1,1)], [(8,1)], [(4,-1), (6,1)])
    
    # P6' = (A10-A00)(B02+B03) -> C13
    add_term([(3,1), (0,-1)], [(2,1), (3,1)], [(7,1)])
    
    # P7' = (A01-A11)(B12+B13) -> C02 - C13
    add_term([(1,1), (4,-1)], [(7,1), (8,1)], [(4,1), (7,-1)])

    # Part 3: Schoolbook for j in {0,1}, k in {4}
    # A indices: 0,1,3,4
    # B indices: 4,9
    # W indices: 8,9
    
    # A00*B04 -> C04
    add_term([(0,1)], [(4,1)], [(8,1)])
    # A01*B14 -> C04
    add_term([(1,1)], [(9,1)], [(8,1)])
    # A10*B04 -> C14
    add_term([(3,1)], [(4,1)], [(9,1)])
    # A11*B14 -> C14
    add_term([(4,1)], [(9,1)], [(9,1)])

    # Part 4: Schoolbook for j in {2}, k in {0..4}
    # A indices: 2,5
    # B indices: 10,11,12,13,14
    # W indices: 0,2,4,6,8 (for i=0) and 1,3,5,7,9 (for i=1)
    
    # i=0
    add_term([(2,1)], [(10,1)], [(0,1)])
    add_term([(2,1)], [(11,1)], [(2,1)])
    add_term([(2,1)], [(12,1)], [(4,1)])
    add_term([(2,1)], [(13,1)], [(6,1)])
    add_term([(2,1)], [(14,1)], [(8,1)])
    
    # i=1
    add_term([(5,1)], [(10,1)], [(1,1)])
    add_term([(5,1)], [(11,1)], [(3,1)])
    add_term([(5,1)], [(12,1)], [(5,1)])
    add_term([(5,1)], [(13,1)], [(7,1)])
    add_term([(5,1)], [(14,1)], [(9,1)])

    # Convert to list of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
