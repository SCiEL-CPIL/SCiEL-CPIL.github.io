import numpy as np

def construct():
    n, m, p = 2, 2, 5
    rank = 19
    
    # U: (4, 19)
    # V: (10, 19)
    # W: (10, 19)
    U = np.zeros((4, rank))
    V = np.zeros((10, rank))
    W = np.zeros((10, rank))
    
    # Helper to add a term
    # u_idx: list of (index, coeff) for U
    # v_idx: list of (index, coeff) for V
    # w_idx: list of (index, coeff) for W
    def add_term(col, u_data, v_data, w_data):
        for idx, coeff in u_data:
            U[idx, col] += coeff
        for idx, coeff in v_data:
            V[idx, col] += coeff
        for idx, coeff in w_data:
            W[idx, col] += coeff
            
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:02, 3:03, 4:04, 5:10, 6:11, 7:12, 8:13, 9:14
    # C indices: 0:00, 1:01, 2:10, 3:11, 4:20, 5:21, 6:30, 7:31, 8:40, 9:41
    
    # Strassen for cols 0, 1
    # M1 = (A00+A11)(B00+B11)
    add_term(0, [(0,1), (3,1)], [(0,1), (6,1)], [(0,1), (3,1)])
    
    # M2 = (A10+A11)B00
    add_term(1, [(2,1), (3,1)], [(0,1)], [(2,1)])
    
    # M3 = A00(B01-B11)
    add_term(2, [(0,1)], [(1,1), (6,-1)], [(1,1)])
    
    # M4 = A11(B10-B00)
    add_term(3, [(3,1)], [(5,1), (0,-1)], [(0,1), (2,1)])
    
    # M5 = (A00+A01)B11
    add_term(4, [(0,1), (1,1)], [(6,1)], [(0,-1), (1,1)])
    
    # M6 = (A10-A00)(B00+B01)
    add_term(5, [(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)])
    
    # M7 = (A01-A11)(B10+B11)
    add_term(6, [(1,1), (3,-1)], [(5,1), (6,1)], [(0,1), (3,-1)])
    
    # Schoolbook for col 2 (indices 2, 7)
    # C20 = A00 B02 + A01 B12
    # C21 = A10 B02 + A11 B12
    # Term 7: A00 B02 C20
    add_term(7, [(0,1)], [(2,1)], [(4,1)])
    # Term 8: A01 B12 C20
    add_term(8, [(1,1)], [(7,1)], [(4,1)])
    # Term 9: A10 B02 C21
    add_term(9, [(2,1)], [(2,1)], [(5,1)])
    # Term 10: A11 B12 C21
    add_term(10, [(3,1)], [(7,1)], [(5,1)])
    
    # Schoolbook for col 3 (indices 3, 8)
    # Term 11: A00 B03 C30
    add_term(11, [(0,1)], [(3,1)], [(6,1)])
    # Term 12: A01 B13 C30
    add_term(12, [(1,1)], [(8,1)], [(6,1)])
    # Term 13: A10 B03 C31
    add_term(13, [(2,1)], [(3,1)], [(7,1)])
    # Term 14: A11 B13 C31
    add_term(14, [(3,1)], [(8,1)], [(7,1)])
    
    # Schoolbook for col 4 (indices 4, 9)
    # Term 15: A00 B04 C40
    add_term(15, [(0,1)], [(4,1)], [(8,1)])
    # Term 16: A01 B14 C40
    add_term(16, [(1,1)], [(9,1)], [(8,1)])
    # Term 17: A10 B04 C41
    add_term(17, [(2,1)], [(4,1)], [(9,1)])
    # Term 18: A11 B14 C41
    add_term(18, [(3,1)], [(9,1)], [(9,1)])
    
    # Convert to lists of lists as per interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
