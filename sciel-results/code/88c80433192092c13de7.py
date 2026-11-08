import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 2x3 by 3x7 matrix multiplication.
    
    Strategy (rank 39):
    - Split the 3 columns of A into a 2-column block (j=0,1) and a 1-column block (j=2).
    - For the 2-column block (j=0,1), split the 7 rows of B into:
        - Three 2-row blocks (k=0,1; k=2,3; k=4,5) using Strassen (rank 7 each)
        - One 1-row block (k=6) using schoolbook (rank 4)
    - For the 1-column block (j=2), use schoolbook for all 7 rows (rank 14)
    
    Total rank = 3*7 + 4 + 14 = 21 + 4 + 14 = 39.
    """
    t0 = time.time()
    tl = t0 + 130.0  # safety margin before 150s limit
    
    n, m, p = 2, 3, 7
    R = 39  # target rank
    
    U = np.zeros((n * m, R), dtype=np.int64)  # 6 x 39
    V = np.zeros((m * p, R), dtype=np.int64)  # 21 x 39
    W = np.zeros((p * n, R), dtype=np.int64)  # 14 x 39
    
    col = [0]
    
    def _add(u_idx, v_idx, w_idx, su=1, sv=1, sw=1):
        """Add a single tensor term with given indices and signs."""
        if time.time() > tl:
            return
        c = col[0]
        for i, s in zip(u_idx, su if isinstance(su, (list, tuple)) else [su] * len(u_idx)):
            U[i, c] += s
        for i, s in zip(v_idx, sv if isinstance(sv, (list, tuple)) else [sv] * len(v_idx)):
            V[i, c] += s
        for i, s in zip(w_idx, sw if isinstance(sw, (list, tuple)) else [sw] * len(w_idx)):
            W[i, c] += s
        col[0] += 1
    
    def _strassen(k0, k1):
        """
        Add Strassen decomposition for a 2x2x2 block.
        A rows: 0,1; B rows: k0,k1; C cols: k0,k1
        """
        if time.time() > tl:
            return
        # A element indices
        a00 = 0  # row 0, col 0
        a01 = 1  # row 0, col 1
        a10 = 3  # row 1, col 0
        a11 = 4  # row 1, col 1
        
        # B element indices
        b00 = k0 * 7 + 0  # row k0, col 0
        b01 = k0 * 7 + 1  # row k0, col 1
        b10 = k1 * 7 + 0  # row k1, col 0
        b11 = k1 * 7 + 1  # row k1, col 1
        
        # C element indices (W[k*n+i, r] convention)
        c00 = k0 * 2 + 0  # row k0, col 0
        c01 = k0 * 2 + 1  # row k0, col 1
        c10 = k1 * 2 + 0  # row k1, col 0
        c11 = k1 * 2 + 1  # row k1, col 1
        
        # M1 = (A00 + A11)(B00 + B11)
        _add([a00, a11], [b00, b11], [c00, c11], su=[1, 1], sv=[1, 1], sw=[1, 1])
        
        # M2 = (A10 + A11)(B00)
        _add([a10, a11], [b00], [c10, c11], su=[1, 1], sv=[1], sw=[1, -1])
        
        # M3 = (A00)(B01 - B11)
        _add([a00], [b01, b11], [c01, c11], su=[1], sv=[1, -1], sw=[1, 1])
        
        # M4 = (A11)(B10 - B00)
        _add([a11], [b10, b00], [c00, c10], su=[1], sv=[1, -1], sw=[1, 1])
        
        # M5 = (A00 + A01)(B11)
        _add([a00, a01], [b11], [c00, c01], su=[1, 1], sv=[1], sw=[-1, 1])
        
        # M6 = (A10 - A00)(B00 + B01)
        _add([a10, a00], [b00, b01], [c11], su=[1, -1], sv=[1, 1], sw=[1])
        
        # M7 = (A01 - A11)(B10 + B11)
        _add([a01, a11], [b10, b11], [c00], su=[1, -1], sv=[1, 1], sw=[1])
    
    # Apply Strassen for three 2x2 blocks
    _strassen(0, 1)
    _strassen(2, 3)
    _strassen(4, 5)
    
    # Schoolbook for k=6 (single row)
    if time.time() > tl:
        return
    # B row 6, cols 0,1
    b60 = 6 * 7 + 0
    b61 = 6 * 7 + 1
    # C row 6, cols 0,1
    c60 = 6 * 2 + 0
    c61 = 6 * 2 + 1
    
    # C[6,0] = A[0,0]*B[6,0] + A[0,1]*B[6,1]
    _add([0], [b60], [c60])
    _add([1], [b61], [c60])
    
    # C[6,1] = A[1,0]*B[6,0] + A[1,1]*B[6,1]
    _add([3], [b60], [c61])
    _add([4], [b61], [c61])
    
    # Schoolbook for j=2 (A column 2)
    for k in range(7):
        if time.time() > tl:
            break
        bk = k * 7 + 2
        ck0 = k * 2 + 0
        ck1 = k * 2 + 1
        # C[k,0] += A[0,2]*B[k,2]
        _add([2], [bk], [ck0])
        # C[k,1] += A[1,2]*B[k,2]
        _add([5], [bk], [ck1])
    
    # Convert to list format as expected by the interface
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
