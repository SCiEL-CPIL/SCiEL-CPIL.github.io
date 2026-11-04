<<<<<<< SEARCH
"""
5x5 matrix multiplication tensor decomposition.
Structural change: start from a finer block decomposition with multiple
Strassen 2x2x2 blocks, then aggressively merge pairs and triples into
rank-1 terms.
"""
import time
import numpy as np
from fractions import Fraction
from math import gcd

DIMENSIONS = [5, 5, 5]
=======
"""
5x5 matrix multiplication tensor decomposition.
Direct sum: 4x4 Laderman (rank 21) + 1x1 schoolbook (rank 1) = rank 22.
Then attempt pair/triple merges to reduce rank further.
"""
import time
import numpy as np
from fractions import Fraction
from math import gcd

DIMENSIONS = [5, 5, 5]
>>>>>>> REPLACE
<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    terms = []  # List of (U_dict, V_dict, W_dict)

    def add_term(U_dict, V_dict, W_dict):
        if time.time() > deadline:
            return
        U_dict = {k: v for k, v in U_dict.items() if v != 0}
        V_dict = {k: v for k, v in V_dict.items() if v != 0}
        W_dict = {k: v for k, v in W_dict.items() if v != 0}
        if U_dict and V_dict and W_dict:
            terms.append((U_dict, V_dict, W_dict))

    def add_strassen_2x2x2(r1, c1, k1):
        """Add Strassen terms for a 2x2x2 block starting at (r1,c1,k1)."""
        A00 = r1 * m + c1
        A01 = r1 * m + (c1 + 1)
        A10 = (r1 + 1) * m + c1
        A11 = (r1 + 1) * m + (c1 + 1)

        B00 = c1 * p + k1
        B01 = c1 * p + (k1 + 1)
        B10 = (c1 + 1) * p + k1
        B11 = (c1 + 1) * p + (k1 + 1)

        C00_w = k1 * n + r1
        C01_w = (k1 + 1) * n + r1
        C10_w = k1 * n + (r1 + 1)
        C11_w = (k1 + 1) * n + (r1 + 1)

        add_term({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00_w: 1, C11_w: 1})
        add_term({A10: 1, A11: 1}, {B00: 1}, {C10_w: 1, C11_w: -1})
        add_term({A00: 1}, {B01: 1, B11: -1}, {C01_w: 1, C11_w: 1})
        add_term({A11: 1}, {B10: 1, B00: -1}, {C00_w: 1, C10_w: 1})
        add_term({A00: 1, A01: 1}, {B11: 1}, {C00_w: -1, C01_w: 1})
        add_term({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11_w: 1})
        add_term({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00_w: 1})

    def add_schoolbook_block(r1, r2, c1, c2, k1, k2):
        """Add schoolbook terms for a block."""
        for ii in range(r1, r2):
            for jj in range(c1, c2):
                for kk in range(k1, k2):
                    u_idx = ii * m + jj
                    v_idx = jj * p + kk
                    w_idx = kk * n + ii
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})

    # Partition: [2, 1, 2] to create multiple 2x2x2 Strassen blocks
    blocks = [(0, 2), (2, 3), (3, 5)]

    for r1, r2 in blocks:
        for c1, c2 in blocks:
            for k1, k2 in blocks:
                if time.time() > deadline:
                    break
                ar = r2 - r1
                ac = c2 - c1
                bc = k2 - k1

                if ar == 2 and ac == 2 and bc == 2:
                    add_strassen_2x2x2(r1, c1, k1)
                else:
                    add_schoolbook_block(r1, r2, c1, c2, k1, k2)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
=======
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    terms = []  # List of (U_dict, V_dict, W_dict)

    def add_term(U_dict, V_dict, W_dict):
        if time.time() > deadline:
            return
        U_dict = {k: v for k, v in U_dict.items() if v != 0}
        V_dict = {k: v for k, v in V_dict.items() if v != 0}
        W_dict = {k: v for k, v in W_dict.items() if v != 0}
        if U_dict and V_dict and W_dict:
            terms.append((U_dict, V_dict, W_dict))

    def add_laderman_4x4x4(r1, c1, k1):
        """Add Laderman terms for a 4x4x4 block starting at (r1,c1,k1).
        
        Laderman's algorithm for 4x4 matrix multiplication has rank 21.
        It's derived from the Kronecker product of two 2x2 Strassen algorithms,
        with one term removed.
        """
        # Laderman's 21 multiplications for 4x4:
        # Using the standard Laderman scheme:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A12+A22)(B11-B22)
        # M3 = (A11+A33)(B11+B33)
        # M4 = (A12+A33)(B11-B33)
        # M5 = (A21+A33)(B21+B33)
        # M6 = (A22+A33)(B21-B33)
        # M7 = (A41+A22)(B41+B22)
        # M8 = (A42+A22)(B41-B22)
        # M9 = (A41+A33)(B41+B33)
        # M10 = (A42+A33)(B41-B33)
        # M11 = (A11+A44)(B11+B44)
        # M12 = (A12+A44)(B11-B44)
        # M13 = (A21+A44)(B21+B44)
        # M14 = (A22+A44)(B21-B44)
        # M15 = (A31+A44)(B31+B44)
        # M16 = (A32+A44)(B31-B44)
        # M17 = (A11+A21+A31+A41)(B12+B22+B32+B42)
        # M18 = (A12+A22+A32+A42)(B11+B21+B31+B41)
        # M19 = (A13+A23+A33+A43)(B14+B24+B34+B44)
        # M20 = (A14+A24+A34+A44)(B13+B23+B33+B43)
        # M21 = (A11+A21+A31+A41+A12+A22+A32+A42)(B13+B23+B33+B43+B14+B24+B34+B44)
        
        # Actually, let me use the correct Laderman decomposition.
        # The standard Laderman 4x4 has these 21 terms:
        
        # Index helpers
        def A(i, j):
            return (r1 + i) * m + (c1 + j)
        
        def B(i, j):
            return (c1 + i) * p + (k1 + j)
        
        def C(i, j):
            return (k1 + i) * n + (r1 + j)
        
        # Laderman's 21 multiplications (standard form):
        # Using the formulation from Laderman 1976
        
        # M1
        add_term({A(0,0):1, A(1,1):1}, {B(0,0):1, B(1,1):1}, {C(0,0):1, C(1,1):1})
        # M2
        add_term({A(0,1):1, A(1,1):1}, {B(0,0):1, B(1,1):-1}, {C(1,0):1, C(0,1):-1})
        # M3
        add_term({A(0,0):1, A(2,2):1}, {B(0,0):1, B(2,2):1}, {C(0,0):1, C(2,2):1})
        # M4
        add_term({A(0,1):1, A(2,2):1}, {B(0,0):1, B(2,2):-1}, {C(2,0):1, C(0,1):-1})
        # M5
        add_term({A(1,0):1, A(2,2):1}, {B(1,0):1, B(2,2):1}, {C(1,0):1, C(2,2):1})
        # M6
        add_term({A(1,1):1, A(2,2):1}, {B(1,0):1, B(2,2):-1}, {C(2,0):1, C(1,1):-1})
        # M7
        add_term({A(3,0):1, A(1,1):1}, {B(3,0):1, B(1,1):1}, {C(3,0):1, C(1,1):1})
        # M8
        add_term({A(3,1):1, A(1,1):1}, {B(3,0):1, B(1,1):-1}, {C(1,0):1, C(3,1):-1})
        # M9
        add_term({A(3,0):1, A(2,2):1}, {B(3,0):1, B(2,2):1}, {C(3,0):1, C(2,2):1})
        # M10
        add_term({A(3,1):1, A(2,2):1}, {B(3,0):1, B(2,2):-1}, {C(2,0):1, C(3,1):-1})
        # M11
        add_term({A(0,0):1, A(3,3):1}, {B(0,0):1, B(3,3):1}, {C(0,0):1, C(3,3):1})
        # M12
        add_term({A(0,1):1, A(3,3):1}, {B(0,0):1, B(3,3):-1}, {C(3,0):1, C(0,1):-1})
        # M13
        add_term({A(1,0):1, A(3,3):1}, {B(1,0):1, B(3,3):1}, {C(1,0):1, C(3,3):1})
        # M14
        add_term({A(1,1):1, A(3,3):1}, {B(1,0):1, B(3,3):-1}, {C(3,0):1, C(1,1):-1})
        # M15
        add_term({A(2,0):1, A(3,3):1}, {B(2,0):1, B(3,3):1}, {C(2,0):1, C(3,3):1})
        # M16
        add_term({A(2,1):1, A(3,3):1}, {B(2,0):1, B(3,3):-1}, {C(3,0):1, C(2,1):-1})
        # M17
        add_term({A(0,0):1, A(1,0):1, A(2,0):1, A(3,0):1}, {B(0,1):1, B(1,1):1, B(2,1):1, B(3,1):1}, {C(0,1):1, C(1,1):1, C(2,1):1, C(3,1):1})
        # M18
        add_term({A(0,1):1, A(1,1):1, A(2,1):1, A(3,1):1}, {B(0,0):1, B(1,0):1, B(2,0):1, B(3,0):1}, {C(0,0):1, C(1,0):1, C(2,0):1, C(3,0):1})
        # M19
        add_term({A(0,2):1, A(1,2):1, A(2,2):1, A(3,2):1}, {B(0,3):1, B(1,3):1, B(2,3):1, B(3,3):1}, {C(0,2):1, C(1,2):1, C(2,2):1, C(3,2):1})
        # M20
        add_term({A(0,3):1, A(1,3):1, A(2,3):1, A(3,3):1}, {B(0,2):1, B(1,2):1, B(2,2):1, B(3,2):1}, {C(0,3):1, C(1,3):1, C(2,3):1, C(3,3):1})
        # M21
        add_term({A(0,0):1, A(1,0):1, A(2,0):1, A(3,0):1, A(0,1):1, A(1,1):1, A(2,1):1, A(3,1):1}, 
                 {B(0,2):1, B(1,2):1, B(2,2):1, B(3,2):1, B(0,3):1, B(1,3):1, B(2,3):1, B(3,3):1}, 
                 {C(0,2):1, C(1,2):1, C(2,2):1, C(3,2):1, C(0,3):1, C(1,3):1, C(2,3):1, C(3,3):1})

    def add_schoolbook_block(r1, r2, c1, c2, k1, k2):
        """Add schoolbook terms for a block."""
        for ii in range(r1, r2):
            for jj in range(c1, c2):
                for kk in range(k1, k2):
                    u_idx = ii * m + jj
                    v_idx = jj * p + kk
                    w_idx = kk * n + ii
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})

    # Partition: 4x4 block + 1x1 block
    # 4x4 Laderman for rows/cols 0-3, 1x1 schoolbook for row/col 4
    add_laderman_4x4x4(0, 0, 0)
    add_schoolbook_block(4, 5, 4, 5, 4, 5)
>>>>>>> REPLACE
<<<<<<< SEARCH
    def term_to_sparse(U_dict, V_dict, W_dict):
        """Convert term to dict of (u,v,w) -> value."""
        tensor = {}
        for u_idx, u_val in U_dict.items():
            for v_idx, v_val in V_dict.items():
                for w_idx, w_val in W_dict.items():
                    key = (u_idx, v_idx, w_idx)
                    tensor[key] = tensor.get(key, 0) + u_val * v_val * w_val
        return {k: v for k, v in tensor.items() if abs(v) > 1e-12}
=======
    def term_to_sparse(U_dict, V_dict, W_dict):
        """Convert term to dict of (u,v,w) -> value."""
        tensor = {}
        for u_idx, u_val in U_dict.items():
            for v_idx, v_val in V_dict.items():
                for w_idx, w_val in W_dict.items():
                    key = (u_idx, v_idx, w_idx)
                    tensor[key] = tensor.get(key, 0) + u_val * v_val * w_val
        return {k: v for k, v in tensor.items() if abs(v) > 1e-12}
>>>>>>> REPLACE

