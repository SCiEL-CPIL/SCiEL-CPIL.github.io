import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 4]
START_TIME = time.time()
TIME_LIMIT = 120.0

def construct():
    n, m, p = DIMENSIONS
    
    def build_rank_22():
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val

        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        # A indices: 00:0, 01:1, 10:3, 11:4
        # B indices: 00:0, 01:1, 10:4, 11:5
        # C indices (W): 00:0, 01:2, 10:1, 11:3
        
        # M1: (A00+A11)(B00+B11) -> C00+C11
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        # M2: (A10+A11)B00 -> C10-C11
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        # M3: A00(B01-B11) -> C01+C11
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        # M4: A11(B10-B00) -> C00+C10
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        # M5: (A00+A01)B11 -> -C00+C01
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        # M6: (A10-A00)(B00+B01) -> C11
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        # M7: (A01-A11)(B10+B11) -> C00
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        # B indices: 02:2, 03:3, 12:6, 13:7
        # C indices (W): 02:4, 03:6, 12:5, 13:7
        
        # M1': (A00+A11)(B02+B13) -> C02+C13
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        # M2': (A10+A11)B02 -> C12-C13
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        # M3': A00(B03-B13) -> C03+C13
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        # M4': A11(B12-B02) -> C02+C12
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        # M5': (A00+A01)B13 -> -C02+C03
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        # M6': (A10-A00)(B02+B03) -> C13
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        # M7': (A01-A11)(B12+B13) -> C02
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

        # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
        # A indices: 02:2, 12:5
        # B indices: 20:8, 21:9, 22:10, 23:11
        # C indices: 00:0, 01:2, 02:4, 03:6 for i=0
        #            10:1, 11:3, 12:5, 13:7 for i=1
        
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return U, V, W, rank

    U, V, W, rank = build_rank_22()
    return {"U": U, "V": V, "W": W}
