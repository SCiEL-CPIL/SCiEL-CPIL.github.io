import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    U = []
    V = []
    W = []
    
    def add_strassen_2x2x2(i0, j0, k0):
        """Add Strassen decomposition for 2x2x2 block starting at (i0, j0, k0)"""
        terms = [
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,-1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,-1)]),
            ([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,1,1)]),
            ([(1,1,1)], [(0,1,1),(1,0,-1)], [(0,0,1),(1,0,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)], [(1,1,1)]),
            ([(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)], [(0,1,-1)]),
        ]
        
        for a_coeffs, b_coeffs, c_coeffs in terms:
            U_row = [0] * (n * m)
            V_row = [0] * (m * p)
            W_row = [0] * (p * n)
            
            for (di, dj, coeff) in a_coeffs:
                U_row[(i0+di)*m + (j0+dj)] += coeff
            
            for (dj, dk, coeff) in b_coeffs:
                V_row[(j0+dj)*p + (k0+dk)] += coeff
            
            for (dk, di, coeff) in c_coeffs:
                W_row[(k0+dk)*n + (i0+di)] += coeff
            
            U.append(U_row)
            V.append(V_row)
            W.append(W_row)
    
    def add_schoolbook_2x1x2(i0, j0, k0):
        """Add schoolbook for 2x1x2 block"""
        for di in range(2):
            for dk in range(2):
                U_row = [0] * (n * m)
                V_row = [0] * (m * p)
                W_row = [0] * (p * n)
                
                U_row[(i0+di)*m + j0] = 1
                V_row[j0*p + (k0+dk)] = 1
                W_row[(k0+dk)*n + (i0+di)] = 1
                
                U.append(U_row)
                V.append(V_row)
                W.append(W_row)
    
    # Partition: i: 2+2, j: 2+2+1, k: 2+2+2
    j_offsets = [0, 2, 4]
    j_sizes = [2, 2, 1]
    k_offsets = [0, 2, 4]
    k_sizes = [2, 2, 2]
    
    for ib in range(2):
        i0 = ib * 2
        for jb in range(3):
            j0 = j_offsets[jb]
            for kb in range(3):
                k0 = k_offsets[kb]
                
                if j_sizes[jb] == 2 and k_sizes[kb] == 2:
                    add_strassen_2x2x2(i0, j0, k0)
                elif j_sizes[jb] == 1 and k_sizes[kb] == 2:
                    add_schoolbook_2x1x2(i0, j0, k0)
    
    return {"U": U, "V": V, "W": W}
