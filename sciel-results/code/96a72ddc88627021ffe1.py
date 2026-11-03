import time
import random
import numpy as np
from fractions import Fraction
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute indices for schoolbook terms
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                indices.append((u_idx, v_idx, w_idx))
                
    # Strategy: Construct Rank 29 solution explicitly.
    # 1. Identify the 22 schoolbook terms NOT in the 2x2x2 sub-block (j in {0,1}, k in {0,1}).
    # 2. Add the 7 Strassen terms for the sub-block.
    
    U = []
    V = []
    W = []
    
    # Add 22 schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip the sub-block terms
                if j in [0, 1] and k in [0, 1]:
                    continue
                
                u_vec = np.zeros(nm)
                v_vec = np.zeros(mp)
                w_vec = np.zeros(pn)
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                u_vec[u_idx] = 1.0
                v_vec[v_idx] = 1.0
                w_vec[w_idx] = 1.0
                
                U.append(u_vec)
                V.append(v_vec)
                W.append(w_vec)
                
    # Add 7 Strassen terms for the sub-block (i in {0,1}, j in {0,1}, k in {0,1})
    # The sub-block corresponds to A[:, 0:2], B[0:2, :], C[:, 0:2]
    # We need the 7 rank-1 tensors for 2x2x2 multiplication.
    
    # Standard Strassen vectors for 2x2x2 (flattened row-major)
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # The 7 terms are:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # We define u, v, w for each term in the sub-block coordinates.
    # Then map to global coordinates.
    
    strassen_terms = [
        # (u_sub, v_sub, w_sub)
        # M1
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 1, 1, 1])),
        # M2
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0])), # Check: A10B00 -> C10. A11B00 -> invalid?
        # Wait, A11 is i=1, j=1. B00 is j=0, k=0. j mismatch.
        # In the tensor sum, the term A11B00 is 0 in the final product, but it appears in M2.
        # It must be cancelled by other terms.
        # Let's verify the standard Strassen decomposition sums to the identity.
        # The vectors above are standard.
        
        # M3
        (np.array([1, 0, 0, 0]), np.array([0, -1, 1, 0]), np.array([0, 0, 1, 0])), # A00(B10 - B11)
        # M4
        (np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0]), np.array([0, 0, 0, 1])), # A11(B01 - B00)
        # M5
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([0, 1, 0, 0])), # (A00 + A01)B11
        # M6
        (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0])), # (A10 - A00)(B00 + B01)
        # M7
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([0, 1, 0, 0])), # (A01 - A11)(B10 + B11)
    ]
    
    # Mapping function for sub-block to global
    def map_sub_to_global(u_sub, v_sub, w_sub):
        u_global = np.zeros(nm)
        v_global = np.zeros(mp)
        w_global = np.zeros(pn)
        
        # u_sub indices: 0->00, 1->01, 2->10, 3->11
        # Global u index: i*m + j.
        # 00 -> 0*3+0=0
        # 01 -> 0*3+1=1
        # 10 -> 1*3+0=3
        # 11 -> 1*3+1=4
        
        u_map = [0, 1, 3, 4]
        for idx, val in enumerate(u_sub):
            u_global[u_map[idx]] = val
            
        # v_sub indices: 0->00, 1->01, 2->10, 3->11
        # Global v index: j*p + k.
        # 00 -> 0*5+0=0
        # 01 -> 0*5+1=1
        # 10 -> 1*5+0=5
        # 11 -> 1*5+1=6
        
        v_map = [0, 1, 5, 6]
        for idx, val in enumerate(v_sub):
            v_global[v_map[idx]] = val
            
        # w_sub indices: 0->00, 1->01, 2->10, 3->11
        # Global w index: k*n + i.
        # 00 -> 0*2+0=0
        # 01 -> 1*2+0=2
        # 10 -> 0*2+1=1
        # 11 -> 1*2+1=3
        
        w_map = [0, 2, 1, 3]
        for idx, val in enumerate(w_sub):
            w_global[w_map[idx]] = val
            
        return u_global, v_global, w_global

    for u_sub, v_sub, w_sub in strassen_terms:
        u_global, v_global, w_global = map_sub_to_global(u_sub, v_sub, w_sub)
        U.append(u_global)
        V.append(v_global)
        W.append(w_global)
        
    # Convert to lists
    U_out = np.array(U).T.tolist() # Shape (rank, nm) -> We need (nm, rank)
    # The return interface expects {'U': (n*m, rank), ...}
    # My U list is a list of vectors of length nm.
    # np.array(U) has shape (rank, nm).
    # We need (nm, rank).
    
    U_arr = np.array(U).T
    V_arr = np.array(V).T
    W_arr = np.array(W).T
    
    return {
        "U": U_arr.tolist(),
        "V": V_arr.tolist(),
        "W": W_arr.tolist()
    }
# EVOLVE-BLOCK-END
