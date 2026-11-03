import numpy as np
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_idx, val_u, v_idx, val_v, w_idx, val_w):
        """
        Adds a term to the decomposition.
        u_idx, v_idx, w_idx are global indices.
        val_u, val_v, val_w are coefficients.
        """
        # We will store terms as tuples (u_idx, val_u, v_idx, val_v, w_idx, val_w)
        # and then convert to matrices later, or build matrices directly.
        # Building directly is easier.
        pass

    # We'll use a list of dictionaries or just append to lists of length Rank.
    # Since Rank is determined by the number of terms, we can first collect terms.
    terms = []
    
    def add_term_simple(u_idx, v_idx, w_idx, coeff=1):
        """
        Adds a term with coefficient 1 for the given indices.
        If coeff is not 1, we split it? No, entries can be integers/half-integers.
        So we can have coeff.
        """
        terms.append((u_idx, coeff, v_idx, coeff, w_idx, coeff)) 
        # Wait, the term is u_r \otimes v_r \otimes w_r.
        # If we add a term, we need to specify the vector entries for that rank slot.
        # It's better to accumulate vectors.
        
    # Let's use a different structure:
    # rank = 0
    # U = []
    # V = []
    # W = []
    # def add_term(u_map, v_map, w_map):
    #    # u_map: {global_idx: coeff}
    #    # v_map: {global_idx: coeff}
    #    # w_map: {global_idx: coeff}
    #    # This creates one rank slot.
    #    pass

    # This is too slow if we do it one by one.
    # Let's define the sub-decompositions and merge them.

    # Helper to create a Strassen 2x2x2 decomposition
    # Returns lists of (u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs)
    # Actually, it's easier to just return the U, V, W vectors for the local 2x2x2 problem.
    # Then map to global.

    def get_strassen_222():
        # Returns U (4x7), V (4x7), W (4x7) for 2x2x2 multiplication
        # Local indices:
        # U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Strassen coefficients:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M2 - M5
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
        # W contributions:
        # C00 += M1 -> W[0,0] += 1
        # C11 += M1 -> W[3,0] += 1
        # C01, C10 do not use M1 directly?
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M2 - M5
        
        # Let's re-evaluate W matrix construction.
        # W[k*n+i, r] corresponds to C[i,k] in local 2x2?
        # Local C is 2x2. i in 0,1. k in 0,1.
        # W index = k*2 + i.
        # W[0] -> C[0,0]
        # W[1] -> C[0,1]
        # W[2] -> C[1,0]
        # W[3] -> C[1,1]
        
        # M1 contributes to C00 (+1) and C11 (+1)
        W[0, 0] = 1
        W[3, 0] = 1
        
        # M2: (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # C10 += M2
        W[2, 1] = 1
        
        # M3: A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        # C01 += M3
        W[1, 2] = 1
        
        # M4: A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        # C00 -= M4
        W[0, 3] = -1
        # C10 += M4
        W[2, 3] = 1
        
        # M5: (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # C01 += M5
        W[1, 4] = 1
        # C11 -= M5
        W[3, 4] = -1
        
        # M6: (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # C00 += M6
        W[0, 5] = 1
        
        # M7: (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        # C00 -= M7
        W[0, 6] = -1
        
        return U, V, W

    def get_schoolbook_221():
        # 2x2x1 multiplication
        # A is 2x2, B is 2x1, C is 2x1
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # Rank 4
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        # Local U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Local V indices: 0:(0,0), 1:(1,0)
        # Local W indices: 0:(0,0), 1:(1,0)
        
        # Term 0: A00 B0 -> C0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 1: A01 B1 -> C0
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        # Term 2: A10 B0 -> C1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        # Term 3: A11 B1 -> C1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    # Function to merge a local decomposition into the global one
    def merge_decomposition(local_U, local_V, local_W, 
                            u_map, v_map, w_map):
        """
        u_map: dict local_idx -> global_idx
        v_map: dict local_idx -> global_idx
        w_map: dict local_idx -> global_idx
        """
        r = local_U.shape[1]
        for col in range(r):
            for i in range(local_U.shape[0]):
                if local_U[i, col] != 0:
                    gi = u_map[i]
                    U_list[gi].append(local_U[i, col])
            for i in range(local_V.shape[0]):
                if local_V[i, col] != 0:
                    gi = v_map[i]
                    V_list[gi].append(local_V[i, col])
            for i in range(local_W.shape[0]):
                if local_W[i, col] != 0:
                    gi = w_map[i]
                    W_list[gi].append(local_W[i, col])
                    
            # Pad other entries with 0
            for i in range(local_U.shape[0]):
                if local_U[i, col] == 0:
                    gi = u_map[i]
                    U_list[gi].append(0)
            for i in range(local_V.shape[0]):
                if local_V[i, col] == 0:
                    gi = v_map[i]
                    V_list[gi].append(0)
            for i in range(local_W.shape[0]):
                if local_W[i, col] == 0:
                    gi = w_map[i]
                    W_list[gi].append(0)

    # Initialize global lists
    U_list = [[] for _ in range(n*m)]
    V_list = [[] for _ in range(m*p)]
    W_list = [[] for _ in range(p*n)]
    
    # --- Sub-problem 1: A[0:2, 0:2] * B[0:2, :] ---
    # Local 2x2x5
    # Split B into B[0:2, 0:2] and B[0:2, 2:5]
    
    # 1a: A[0:2, 0:2] * B[0:2, 0:2] (2x2x2)
    sU, sV, sW = get_strassen_222()
    # Global U indices for A[0:2, 0:2]:
    # i=0,j=0 -> 0; i=0,j=1 -> 1; i=1,j=0 -> 3; i=1,j=1 -> 4
    u_map_1a = {0: 0, 1: 1, 2: 3, 3: 4}
    # Global V indices for B[0:2, 0:2]:
    # j=0,k=0 -> 0; j=0,k=1 -> 1; j=1,k=0 -> 5; j=1,k=1 -> 6
    v_map_1a = {0: 0, 1: 1, 2: 5, 3: 6}
    # Global W indices for C[0:2, 0:2]:
    # i=0,k=0 -> 0; i=0,k=1 -> 3; i=1,k=0 -> 1; i=1,k=1 -> 4
    # W index = k*n + i = k*3 + i
    w_map_1a = {0: 0, 1: 3, 2: 1, 3: 4}
    
    merge_decomposition(sU, sV, sW, u_map_1a, v_map_1a, w_map_1a)

    # 1b: A[0:2, 0:2] * B[0:2, 2:5] (2x2x3)
    # Split B[0:2, 2:5] into B[0:2, 2:4] (2x2x2) and B[0:2, 4:5] (2x2x1)
    
    # 1b1: A[0:2, 0:2] * B[0:2, 2:4] (2x2x2)
    sU, sV, sW = get_strassen_222()
    u_map_1b1 = {0: 0, 1: 1, 2: 3, 3: 4}
    # B cols 2,3 -> k=2,3
    # j=0,k=2 -> 2; j=0,k=3 -> 3; j=1,k=2 -> 7; j=1,k=3 -> 8
    v_map_1b1 = {0: 2, 1: 3, 2: 7, 3: 8}
    # C cols 2,3 -> k=2,3
    # i=0,k=2 -> 6; i=0,k=3 -> 9; i=1,k=2 -> 7; i=1,k=3 -> 10
    w_map_1b1 = {0: 6, 1: 9, 2: 7, 3: 10}
    merge_decomposition(sU, sV, sW, u_map_1b1, v_map_1b1, w_map_1b1)

    # 1b2: A[0:2, 0:2] * B[0:2, 4:5] (2x2x1)
    sU, sV, sW = get_schoolbook_221()
    u_map_1b2 = {0: 0, 1: 1, 2: 3, 3: 4}
    # B col 4 -> k=4
    # j=0,k=4 -> 4; j=1,k=4 -> 9
    v_map_1b2 = {0: 4, 1: 9}
    # C col 4 -> k=4
    # i=0,k=4 -> 12; i=1,k=4 -> 13
    w_map_1b2 = {0: 12, 1: 13}
    merge_decomposition(sU, sV, sW, u_map_1b2, v_map_1b2, w_map_1b2)

    # --- Sub-problem 2: A[0:2, 2:3] * B[2:3, :] ---
    # 2x1x5
    # C[i,k] = A[i,2] B[2,k]
    # 10 terms
    for i in range(2):
        for k in range(5):
            u_idx = i * m + 2 # i*3+2
            v_idx = 2 * p + k # 2*5+k
            w_idx = k * n + i # k*3+i
            U_list[u_idx].append(1)
            V_list[v_idx].append(1)
            W_list[w_idx].append(1)
            # Pad others
            for idx in range(n*m):
                if idx != u_idx: U_list[idx].append(0)
            for idx in range(m*p):
                if idx != v_idx: V_list[idx].append(0)
            for idx in range(p*n):
                if idx != w_idx: W_list[idx].append(0)

    # --- Sub-problem 3: A[2:3, 0:2] * B[0:2, :] ---
    # 1x2x5
    # C[2,k] = A[2,0] B[0,k] + A[2,1] B[1,k]
    # 10 terms
    for k in range(5):
        # Term A[2,0] B[0,k]
        u_idx = 2 * m + 0 # 6
        v_idx = 0 * p + k # k
        w_idx = k * n + 2 # k*3+2
        U_list[u_idx].append(1)
        V_list[v_idx].append(1)
        W_list[w_idx].append(1)
        for idx in range(n*m):
            if idx != u_idx: U_list[idx].append(0)
        for idx in range(m*p):
            if idx != v_idx: V_list[idx].append(0)
        for idx in range(p*n):
            if idx != w_idx: W_list[idx].append(0)
            
        # Term A[2,1] B[1,k]
        u_idx = 2 * m + 1 # 7
        v_idx = 1 * p + k # 5+k
        w_idx = k * n + 2 # k*3+2
        U_list[u_idx].append(1)
        V_list[v_idx].append(1)
        W_list[w_idx].append(1)
        for idx in range(n*m):
            if idx != u_idx: U_list[idx].append(0)
        for idx in range(m*p):
            if idx != v_idx: V_list[idx].append(0)
        for idx in range(p*n):
            if idx != w_idx: W_list[idx].append(0)

    # --- Sub-problem 4: A[2:3, 2:3] * B[2:3, :] ---
    # 1x1x5
    # C[2,k] = A[2,2] B[2,k]
    # 5 terms
    for k in range(5):
        u_idx = 2 * m + 2 # 8
        v_idx = 2 * p + k # 10+k
        w_idx = k * n + 2 # k*3+2
        U_list[u_idx].append(1)
        V_list[v_idx].append(1)
        W_list[w_idx].append(1)
        for idx in range(n*m):
            if idx != u_idx: U_list[idx].append(0)
        for idx in range(m*p):
            if idx != v_idx: V_list[idx].append(0)
        for idx in range(p*n):
            if idx != w_idx: W_list[idx].append(0)

    # Convert lists to matrices
    rank = len(U_list[0])
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
