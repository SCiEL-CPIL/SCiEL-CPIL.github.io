import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Constructs a rank-55 tensor decomposition for matrix multiplication 2x3 by 3x10.
    
    Strategy:
    1. Split A (2x3) into A1 (2x2) and A2 (2x1).
    2. Split B (3x10) into B1 (2x10) and B2 (1x10).
    3. C = A1*B1 + A2*B2.
    
    For A1*B1 (2x2 * 2x10):
    - Group the 10 columns of B1 into 5 pairs of 2 columns.
    - For each pair, perform a 2x2 * 2x2 multiplication using Strassen's algorithm (rank 7).
    - Total rank for this part: 5 * 7 = 35.
    
    For A2*B2 (2x1 * 1x10):
    - This is a standard outer product scaled by the 2 elements of A2.
    - Total rank for this part: 2 * 10 = 20.
    
    Total Rank = 35 + 20 = 55.
    """
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 30
    pn = p * n  # 20

    # We will build U, V, W as lists of lists, then convert to numpy if needed, 
    # but the return type expects lists.
    
    # Initialize empty lists for the columns of U, V, W
    U_cols = []
    V_cols = []
    W_cols = []

    # --- Part 1: A1 * B1 (Rank 35) ---
    # A1 is rows 0,1 and cols 0,1 of A.
    # B1 is rows 0,1 and all cols of B.
    # We process 5 pairs of columns from B1.
    
    # Strassen coefficients for 2x2 multiplication:
    # U (A side) coefficients for the 7 products:
    # M1: A00 + A11
    # M2: A10 + A11
    # M3: A00
    # M4: A11
    # M5: A00 + A10
    # M6: A11 - A00
    # M7: A01 - A11
    
    # Note: Indices in A1: (0,0), (0,1), (1,0), (1,1)
    # Mapping to global A indices (i*m + j):
    # A1[0,0] -> idx 0
    # A1[0,1] -> idx 1
    # A1[1,0] -> idx 3
    # A1[1,1] -> idx 4
    
    # V (B side) coefficients for the 7 products:
    # M1: B00 + B11
    # M2: B00
    # M3: B10 - B11
    # M4: B01 - B00
    # M5: B11
    # M6: B00 + B10
    # M7: B11 + B01
    
    # Note: Indices in B1 (local 2x2): (0,0), (0,1), (1,0), (1,1)
    # Mapping to global B indices (j*p + k):
    # For a specific pair of columns k0, k1:
    # B1[0,0] (local) -> B[0, k0] -> idx 0*p + k0 = k0
    # B1[0,1] (local) -> B[0, k1] -> idx 0*p + k1 = k1
    # B1[1,0] (local) -> B[1, k0] -> idx 1*p + k0 = p + k0
    # B1[1,1] (local) -> B[1, k1] -> idx 1*p + k1 = p + k1
    
    # W (C side) coefficients for the 7 products:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Note: Indices in C1 (local 2x2): (0,0), (0,1), (1,0), (1,1)
    # Mapping to global C indices (k*n + i):
    # For a specific pair of columns k0, k1:
    # C1[0,0] (local) -> C[0, k0] -> idx k0*n + 0 = k0*2
    # C1[0,1] (local) -> C[0, k1] -> idx k1*n + 0 = k1*2
    # C1[1,0] (local) -> C[1, k0] -> idx k0*n + 1 = k0*2 + 1
    # C1[1,1] (local) -> C[1, k1] -> idx k1*n + 1 = k1*2 + 1

    # Define the Strassen structure
    # Each entry is (u_coeff, v_coeff, w_coeff) for the 4 components of U, V, W?
    # No, U, V, W are vectors.
    # For each of the 7 Strassen terms, we add a column to U, V, W.
    
    # U vectors (length 6, but only indices 0,1,3,4 are non-zero)
    # V vectors (length 30, but only specific k indices are non-zero)
    # W vectors (length 20, but only specific k indices are non-zero)
    
    # Let's define the 7 Strassen terms' contributions to U, V, W (local 2x2 indices)
    # U_local: [u00, u01, u10, u11]
    # V_local: [v00, v01, v10, v11]
    # W_local: [w00, w01, w10, w11]
    
    strassen_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),
        # M4
        ([0, 0, 0, 1], [1, -1, 0, 0], [1, 0, 1, 0]),
        # M5
        ([1, 0, 1, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([-1, 0, 0, 1], [1, 0, 1, 0], [0, 0, 1, 0]),
        # M7
        ([0, 1, 0, -1], [0, 1, 0, 1], [-1, 0, 0, 1]),
    ]
    
    # Map local indices to global indices
    # U global: i*m + j. A1 uses i in {0,1}, j in {0,1}.
    # Local (0,0) -> Global 0
    # Local (0,1) -> Global 1
    # Local (1,0) -> Global 3
    # Local (1,1) -> Global 4
    u_map = [0, 1, 3, 4]
    
    # V global: j*p + k. B1 uses j in {0,1}, k in {k0, k1}.
    # Local (0,0) -> Global 0*p + k0 = k0
    # Local (0,1) -> Global 0*p + k1 = k1
    # Local (1,0) -> Global 1*p + k0 = p + k0
    # Local (1,1) -> Global 1*p + k1 = p + k1
    
    # W global: k*n + i. C1 uses i in {0,1}, k in {k0, k1}.
    # Local (0,0) -> Global k0*n + 0 = 2*k0
    # Local (0,1) -> Global k1*n + 0 = 2*k1
    # Local (1,0) -> Global k0*n + 1 = 2*k0 + 1
    # Local (1,1) -> Global k1*n + 1 = 2*k1 + 1

    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        v_map = [k0, k1, p + k0, p + k1]
        w_map = [2 * k0, 2 * k1, 2 * k0 + 1, 2 * k1 + 1]
        
        for u_local, v_local, w_local in strassen_terms:
            # Create sparse vectors
            U_vec = [0.0] * nm
            V_vec = [0.0] * mp
            W_vec = [0.0] * pn
            
            for idx, val in enumerate(u_local):
                if val != 0:
                    U_vec[u_map[idx]] = val
            
            for idx, val in enumerate(v_local):
                if val != 0:
                    V_vec[v_map[idx]] = val
            
            for idx, val in enumerate(w_local):
                if val != 0:
                    W_vec[w_map[idx]] = val
            
            U_cols.append(U_vec)
            V_cols.append(V_vec)
            W_cols.append(W_vec)
            
            if time.time() > deadline:
                break

    if time.time() > deadline:
        # Return whatever we have, or fallback? 
        # Better to return a valid smaller decomposition if possible, 
        # but we should be fast enough.
        pass

    # --- Part 2: A2 * B2 (Rank 20) ---
    # A2 is rows 0,1 and col 2 of A.
    # Indices: (0,2) -> 0*3+2=2, (1,2) -> 1*3+2=5.
    # B2 is row 2 and all cols of B.
    # Indices: (2,k) -> 2*10+k = 20+k.
    # C2 is rows 0,1 and all cols of C.
    # Indices: (k,0) -> k*2+0=2k, (k,1) -> k*2+1=2k+1.
    
    # We need 20 terms.
    # For each k in 0..9:
    #   Term 1: A[0,2] * B[2,k] -> C[0,k]
    #   Term 2: A[1,2] * B[2,k] -> C[1,k]
    
    for k in range(p):
        # Term 1
        U_vec = [0.0] * nm
        V_vec = [0.0] * mp
        W_vec = [0.0] * pn
        
        U_vec[2] = 1.0   # A[0,2]
        V_vec[20 + k] = 1.0 # B[2,k]
        W_vec[2 * k] = 1.0  # C[0,k]
        
        U_cols.append(U_vec)
        V_cols.append(V_vec)
        W_cols.append(W_vec)
        
        # Term 2
        U_vec = [0.0] * nm
        V_vec = [0.0] * mp
        W_vec = [0.0] * pn
        
        U_vec[5] = 1.0   # A[1,2]
        V_vec[20 + k] = 1.0 # B[2,k]
        W_vec[2 * k + 1] = 1.0  # C[1,k]
        
        U_cols.append(U_vec)
        V_cols.append(V_vec)
        W_cols.append(W_vec)
        
        if time.time() > deadline:
            break

    # Transpose to get (rows, rank) matrices
    # U_cols is list of rank vectors, each of length nm.
    # We need U[i, r] = U_cols[r][i]
    
    rank = len(U_cols)
    
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    for r in range(rank):
        U[:, r] = U_cols[r]
        V[:, r] = V_cols[r]
        W[:, r] = W_cols[r]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
