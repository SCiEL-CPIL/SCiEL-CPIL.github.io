"""Matrix multiplication 2x5 by 5x5: find a rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _schoolbook(n, m, p):
    """Return the standard (schoolbook) decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check whether the decomposition exactly reproduces the matmul tensor."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # For each (i,j,k) in the tensor:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = float(np.dot(Ua[a], Va[b] * Wa[c]))
                if abs(val - 1.0) > tol:
                    return False
    # Also verify that entries where the tensor should be 0 are 0.
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in valid:
                    val = float(np.dot(Ua[a], Va[b] * Wa[c]))
                    if abs(val) > tol:
                        return False
    return True

def _strassen_2x2x2():
    """
    Return a rank-7 Strassen decomposition for M_{2,2,2}.
    The 7 products are:
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22)B11
      M3 = A11(B21-B22)
      M4 = A22(B12-B11)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
    Output:
      C11 = M1 + M4 - M5 + M7
      C12 = M3 + M5
      C21 = M2 + M4
      C22 = M1 - M2 + M3 + M6
    """
    # Indices: A rows: i=0,1; cols: j=0,1 -> a = i*2+j
    #           B rows: j=0,1; cols: k=0,1 -> b = j*2+k
    #           C rows: i=0,1; cols: k=0,1 -> c = k*2+i (trace convention: c = k*n+i, n=2)
    
    # For M_{2,2,2}: n=2, m=2, p=2
    # A[i,j]: a = i*2 + j
    # B[j,k]: b = j*2 + k
    # C[i,k]: c = k*2 + i
    
    U = []
    V = []
    W = []
    
    # M1 = (A00+A11)(B00+B11)
    U.append({0: 1, 3: 1})   # A00, A11
    V.append({0: 1, 3: 1})   # B00, B11
    W.append({0: 1, 3: 1})   # C00 (k=0,i=0), C11 (k=1,i=1)
    
    # M2 = (A10+A11)B00
    U.append({2: 1, 3: 1})   # A10, A11
    V.append({0: 1})          # B00
    W.append({2: 1, 3: 1})   # C10 (k=0,i=1), C11 (k=1,i=1)
    
    # M3 = A00(B10-B11)
    U.append({0: 1})          # A00
    V.append({2: 1, 3: -1})  # B10, B11
    W.append({1: 1, 3: 1})   # C01 (k=1,i=0), C11 (k=1,i=1)
    
    # M4 = A11(B01-B00)
    U.append({3: 1})          # A11
    V.append({1: 1, 0: -1})  # B01, B00
    W.append({0: 1, 2: 1})   # C00 (k=0,i=0), C10 (k=0,i=1)
    
    # M5 = (A00+A01)B11
    U.append({0: 1, 1: 1})   # A00, A01
    V.append({3: 1})          # B11
    W.append({0: -1, 1: 1})  # C00 neg, C01 pos
    
    # M6 = (A10-A00)(B00+B01)
    U.append({2: 1, 0: -1})  # A10, A00
    V.append({0: 1, 1: 1})   # B00, B01
    W.append({3: 1})         # C11
    
    # M7 = (A01-A11)(B10+B11)
    U.append({1: 1, 3: -1})  # A01, A11
    V.append({2: 1, 3: 1})   # B10, B11
    W.append({0: 1, 1: -1})  # C00 pos, C01 neg
    
    # Convert to dense matrices
    n, m, p = 2, 2, 2
    rank = 7
    Umat = [[0]*rank for _ in range(n*m)]
    Vmat = [[0]*rank for _ in range(m*p)]
    Wmat = [[0]*rank for _ in range(p*n)]
    for r in range(rank):
        for idx, val in U[r].items():
            Umat[idx][r] = val
        for idx, val in V[r].items():
            Vmat[idx][r] = val
        for idx, val in W[r].items():
            Wmat[idx][r] = val
    
    return {"U": Umat, "V": Vmat, "W": Wmat}

def _block_strassen_decomposition():
    """
    Construct a block decomposition for M_{2,5,5} using Strassen on 2x2 blocks.
    
    Split:
    - A is 2x5: rows i=0,1; cols j=0,1,2,3,4
    - B is 5x5: rows j=0,1,2,3,4; cols k=0,1,2,3,4
    - C is 2x5: rows i=0,1; cols k=0,1,2,3,4
    
    Block split:
    - Block 1: j={0,1}, k={0,1} -> M_{2,2,2} with Strassen (rank 7)
    - Block 2: j={0,1}, k={2,3,4} -> M_{2,2,3} schoolbook (rank 12)
    - Block 3: j={2,3,4}, k={0,1} -> M_{2,3,2} schoolbook (rank 12)
    - Block 4: j={2,3,4}, k={2,3,4} -> M_{2,3,3} schoolbook (rank 18)
    
    Total rank: 7 + 12 + 12 + 18 = 49
    
    Wait, that's only 49. Let me reconsider.
    
    Actually, let me try a better split:
    - Block 1: j={0,1}, k={0,1} -> Strassen (rank 7)
    - Block 2: j={0,1}, k={2,3,4} -> schoolbook (rank 2*2*3=12)
    - Block 3: j={2,3,4}, k={0,1} -> schoolbook (rank 2*3*2=12)
    - Block 4: j={2,3,4}, k={2,3,4} -> schoolbook (rank 2*3*3=18)
    Total: 7+12+12+18 = 49
    
    Hmm, only 1 better. Let me try using Strassen more aggressively.
    
    Alternative: split into 2x2 + 2x3 + 1x2 + 1x3 for the j/k blocks?
    But n=2 is fixed.
    
    Let me try:
    - j={0,1}, k={0,1}: Strassen rank 7
    - j={0,1}, k={2,3}: schoolbook rank 8
    - j={0,1}, k={4}: schoolbook rank 4
    - j={2,3}, k={0,1}: schoolbook rank 8
    - j={2,3}, k={2,3}: schoolbook rank 8
    - j={2,3}, k={4}: schoolbook rank 4
    - j={4}, k={0,1}: schoolbook rank 4
    - j={4}, k={2,3}: schoolbook rank 4
    - j={4}, k={4}: schoolbook rank 2
    Total: 7+8+4+8+8+4+4+4+2 = 49. Still 49.
    
    The issue is that only the 2x2x2 block benefits from Strassen.
    
    Let me try a different approach: use Strassen on multiple 2x2 blocks.
    If I split k={0,1,2,3,4} into pairs: {0,1}, {2,3}, {4}
    And j={0,1,2,3,4} into pairs: {0,1}, {2,3}, {4}
    
    Then I have 3x3 = 9 blocks, each of size M_{2,j_size,k_size}:
    - (2,2) blocks: 4 of them, each can use Strassen if both j and k are size 2
    - (2,1) blocks: 4 of them, schoolbook rank 4
    - (1,2) blocks: 4 of them, schoolbook rank 4  
    - (1,1) blocks: 1 of them, schoolbook rank 2
    
    Wait, I need to be more careful. The blocks are:
    j_groups = [{0,1}, {2,3}, {4}]
    k_groups = [{0,1}, {2,3}, {4}]
    
    Block (j_g, k_g) has size M_{2, |j_g|, |k_g|}
    
    (2,2): 4 blocks -> Strassen rank 7 each = 28
    (2,1): 2 blocks -> schoolbook rank 4 each = 8
    (1,2): 2 blocks -> schoolbook rank 4 each = 8
    (1,1): 1 block -> schoolbook rank 2 = 2
    Total: 28+8+8+2 = 46
    
    That's better! Rank 46.
    """
    n = 2
    m = 5
    p = 5
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_block(j_indices, k_indices, use_strassen=False):
        """Add a block decomposition for A[:,j_indices] * B[j_indices,k_indices] -> C[:,k_indices]"""
        if use_strassen and len(j_indices) == 2 and len(k_indices) == 2:
            # Use Strassen for this 2x2x2 block
            # j_indices = [j0, j1], k_indices = [k0, k1]
            # A indices: i*m + j for i in {0,1}, j in j_indices
            # B indices: j*p + k for j in j_indices, k in k_indices
            # C indices: k*n + i for k in k_indices, i in {0,1}
            
            j0, j1 = j_indices
            k0, k1 = k_indices
            
            # Map local Strassen indices to global
            # Local A: [A00, A01, A10, A11] -> global [i*m+j0, i*m+j1] for i=0,1
            # Local B: [B00, B01, B10, B11] -> global [j0*p+k0, j0*p+k1, j1*p+k0, j1*p+k1]
            # Local C: [C00, C01, C10, C11] -> global [k0*n+0, k0*n+1, k1*n+0, k1*n+1]
            
            strassen = _strassen_2x2x2()
            # Map local indices to global
            a_map = {0: 0*m+j0, 1: 0*m+j1, 2: 1*m+j0, 3: 1*m+j1}
            b_map = {0: j0*p+k0, 1: j0*p+k1, 2: j1*p+k0, 3: j1*p+k1}
            c_map = {0: k0*n+0, 1: k0*n+1, 2: k1*n+0, 3: k1*n+1}
            
            for r in range(7):
                U_row = [0] * (n*m)
                V_row = [0] * (m*p)
                W_row = [0] * (p*n)
                for local_a, val in enumerate(strassen["U"][r]):
                    if val != 0:
                        U_row[a_map[local_a]] += val
                for local_b, val in enumerate(strassen["V"][r]):
                    if val != 0:
                        V_row[b_map[local_b]] += val
                for local_c, val in enumerate(strassen["W"][r]):
                    if val != 0:
                        W_row[c_map[local_c]] += val
                U_list.append(U_row)
                V_list.append(V_row)
                W_list.append(W_row)
        else:
            # Schoolbook for this block
            for j in j_indices:
                for k in k_indices:
                    for i in range(n):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        U_row = [0] * (n*m)
                        V_row = [0] * (m*p)
                        W_row = [0] * (p*n)
                        U_row[a] = 1
                        V_row[b] = 1
                        W_row[c] = 1
                        U_list.append(U_row)
                        V_list.append(V_row)
                        W_list.append(W_row)
    
    # Define blocks
    j_groups = [[0, 1], [2, 3], [4]]
    k_groups = [[0, 1], [2, 3], [4]]
    
    for j_g in j_groups:
        for k_g in k_groups:
            use_strassen = (len(j_g) == 2 and len(k_g) == 2)
            add_block(j_g, k_g, use_strassen=use_strassen)
    
    rank = len(U_list)
    return {"U": U_list, "V": V_list, "W": W_list, "rank": rank}

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x5 matrix by a 5x5 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Try block Strassen decomposition
    try:
        result = _block_strassen_decomposition()
        rank = result["rank"]
        U, V, W = result["U"], result["V"], result["W"]
        
        # Verify the decomposition
        if _verify(U, V, W, n, m, p, tol=1e-9):
            if time.time() > deadline - 5.0:
                pass  # Still return, we're done
            return {"U": U, "V": V, "W": W}
        else:
            # Verification failed, fall back to schoolbook
            return _schoolbook(n, m, p)
    except Exception:
        # Fall back to schoolbook on any error
        return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
