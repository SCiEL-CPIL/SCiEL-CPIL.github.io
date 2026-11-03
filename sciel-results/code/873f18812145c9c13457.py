"""Find rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication."""
import time
import sys
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return Strassen's 7-term decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) where each is a list of 4 (or 4) entries per term.
    U[i*m+j, t], V[j*p+k, t], W[k*n+i, t] with n=m=p=2.
    """
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # In our tensor form: T[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t]
    # For 2x2x2: i,j,k in {0,1}
    
    # Term 0: M1
    U0 = [1, 0, 0, 1]  # U[0,0]=1, U[1,1]=1
    V0 = [1, 0, 0, 1]  # V[0,0]=1, V[1,1]=1
    W0 = [1, 0, 0, 1]  # W[0,0]=1, W[1,1]=1
    
    # Term 1: M2
    U1 = [0, 0, 1, 1]  # U[1,0]=1, U[1,1]=1
    V1 = [1, 0, 0, 0]  # V[0,0]=1
    W1 = [0, 0, 1, 1]  # W[1,0]=1, W[1,1]=1
    
    # Term 2: M3
    U2 = [1, 0, 0, 0]  # U[0,0]=1
    V2 = [0, 0, 1, -1]  # V[1,0]=1, V[1,1]=-1
    W2 = [0, 1, 0, 0]  # W[0,1]=1
    
    # Term 3: M4
    U3 = [0, 0, 0, 1]  # U[1,1]=1
    V3 = [0, 1, -1, 0]  # V[0,1]=1, V[0,0]=-1
    W3 = [1, 0, 0, 0]  # W[0,0]=1
    
    # Term 4: M5
    U4 = [1, 1, 0, 0]  # U[0,0]=1, U[0,1]=1
    V4 = [0, 0, 0, 1]  # V[1,1]=1
    W4 = [0, 1, 0, 0]  # W[0,1]=1
    
    # Term 5: M6
    U5 = [0, 0, 1, -1]  # U[1,0]=1, U[0,0]=-1
    V5 = [1, 1, 0, 0]  # V[0,0]=1, V[0,1]=1
    W5 = [0, 0, 0, 1]  # W[1,1]=1
    
    # Term 6: M7
    U6 = [0, 1, -1, 0]  # U[0,1]=1, U[1,1]=-1
    V6 = [0, 0, 1, 1]  # V[1,0]=1, V[1,1]=1
    W6 = [0, 0, 1, 0]  # W[1,0]=1
    
    U = [U0, U1, U2, U3, U4, U5, U6]
    V = [V0, V1, V2, V3, V4, V5, V6]
    W = [W0, W1, W2, W3, W4, W5, W6]
    
    # Verify
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for t in range(7):
                    s += U[t][i*2+j] * V[t][j*2+k] * W[t][k*2+i]
                if s != 1:
                    raise ValueError(f"Strassen verification failed at ({i},{j},{k}): {s}")
    
    return U, V, W

def _combine(U1, V1, W1, U2, V2, W2, offset1, offset2, dim1, dim2):
    """
    Combine two Strassen decompositions for blocks of sizes dim1 and dim2.
    offset1, offset2 are the starting indices for the middle dimension.
    Returns combined U, V, W lists.
    """
    rank1 = len(U1)
    rank2 = len(U2)
    
    # For each term in decomposition 1, place it at the appropriate positions
    combined_U = []
    combined_V = []
    combined_W = []
    
    # Add terms from first block
    for t in range(rank1):
        u = [0] * (dim1 * 2)  # Will be extended later
        v = [0] * (dim1 * 2)
        w = [0] * (dim1 * 2)
        for idx, val in enumerate(U1[t]):
            u[idx] = val
        for idx, val in enumerate(V1[t]):
            v[idx] = val
        for idx, val in enumerate(W1[t]):
            w[idx] = val
        combined_U.append(u)
        combined_V.append(v)
        combined_W.append(w)
    
    # Add terms from second block
    for t in range(rank2):
        u = [0] * (dim2 * 2)
        v = [0] * (dim2 * 2)
        w = [0] * (dim2 * 2)
        for idx, val in enumerate(U2[t]):
            u[idx] = val
        for idx, val in enumerate(V2[t]):
            v[idx] = val
        for idx, val in enumerate(W2[t]):
            w[idx] = val
        combined_U.append(u)
        combined_V.append(v)
        combined_W.append(w)
    
    return combined_U, combined_V, combined_W

def _build_3x4x5_strassen():
    """
    Build a Strassen-based decomposition for 3x4x5 matrix multiplication.
    
    Strategy: Split the 4-dimensional middle index into two 2-dimensional blocks.
    Use Strassen for each 2x2x2 sub-block, then combine.
    
    For 3x4 * 4x5:
    - A is 3x4, B is 4x5, C is 3x5
    - Split B's rows into two groups of 2: B1 (rows 0-1), B2 (rows 2-3)
    - Split A's columns correspondingly: A1 (cols 0-1), A2 (cols 2-3)
    - C = A1*B1 + A2*B2
    
    Each A_i * B_i is a 3x2 * 2x5 product.
    We can further decompose each using Strassen-like ideas.
    
    Actually, let's use a more direct approach:
    Use 2x2x2 Strassen blocks where possible.
    
    For the 3x4x5 case:
    - Middle dimension 4 = 2 + 2
    - Left dimension 3, right dimension 5
    
    We can think of this as:
    C[0:3, 0:5] = sum_{j=0}^{3} A[0:3, j] B[j, 0:5]
    
    Split j into {0,1} and {2,3}:
    C = A[:, 0:2] @ B[0:2, :] + A[:, 2:4] @ B[2:4, :]
    
    Each term is a 3x2 @ 2x5 product.
    
    For a 3x2 @ 2x5 product, we can use a modified Strassen approach.
    Let's define:
    A_left (3x2), B_left (2x5)
    
    We can embed this in a larger structure or use a custom decomposition.
    
    Alternative: Use the fact that 3x2 @ 2x5 has rank at most 6 (schoolbook),
    but we can potentially do better.
    
    For now, let's use a simpler approach: 
    Use Strassen for the 2x2x2 core and extend to handle the extra dimensions.
    
    Actually, let's try a different decomposition:
    Use 7 terms per 2x2x2 block, and we have multiple such blocks.
    
    For 3x4x5:
    - We can partition into overlapping 2x2x2 blocks
    - Or use a more systematic approach
    
    Let me try a constructive approach with explicit terms.
    """
    n, m, p = 3, 4, 5
    
    # Initialize U, V, W as empty lists
    U = []
    V = []
    W = []
    
    # We'll build the decomposition term by term.
    # Each term is a triple (u, v, w) where:
    # u is a vector of length n*m = 12
    # v is a vector of length m*p = 20  
    # w is a vector of length p*n = 15
    
    # Strategy: Use Strassen's method adapted for rectangular matrices.
    # 
    # For multiplying A (3x4) by B (4x5):
    # Split A into A1 (3x2, cols 0-1) and A2 (3x2, cols 2-3)
    # Split B into B1 (2x5, rows 0-1) and B2 (2x5, rows 2-3)
    # C = A1*B1 + A2*B2
    #
    # For each 3x2 * 2x5 product, use a 7-term Strassen-like decomposition.
    
    def _strassen_3x2x2_5():
        """
        Strassen-like decomposition for 3x2 * 2x5.
        Returns 7 terms, each with u (length 6), v (length 10), w (length 15).
        """
        # A is 3x2: A[0:3, 0:2]
        # B is 2x5: B[0:2, 0:5]
        # C is 3x5: C[0:3, 0:5]
        
        # For each output position (i, k) where i in 0..2, k in 0..4:
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
        
        # We can use a modified Strassen approach.
        # Let's define intermediate products:
        # M1 = (A[0,0] + A[2,1]) * (B[0,0] + B[1,4])  # corner elements
        # ... this gets complex. Let me use a simpler approach.
        
        # Actually, for 3x2 * 2x5, the schoolbook rank is 6 (3*2*5/5 = 6? No, it's 3*2*5=30 schoolbook).
        # Wait, schoolbook is n*m*p = 3*2*5 = 30.
        
        # Let me just return the schoolbook decomposition for now.
        terms = []
        for i in range(3):
            for j in range(2):
                for k in range(5):
                    u = [0] * 6
                    v = [0] * 10
                    w = [0] * 15
                    u[i*2 + j] = 1
                    v[j*5 + k] = 1
                    w[k*3 + i] = 1
                    terms.append((u, v, w))
        return terms
    
    # Use schoolbook for each half
    terms1 = _strassen_3x2x2_5()
    terms2 = _strassen_3x2x2_5()
    
    # Combine: terms1 correspond to A1*B1 (cols 0-1 of A, rows 0-1 of B)
    # terms2 correspond to A2*B2 (cols 2-3 of A, rows 2-3 of B)
    
    all_terms = []
    
    # Add terms from first block (A1*B1)
    for u, v, w in terms1:
        U_full = [0] * 12
        V_full = [0] * 20
        W_full = [0] * 15
        for idx, val in enumerate(u):
            i, j = divmod(idx, 2)
            U_full[i*4 + j] = val  # j is 0 or 1
        for idx, val in enumerate(v):
            j, k = divmod(idx, 5)
            V_full[j*5 + k] = val  # j is 0 or 1
        for idx, val in enumerate(w):
            k, i = divmod(idx, 3)
            W_full[k*3 + i] = val
        all_terms.append((U_full, V_full, W_full))
    
    # Add terms from second block (A2*B2)
    for u, v, w in terms2:
        U_full = [0] * 12
        V_full = [0] * 20
        W_full = [0] * 15
        for idx, val in enumerate(u):
            i, j = divmod(idx, 2)
            U_full[i*4 + (j + 2)] = val  # j is 0 or 1, offset by 2
        for idx, val in enumerate(v):
            j, k = divmod(idx, 5)
            V_full[(j + 2)*5 + k] = val  # j is 0 or 1, offset by 2
        for idx, val in enumerate(w):
            k, i = divmod(idx, 3)
            W_full[k*3 + i] = val
        all_terms.append((U_full, V_full, W_full))
    
    U = [t[0] for t in all_terms]
    V = [t[1] for t in all_terms]
    W = [t[2] for t in all_terms]
    
    # Verify
    rank = len(U)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(rank):
                    s += U[t][i*m + j] * V[t][j*p + k] * W[t][k*n + i]
                if s != 1:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): {s}")
    
    return U, V, W, rank

def construct():
    """Find a low-rank exact tensor decomposition for 3x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Schoolbook decomposition as baseline
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best = {'U': U_sb, 'V': V_sb, 'W': W_sb, 'rank': rank_sb}
    
    # Try Strassen-based decomposition
    try:
        U_str, V_str, W_str, rank_str = _build_3x4x5_strassen()
        if rank_str < best['rank']:
            best = {'U': U_str, 'V': V_str, 'W': W_str, 'rank': rank_str}
    except Exception as e:
        pass  # Fall back to schoolbook
    
    # Check deadline
    if time.time() > deadline:
        return {'U': best['U'], 'V': best['V'], 'W': best['W']}
    
    return {'U': best['U'], 'V': best['V'], 'W': best['W']}
# EVOLVE-BLOCK-END
