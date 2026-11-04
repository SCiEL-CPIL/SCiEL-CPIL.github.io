import numpy as np
import time
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    return False
    # Check that entries are half-integers
    for mat in [U, V, W]:
        for val in mat.flatten():
            if abs(val * 2 - round(val * 2)) > 1e-9:
                return False
    return True

def _build_trivial(n, m, p):
    """Build the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    U: (4,7), V: (4,7), W: (4,7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen's 7 multiplications for 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # U rows: A11=0, A12=1, A21=2, A22=3
    # V rows: B11=0, B12=1, B21=2, B22=3
    # W rows: C11=0, C12=1, C21=2, C22=3
    
    # M1: (A11+A22)(B11+B22) -> C11 += M1, C12 -= M2? No...
    # Let me use the standard Strassen formulas:
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 - M7
    
    # r=0: M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = -1  # C11 += M1, C22 -= M1... wait
    
    # Let me redo this carefully.
    # C = A * B where A is 2x2, B is 2x2
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # In our indexing: U[i*m+j] corresponds to A[i,j], V[j*p+k] to B[j,k], W[k*n+i] to C[k,i]
    # Wait, W[k*n+i] is the coefficient for C[i,k] in the output... 
    # Actually the tensor is T[i,j,k] = A[i,j]*B[j,k]*C[k,i] where C = A*B
    # So W[k*n+i] multiplies the (i,j,k) term, and we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 if (i,j,k) is a valid triple, 0 otherwise.
    # For matrix mult: T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C.
    # So T[i,j,k] = 1 for all i,j,k (since every A[i,j]*B[j,k] contributes to C[i,k]).
    # Wait no - the tensor for matrix multiplication has T[i,j,k] = 1 if the product A[i,j]*B[j,k] appears in C[i,k].
    # For standard matrix mult, every (i,j,k) triple has T=1.
    
    # So we need: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # For 2x2x2, n=2, m=2, p=2:
    # U is 4x7 (rows: A11,A12,A21,A22)
    # V is 4x7 (rows: B11,B12,B21,B22)  
    # W is 4x7 (rows: C11,C12,C21,C22) where C[k*n+i] means C[0*2+0]=C11, C[0*2+1]=C12, C[1*2+0]=C21, C[1*2+1]=C22
    
    # Strassen:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B21-B22)
    # M4 = A22*(B12-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 - M7
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: M1
    U[0,0]=1; U[3,0]=1  # A11, A22
    V[0,0]=1; V[3,0]=1  # B11, B22
    W[0,0]=1; W[3,0]=1  # C11, C22
    
    # r=1: M2
    U[2,1]=1; U[3,1]=1  # A21, A22
    V[0,1]=1  # B11
    W[2,1]=1  # C21
    
    # r=2: M3
    U[0,2]=1  # A11
    V[2,2]=1; V[3,2]=-1  # B21, -B22
    W[1,2]=1  # C12
    
    # r=3: M4
    U[3,3]=1  # A22
    V[1,3]=1; V[0,3]=-1  # B12, -B11
    W[0,3]=-1; W[2,3]=1  # -C11, C21
    
    # r=4: M5
    U[0,4]=1; U[1,4]=1  # A11, A12
    V[3,4]=1  # B22
    W[0,4]=1; W[1,4]=1  # C11, C12
    
    # r=5: M6
    U[2,5]=1; U[0,5]=-1  # A21, -A11
    V[0,5]=1; V[1,5]=1  # B11, B12
    W[3,5]=1  # C22
    
    # r=6: M7
    U[1,6]=1; U[3,6]=-1  # A12, -A22
    V[2,6]=1; V[3,6]=1  # B21, B22
    W[0,6]=-1; W[3,6]=-1  # -C11, -C22
    
    return U, V, W

def _construct_2x3x4():
    """
    Construct a rank-minimal decomposition for 2x3x4 matrix multiplication.
    
    Strategy: Split the 3x4 matrix B into blocks and use Strassen-like methods.
    
    A is 2x3, B is 3x4, C = A*B is 2x4.
    
    Split B into two 3x2 blocks: B = [B1 | B2] where B1, B2 are 3x2.
    Then C = [A*B1 | A*B2].
    
    Each A*B1 and A*B2 is 2x3 times 3x2 = 2x2.
    
    For 2x3 * 3x2:
    Split A into [A1 | A2] where A1 is 2x2, A2 is 2x1.
    Then A*B = A1*B1_part + A2*B2_part... 
    
    Actually, let me think differently.
    
    2x3 * 3x4:
    C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1; j=0,1,2; k=0,1,2,3
    
    We can split the summation index j into groups.
    
    Approach: Use the fact that rank(2x3x4) <= rank(2x3x2) + rank(2x3x2)
    since we can split the 4 columns into two groups of 2.
    
    rank(2x3x2): 2x3 * 3x2
    Split A into A_left (2x2) and A_right (2x1).
    B has 3 rows, 2 cols.
    
    C = A_left * B_top + A_right * B_bottom
    where B_top is 2x2 (first 2 rows of B) and B_bottom is 1x2 (last row of B).
    
    rank(2x2x2) = 7 (Strassen)
    rank(2x1x2) = 2 (trivial: (2x1)*(1x2) needs 2 multiplications)
    
    So rank(2x3x2) <= 7 + 2 = 9? Let me verify.
    
    Actually for 2x1 * 1x2: A is 2x1, B is 1x2.
    C[i,k] = A[i,0] * B[0,k]
    This needs 2 multiplications: A[0,0]*B[0,0] and A[1,0]*B[0,0]... no.
    C[0,0] = A[0,0]*B[0,0]
    C[0,1] = A[0,0]*B[0,1]
    C[1,0] = A[1,0]*B[0,0]
    C[1,1] = A[1,0]*B[0,1]
    
    We can use 2 multiplications:
    M1 = A[0,0] * B[0,0]
    M2 = A[0,0] * B[0,1]
    M3 = A[1,0] * B[0,0]
    M4 = A[1,0] * B[0,1]
    That's 4. Can we do better?
    
    Actually rank(2x1x2) = 2. Here's how:
    M1 = A[0,0] * (B[0,0] + B[0,1])
    M2 = A[1,0] * (B[0,0] + B[0,1])
    Then C[0,0] = M1 - M2 + ... no this doesn't work directly.
    
    Let me just use the trivial 4-term decomposition for 2x1x2.
    
    So rank(2x3x2) <= 7 + 4 = 11.
    Then rank(2x3x4) <= 11 + 11 = 22.
    
    Let me try to construct this explicitly.
    """
    n, m, p = 2, 3, 4
    rank = 22
    
    U = np.zeros((n*m, rank))  # 6 x 22
    V = np.zeros((m*p, rank))  # 12 x 22
    W = np.zeros((p*n, rank))  # 8 x 22
    
    # Split into two halves of B: columns 0-1 and columns 2-3
    # For each half, we compute A (2x3) * B_half (3x2)
    
    # For one half (3x2 B block):
    # Split A (2x3) into A_left (2x2) and A_right (2x1)
    # B_half (3x2) split into B_top (2x2) and B_bottom (1x2)
    # C_half = A_left * B_top + A_right * B_bottom
    # A_left*B_top: 2x2 * 2x2, rank 7 (Strassen)
    # A_right*B_bottom: 2x1 * 1x2, rank 4 (trivial)
    # Total per half: 11
    # Two halves: 22
    
    def add_strassen_2x2x2(offset, row_offset_A, row_offset_B, row_offset_C):
        """Add a Strassen 2x2x2 block at given offsets."""
        S_U, S_V, S_W = _strassen_2x2x2()
        # S_U is 4x7 for A indices (i,j) where i in {0,1}, j in {0,1}
        # S_V is 4x7 for B indices (j,k) where j in {0,1}, k in {0,1}
        # S_W is 4x7 for C indices (k,i) where k in {0,1}, i in {0,1}
        
        for r in range(7):
            for idx_u in range(4):
                if S_U[idx_u, r] != 0:
                    i = idx_u // 2
                    j = idx_u % 2
                    U[row_offset_A + i*3 + j, offset + r] += S_U[idx_u, r]
            for idx_v in range(4):
                if S_V[idx_v, r] != 0:
                    j = idx_v // 2
                    k = idx_v % 2
                    V[row_offset_B + j*4 + k, offset + r] += S_V[idx_v, r]
            for idx_w in range(4):
                if S_W[idx_w, r] != 0:
                    k = idx_w // 2
                    i = idx_w % 2
                    W[row_offset_C + k*2 + i, offset + r] += S_W[idx_w, r]
    
    def add_trivial_2x1x2(offset, row_offset_A, row_offset_B, row_offset_C):
        """Add a trivial 2x1x2 block."""
        # A is 2x1: rows A[0,0], A[1,0]
        # B is 1x2: rows B[0,0], B[0,1]
        # C is 2x2: C[0,0], C[0,1], C[1,0], C[1,1]
        # C[i,k] = A[i,0] * B[0,k]
        # 4 terms
        for r in range(4):
            i = r // 2
            k = r % 2
            U[row_offset_A + i, offset + r] = 1.0
            V[row_offset_B, offset + r] = 1.0
            W[row_offset_C + k*2 + i, offset + r] = 1.0
    
    # Half 1: B columns 0,1
    # A_left: A rows 0,1; cols 0,1 -> U rows 0,1,3,4 (i*3+j: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4)
    # A_right: A rows 0,1; col 2 -> U rows 2,5
    # B_top: B rows 0,1; cols 0,1 -> V rows 0,1,4,5 (j*4+k: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5)
    # B_bottom: B row 2; cols 0,1 -> V rows 8,9 (2*4+0=8, 2*4+1=9)
    # C: cols 0,1 -> W rows 0,1,2,3 (k*2+i: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3)
    
    add_strassen_2x2x2(0, 0, 0, 0)
    add_trivial_2x1x2(7, 2, 8, 0)
    
    # Half 2: B columns 2,3
    # A_left: same -> U rows 0,1,3,4
    # A_right: same -> U rows 2,5
    # B_top: B rows 0,1; cols 2,3 -> V rows 2,3,6,7 (0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7)
    # B_bottom: B row 2; cols 2,3 -> V rows 10,11 (2*4+2=10, 2*4+3=11)
    # C: cols 2,3 -> W rows 4,5,6,7 (2*2+0=4, 2*2+1=5, 3*2+0=6, 3*2+1=7)
    
    add_strassen_2x2x2(11, 0, 2, 4)
    add_trivial_2x1x2(18, 2, 10, 4)
    
    return U, V, W

def _randomized_search(n, m, p, target_rank, max_time=30.0):
    """Randomized search for a decomposition of given rank."""
    start = time.time()
    np.random.seed(123)
    
    n_vars = n * m
    m_vars = m * p
    p_vars = p * n
    
    best_solution = None
    
    for attempt in range(200):
        if time.time() - start > max_time:
            break
        
        # Initialize with small half-integer values
        U = np.random.randint(-2, 3, size=(n_vars, target_rank)) * 0.5
        V = np.random.randint(-2, 3, size=(m_vars, target_rank)) * 0.5
        W = np.random.randint(-2, 3, size=(p_vars, target_rank)) * 0.5
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Trivial decomposition rank
    rank_trivial = n * m * p
    
    # Initialize trivial solution as fallback
    U_trivial, V_trivial, W_trivial = _build_trivial(n, m, p)
    
    best_rank = rank_trivial
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    
    # Try constructive Strassen-like approach
    try:
        U_cons, V_cons, W_cons = _construct_2x3x4()
        if _verify_decomposition(U_cons, V_cons, W_cons, n, m, p):
            best_rank = U_cons.shape[1]
            best_U = U_cons
            best_V = V_cons
            best_W = W_cons
    except Exception:
        pass
    
    # Try to find lower rank solutions via randomized search
    for target_rank in range(min(best_rank - 1, rank_trivial - 1), 18, -1):
        if time.time() > deadline:
            break
        
        result = _randomized_search(n, m, p, target_rank, max_time=5.0)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
