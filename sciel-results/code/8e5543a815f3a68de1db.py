"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_2x2():
    """Return rank-7 factor matrices for 2x2 by 2x2 matrix multiplication.
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    n = 2
    # Strassen's 7 products for 2x2 * 2x2
    # A = [[a,b],[c,d]], B = [[x,y],[z,w]]
    # P1 = (a+d)(x+w)
    # P2 = (c+d)(x)
    # P3 = (a)(y+w)
    # P4 = (d)(z-y)
    # P5 = (a+b)(w)
    # P6 = (c-a)(x+z)
    # P7 = (b-d)(y+w)
    
    # U: rows = A elements (0:a, 1:b, 2:c, 3:d), cols = 7 products
    # V: rows = B elements (0:x, 1:y, 2:z, 3:w), cols = 7 products
    # W: rows = C elements (0:c00, 1:c01, 2:c10, 3:c11), cols = 7 products
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Row indices in U: 0=a, 1=b, 2=c, 3=d
    # Row indices in V: 0=x, 1=y, 2=z, 3=w
    # Row indices in W: 0=C[0,0], 1=C[0,1], 2=C[1,0], 3=C[1,1]
    
    # P1 = (a+d)(x+w): U[a]=1, U[d]=1; V[x]=1, V[w]=1
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    # C00 = P1 - P4 - P5 + P7, C01 = P3 + P5, C10 = P2 + P4, C11 = P1 - P2 - P3 + P6
    W[0,0] = 1; W[1,0] = 0; W[2,0] = 0; W[3,0] = 1  # contributes to C00 and C11
    
    # P2 = (c+d)(x): U[c]=1, U[d]=1; V[x]=1
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[0,1] = 0; W[1,1] = 0; W[2,1] = 1; W[3,1] = -1  # C10 += P2, C11 -= P2
    
    # P3 = a*(y+w): U[a]=1; V[y]=1, V[w]=1
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = 1
    W[0,2] = 0; W[1,2] = 1; W[2,2] = 0; W[3,2] = -1  # C01 += P3, C11 -= P3
    
    # P4 = d*(z-y): U[d]=1; V[z]=1, V[y]=-1
    U[3,3] = 1
    V[2,3] = 1; V[1,3] = -1
    W[0,3] = -1; W[1,3] = 0; W[2,3] = 1; W[3,3] = 0  # C00 -= P4, C10 += P4
    
    # P5 = (a+b)*w: U[a]=1, U[b]=1; V[w]=1
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1; W[2,4] = 0; W[3,4] = 0  # C00 -= P5, C01 += P5
    
    # P6 = (c-a)*(x+z): U[c]=1, U[a]=-1; V[x]=1, V[z]=1
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[2,5] = 1
    W[0,5] = 0; W[1,5] = 0; W[2,5] = 0; W[3,5] = 1  # C11 += P6
    
    # P7 = (b-d)*(y+w): U[b]=1, U[d]=-1; V[y]=1, V[w]=1
    U[1,6] = 1; U[3,6] = -1
    V[1,6] = 1; V[3,6] = 1
    W[0,6] = 1; W[1,6] = 0; W[2,6] = 0; W[3,6] = 0  # C00 += P7
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the factorization exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    assert U.shape[0] == n * m
    assert V.shape[0] == m * p
    assert W.shape[0] == p * n
    assert V.shape[1] == rank
    assert W.shape[1] == rank
    
    # Build the tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix mult: T[i,j,k] should be 1 if... wait, no.
    # Actually the tensor for matrix multiplication is:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T_{ijk} = A_{ij} B_{jk} C_{ki} where C = AB
    # So the tensor entry at (i,j,k) is 1 for the "standard" tensor? No.
    # The matrix multiplication tensor has T_{ijk} = 1 when... 
    # Actually, the decomposition means:
    # sum_c U_{(i,j),c} * V_{(j,k),c} * W_{(k,i),c} = 1 for all i,j,k? No, that's not right either.
    # 
    # Let me reconsider. The matrix multiplication tensor T satisfies:
    # T_{ijk} = 1 if the element (i,j) of A and (j,k) of B contribute to (i,k) of C.
    # Actually T_{ijk} is just 1 for all valid i,j,k in the standard tensor.
    # The decomposition is: T_{ijk} = sum_c U_{ij,c} V_{jk,c} W_{ki,c}
    # where T_{ijk} = 1 for all i,j,k (since every triple contributes).
    
    # Let's verify: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_block_decomposition(n, m, p):
    """Build a Strassen-based block decomposition for n x m times m x p.
    For n=2, m=3, p=6, we can view A as 2x3 and B as 3x6.
    
    One approach: view the 2x3 matrix as having one 2x2 block and a 2x1 block,
    and the 3x6 matrix as having a 2x6 part and 1x6 part.
    
    Actually, a cleaner approach for 2x3 * 3x6:
    Split A (2x3) into A1 (2x2) and A2 (2x1)
    Split B (3x6) into B1 (2x6) and B2 (1x6)
    
    C = A1*B1 + A2*B2
    
    A1*B1: 2x2 * 2x6. We can do this with 2 Strassen 2x2 multiplications
    (treating B1 as two 2x2 blocks side by side). That's 2*7 = 14 products.
    A2*B2: 2x1 * 1x6 = 12 products (schoolbook).
    
    Total: 14 + 12 = 26 products.
    
    Let me implement this.
    """
    n, m, p = 2, 3, 6
    
    # Split: A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    # B = [B1; B2] where B1 is 2x6, B2 is 1x6
    # C = A1*B1 + A2*B2
    
    # A1*B1: A1 is 2x2, B1 is 2x6 = two 2x2 blocks
    # Use Strassen for each 2x2 * 2x2 block: 2 blocks * 7 = 14 products
    # A2*B2: 2x1 * 1x6: 2*6 = 12 schoolbook products
    # Total rank = 14 + 12 = 26
    
    rank = 26
    U = np.zeros((n*m, rank), dtype=float)  # 6 x 26
    V = np.zeros((m*p, rank), dtype=float)  # 18 x 26
    W = np.zeros((p*n, rank), dtype=float)  # 12 x 26
    
    # A element indices: A[i,j] -> U index i*3+j
    # i in {0,1}, j in {0,1,2}
    # A1: j in {0,1}, A2: j in {2}
    
    # B element indices: B[j,k] -> V index j*6+k
    # j in {0,1,2}, k in {0,...,5}
    # B1: j in {0,1}, B2: j in {2}
    
    # C element indices: C[i,k] -> W index k*2+i
    # i in {0,1}, k in {0,...,5}
    
    # Part 1: A1 * B1 (14 Strassen products)
    S2 = _strassen_2x2_2x2()
    U2, V2, W2 = S2
    # U2 is 4x7: rows are A1 elements (a,b,c,d) = A[0,0],A[0,1],A[1,0],A[1,1]
    # V2 is 4x7: rows are B elements (x,y,z,w) for a 2x2 block
    # W2 is 4x7: rows are C elements (C00,C01,C10,C11) for a 2x2 block
    
    # B1 has two 2x2 blocks: columns 0-1 and 2-3? No, B1 is 2x6.
    # Let's use 3 blocks of 2x2: columns 0-1, 2-3, 4-5. That's 3*7=21 products.
    # Wait, 2x6 = three 2x2 blocks? No, 6 columns = three pairs.
    # Actually 2x6 can be viewed as three 2x2 blocks. So 3 * 7 = 21 products for A1*B1.
    # A2*B2: 12 products. Total = 33.
    
    # Hmm, that gives rank 33. Let me reconsider.
    # Actually, let me just do it properly.
    
    # A1*B1: A1 is 2x2, B1 is 2x6.
    # View B1 as 3 blocks of 2x2: B1[:,0:2], B1[:,2:4], B1[:,4:6]
    # Each block uses 7 Strassen products. Total: 21 products.
    
    # A2*B2: A2 is 2x1, B2 is 1x6.
    # C contribution: C[i,k] += A2[i,0] * B2[0,k]
    # This is 2*6 = 12 schoolbook products.
    
    # Total rank = 21 + 12 = 33
    
    rank = 33
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen part: columns 0-20
    for block in range(3):
        col_offset = block * 7
        k_start = block * 2
        
        # U mapping: A1 elements
        # A[0,0] -> U[0, col], A[0,1] -> U[1, col], A[1,0] -> U[3, col], A[1,1] -> U[4, col]
        # U2 rows: 0=a=A[0,0], 1=b=A[0,1], 2=c=A[1,0], 3=d=A[1,1]
        # U indices: A[i,j] -> i*3+j
        # a: i=0,j=0 -> U[0]
        # b: i=0,j=1 -> U[1]
        # c: i=1,j=0 -> U[3]
        # d: i=1,j=1 -> U[4]
        
        for u_row, u_val in [(0, 1), (1, 1), (2, 1), (3, 1)]:
            # u_row in U2: 0=a,1=b,2=c,3=d
            # Map to U index
            if u_row == 0:  # a = A[0,0]
                U_idx = 0
            elif u_row == 1:  # b = A[0,1]
                U_idx = 1
            elif u_row == 2:  # c = A[1,0]
                U_idx = 3
            elif u_row == 3:  # d = A[1,1]
                U_idx = 4
            U[U_idx, col_offset:col_offset+7] += U2[u_row, :] * 1.0
        
        # V mapping: B1 block elements
        # V2 rows: 0=x, 1=y, 2=z, 3=w
        # For block starting at k_start:
        # x = B[0, k_start], y = B[0, k_start+1], z = B[1, k_start], w = B[1, k_start+1]
        # V index: B[j,k] -> j*6+k
        for v_row, v_val in [(0, 1), (1, 1), (2, 1), (3, 1)]:
            if v_row == 0:  # x = B[0, k_start]
                V_idx = 0 * 6 + k_start
            elif v_row == 1:  # y = B[0, k_start+1]
                V_idx = 0 * 6 + k_start + 1
            elif v_row == 2:  # z = B[1, k_start]
                V_idx = 1 * 6 + k_start
            elif v_row == 3:  # w = B[1, k_start+1]
                V_idx = 1 * 6 + k_start + 1
            V[V_idx, col_offset:col_offset+7] += V2[v_row, :] * 1.0
        
        # W mapping: C elements for this block
        # W2 rows: 0=C[0,0], 1=C[0,1], 2=C[1,0], 3=C[1,1]
        # For block starting at k_start:
        # C[0,k_start], C[0,k_start+1], C[1,k_start], C[1,k_start+1]
        # W index: C[i,k] -> k*2+i
        for w_row, w_val in [(0, 1), (1, 1), (2, 1), (3, 1)]:
            if w_row == 0:  # C[0, k_start]
                W_idx = k_start * 2 + 0
            elif w_row == 1:  # C[0, k_start+1]
                W_idx = (k_start + 1) * 2 + 0
            elif w_row == 2:  # C[1, k_start]
                W_idx = k_start * 2 + 1
            elif w_row == 3:  # C[1, k_start+1]
                W_idx = (k_start + 1) * 2 + 1
            W[W_idx, col_offset:col_offset+7] += W2[w_row, :] * 1.0
    
    # Schoolbook part for A2*B2: columns 21-32
    # A2[i,0] * B2[0,k] -> C[i,k]
    # A2: i in {0,1}, j=2 -> U index i*3+2
    # B2: j=2, k in {0,...,5} -> V index 2*6+k
    # C: i in {0,1}, k in {0,...,5} -> W index k*2+i
    
    col = 21
    for i in range(2):
        for k in range(6):
            U[i*3+2, col] = 1
            V[2*6+k, col] = 1
            W[k*2+i, col] = 1
            col += 1
    
    return U, V, W

def _convert_to_integer_or_half(U, V, W):
    """Try to convert float matrices to integer or half-integer representation.
    If entries are already integers or half-integers, keep them.
    """
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Check if all entries are integers or half-integers
    def is_clean(mat):
        for val in mat.flat:
            if abs(val * 2 - round(val * 2)) > 1e-9:
                return False
        return True
    
    if not (is_clean(U) and is_clean(V) and is_clean(W)):
        return None
    
    # Convert to integer representation by multiplying by 2
    U_int = (U * 2).round().astype(int)
    V_int = (V * 2).round().astype(int)
    W_int = (W * 2).round().astype(int)
    
    # Convert back to list of lists with integer or half-integer values
    U_list = [[int(x) for x in row] for row in U_int]
    V_list = [[int(x) for x in row] for row in V_int]
    W_list = [[int(x) for x in row] for row in W_int]
    
    return U_list, V_list, W_list

def construct():
    deadline = time.time() + 120  # 120 second deadline
    
    n, m, p = DIMENSIONS
    
    # Try Strassen-based block decomposition (rank 33)
    try:
        U, V, W = _build_strassen_block_decomposition(n, m, p)
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            # Convert to integer representation
            U_int, V_int, W_int = _convert_to_integer_or_half(U, V, W)
            if U_int is not None:
                return {"U": U_int, "V": V_int, "W": W_int}
        else:
            pass  # Verification failed, fall through to schoolbook
    except Exception:
        pass
    
    # Fallback: schoolbook rank 36
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
