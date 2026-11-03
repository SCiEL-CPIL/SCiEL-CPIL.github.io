"""Search for a rank-minimal exact tensor decomposition of 2x4 times 4x7 matrix multiply."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 7]
DEADLINE = time.time() + 110.0  # safety margin within 120s


def _verify_np(U, V, W, n, m, p, tol=1e-6):
    """Faster vectorized check. U, V, W are 2D arrays (rows, rank)."""
    for i in range(n):
        for k in range(p):
            w = W[k * n + i]
            s = 0.0
            for j in range(m):
                u = U[i * m + j]
                v = V[j * p + k]
                if np.any(u != 0) and np.any(v != 0):
                    s += np.sum(u * v * w)
            if abs(s - 1.0) > tol:
                return False
    return True


def _to_fraction_list(mat):
    """Convert numpy array to list of Fraction or int for exact representation."""
    result = []
    for row in mat:
        row_list = []
        for val in row:
            # Try to represent as fraction with small denominator
            if abs(val - round(val)) < 1e-9:
                row_list.append(int(round(val)))
            else:
                # Try half-integers
                if abs(2*val - round(2*val)) < 1e-9:
                    num = int(round(2*val))
                    if num % 2 == 0:
                        row_list.append(num // 2)
                    else:
                        row_list.append(Fraction(num, 2))
                else:
                    # Use Fraction directly
                    row_list.append(Fraction(val).limit_denominator(1000))
        result.append(row_list)
    return result


def _strassen_2x2x2():
    """
    Return the 7-term Strassen decomposition for 2x2 * 2x2.
    Returns U, V, W matrices of shape (4, 7), (4, 7), (4, 7) respectively.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # Result: [[M1+M4-M5+M7, M1-M2+M3+M7], [M2+M4-M6, M1-M3+M4+M6]]
    
    rank = 7
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)
    
    # Column indices in U: row = i*2+j where i=0,1; j=0,1
    # Column indices in V: row = j*2+k where j=0,1; k=0,1
    # Column indices in W: row = k*2+i where k=0,1; i=0,1
    
    # M1 = (a+d)(e+h): U has a,d in col0; V has e,h in col0; W contributes to all 4 outputs
    # U[0,0]=1 (a), U[1,0]=1 (d) [row 0=i*2+j: i=0,j=0; row 1=i=0,j=1]
    # Actually: U[i*2+j]: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V[j*2+k]: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W[k*2+i]: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1  # a
    U[1, 0] = 1  # d
    V[0, 0] = 1  # e
    V[3, 0] = 1  # h (j=1,k=1 -> row 3)
    W[0, 0] = 1  # output (k=0,i=0)
    W[1, 0] = 1  # output (k=0,i=1)
    W[2, 0] = 1  # output (k=1,i=0)
    W[3, 0] = 1  # output (k=1,i=1)
    
    # M2 = (c+d)e
    U[2, 1] = 1  # c (i=1,j=0)
    U[3, 1] = 1  # d (i=1,j=1)
    V[0, 1] = 1  # e (j=0,k=0)
    W[2, 1] = 1  # output (k=1,i=0)
    W[3, 1] = 1  # output (k=1,i=1)
    
    # M3 = a(f-h)
    U[0, 2] = 1  # a
    V[1, 2] = 1  # f (j=0,k=1)
    V[3, 2] = -1  # -h (j=1,k=1)
    W[1, 2] = 1  # output (k=0,i=1)
    W[3, 2] = 1  # output (k=1,i=1)
    
    # M4 = d(g-e)
    U[1, 3] = 1  # d (i=0,j=1)
    V[2, 3] = 1  # g (j=1,k=0)
    V[0, 3] = -1  # -e (j=0,k=0)
    W[0, 3] = 1  # output (k=0,i=0)
    W[2, 3] = 1  # output (k=1,i=0)
    
    # M5 = (a+b)h
    U[0, 4] = 1  # a
    U[1, 4] = 1  # b (i=0,j=1)
    V[3, 4] = 1  # h
    W[0, 4] = -1  # output (k=0,i=0)
    W[2, 4] = -1  # output (k=1,i=0)
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1  # c
    U[0, 5] = -1  # -a
    V[0, 5] = 1  # e
    V[1, 5] = 1  # f
    W[2, 5] = -1  # output (k=1,i=0)
    W[3, 5] = -1  # output (k=1,i=1)
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1  # b
    U[3, 6] = -1  # -d (i=1,j=1)
    V[2, 6] = 1  # g
    V[3, 6] = 1  # h
    W[0, 6] = 1  # output (k=0,i=0)
    W[1, 6] = 1  # output (k=0,i=1)
    
    return U, V, W


def _strassen_2x2x1():
    """
    Return the 4-term decomposition for 2x2 * 2x1.
    Returns U, V, W of shape (4, 4), (2, 4), (2, 4).
    U[i*2+j, c], V[j*2+0, c], W[0*2+i, c]
    """
    rank = 4
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((2, rank), dtype=float)
    W = np.zeros((2, rank), dtype=float)
    
    # For 2x2 * 2x1: result = [a*e + b*g, c*e + d*g]
    # Optimal rank 4 decomposition:
    # P1 = a*e, P2 = b*g, P3 = c*e, P4 = d*g
    # Or use: P1=(a+c)e, P2=(b+d)g, P3=a*g, P4=c*e ... let's use simple
    # Actually optimal is 4: just the 4 products
    
    # P1 = a*e
    U[0, 0] = 1  # a
    V[0, 0] = 1  # e
    W[0, 0] = 1  # output i=0
    
    # P2 = b*g  
    U[1, 1] = 1  # b
    V[1, 1] = 1  # g
    W[0, 1] = 1  # output i=0
    
    # P3 = c*e
    U[2, 2] = 1  # c
    V[0, 2] = 1  # e
    W[1, 2] = 1  # output i=1
    
    # P4 = d*g
    U[3, 3] = 1  # d
    V[1, 3] = 1  # g
    W[1, 3] = 1  # output i=1
    
    return U, V, W


def _construct_strassen_2x4x7():
    """
    Construct decomposition of 2x4 * 4x7 using Strassen blocks.
    
    Partition:
    - A is 2x4: split columns into 2+2 -> A = [A1, A2] where A1,A2 are 2x2
    - B is 4x7: split rows into 2+2+2+1 -> B = [B1; B2; B3; B4] where B1,B2,B3 are 2x7, B4 is 1x7
      But wait, we need the middle dimension to match. Let's think differently.
    
    Better partition:
    - A is 2x4: A = [A_left, A_right] where each is 2x2
    - B is 4x7: B = [B_top; B_bottom] where B_top is 2x7, B_bottom is 2x7
    - Result C = A*B = A_left * B_top + A_right * B_bottom (each 2x7)
    
    Now for each 2x2 * 2x7:
    - Split the 7 columns of B_top into 2+2+2+1
    - So 2x2 * 2x7 = (2x2*2x2) + (2x2*2x2) + (2x2*2x2) + (2x2*2x1)
    
    Total rank: 2 * (3*7 + 4) = 2 * 25 = 50
    
    Alternatively, we can try to share some structure.
    """
    n, m, p = 2, 4, 7
    
    # We'll build the full U, V, W matrices
    # Strategy: 
    # A (2x4) = [A1 | A2] where A1, A2 are 2x2
    # B (4x7) = [B1; B2] where B1, B2 are 2x7
    # C (2x7) = A1*B1 + A2*B2
    #
    # For each 2x2 * 2x7 product, split B into 2x2 + 2x2 + 2x2 + 2x1
    # Use Strassen for 2x2*2x2 (rank 7) and simple for 2x2*2x1 (rank 4)
    
    # Total rank = 2 * (3*7 + 4) = 50
    
    total_rank = 50
    U = np.zeros((n*m, total_rank), dtype=float)  # (8, 50)
    V = np.zeros((m*p, total_rank), dtype=float)  # (28, 50)
    W = np.zeros((p*n, total_rank), dtype=float)  # (14, 50)
    
    col_offset = 0
    
    # For A1 * B1 (2x2 * 2x7)
    # A1 rows: i=0,1; cols j=0,1 -> U rows 0,1,2,3
    # B1 rows: j=0,1; cols k=0..6 -> V rows 0..13 (j*p+k for j=0,1; k=0..6)
    #    j=0: k=0..6 -> V rows 0..6
    #    j=1: k=0..6 -> V rows 7..13
    # Output: 2x7 -> W rows k*n+i for k=0..6, i=0,1
    #    k=0: i=0,1 -> W rows 0,1
    #    k=1: i=0,1 -> W rows 2,3
    #    etc.
    
    for block_idx in range(2):  # 0 for A1*B1, 1 for A2*B2
        # A block: 2x2
        # If block_idx=0: A1 uses U rows 0-3 (i=0,1; j=0,1)
        # If block_idx=1: A2 uses U rows 4-7 (i=0,1; j=2,3)
        a_row_base = block_idx * 4  # 0 or 4
        
        # B block: 2x7
        # If block_idx=0: B1 uses V rows 0-13 (j=0,1; k=0..6)
        # If block_idx=1: B2 uses V rows 14-27 (j=2,3; k=0..6)
        #    j=2: k=0..6 -> V rows 14..20
        #    j=3: k=0..6 -> V rows 21..27
        b_row_base = block_idx * 14  # 0 or 14
        
        # For each 2x2*2x2 sub-block (3 of them) and 2x2*2x1 (1 of them)
        for sub in range(4):
            if sub < 3:
                # 2x2 * 2x2: columns k = 2*sub .. 2*sub+1
                k_base = 2 * sub
                sU, sV, sW = _strassen_2x2x2()
                # Map sU (4x7) to U[a_row_base:a_row_base+4, col_offset:col_offset+7]
                U[a_row_base:a_row_base+4, col_offset:col_offset+7] = sU
                # Map sV (4x7) to V[b_row_base + j*7 + k, ...]
                # sV row j*2+k where j=0,1; k=0,1
                # V row = b_row_base + j*7 + (k_base + k)
                for j in range(2):
                    for k in range(2):
                        v_idx = j * 2 + k
                        V[b_row_base + j * 7 + k_base + k, col_offset:col_offset+7] = sV[v_idx]
                # Map sW (4x7) to W[k*n+i, ...]
                # sW row k*2+i where k=0,1; i=0,1
                # W row = (k_base + k)*2 + i
                for k in range(2):
                    for i in range(2):
                        w_idx = k * 2 + i
                        W[(k_base + k) * 2 + i, col_offset:col_offset+7] = sW[w_idx]
                col_offset += 7
            else:
                # 2x2 * 2x1: column k = 6
                k_base = 6
                sU, sV, sW = _strassen_2x2x1()
                # sU (4x4) -> U[a_row_base:a_row_base+4, col_offset:col_offset+4]
                U[a_row_base:a_row_base+4, col_offset:col_offset+4] = sU
                # sV (2x4) -> V[b_row_base + j*7 + k_base, ...]
                for j in range(2):
                    V[b_row_base + j * 7 + k_base, col_offset:col_offset+4] = sV[j]
                # sW (2x4) -> W[k_base*2 + i, ...]
                for i in range(2):
                    W[k_base * 2 + i, col_offset:col_offset+4] = sW[i]
                col_offset += 4
    
    return U, V, W, total_rank


def _construct_np(n, m, p):
    """Construct rank n*p decomposition."""
    rank = n * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    for i in range(n):
        for k in range(p):
            c = k * n + i
            for j in range(m):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
            W[k * n + i, c] = 1
    return U, V, W


def _try_alternative_partition():
    """
    Try different block partitionings to find lower rank.
    
    Idea: Instead of 2x4*4x7 = (2x2*2x7) + (2x2*2x7),
    try 2x4*4x7 = (2x1*1x7) + (2x1*1x7) + (2x1*1x7) + (2x1*1x7)
    Each 2x1*1x7 has rank... well 2x1*1x7 is just outer product, rank 1? No.
    
    Actually 2x1 * 1x7: A is 2x1, B is 1x7. Result is 2x7.
    C[i,k] = A[i,0] * B[0,k]
    This is rank 1 tensor! U[i,0] = A[i,0], V[0,k] = B[0,k], W[k,i] = 1.
    So rank 1 per block. Total: 4 blocks * 1 = 4? That can't be right.
    
    Wait, no. The issue is that A and B are arbitrary, so we can't assume they have
    special structure. The tensor decomposition must work for ALL A and B.
    
    Let me reconsider. The matrix multiplication tensor for n×m × m×p has
    entries T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k].
    
    For the full 2x4x4x7 tensor, the rank is what we're looking for.
    """
    pass


def _verify_strassen(U, V, W, n, m, p):
    """Verify the Strassen-based decomposition."""
    return _verify_np(U, V, W, n, m, p)


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()

    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    # 1) Try Strassen-based decomposition (rank 50)
    U_s, V_s, W_s, rank_s = _construct_strassen_2x4x7()
    if _verify_np(U_s, V_s, W_s, n, m, p):
        best_U, best_V, best_W = U_s, V_s, W_s
        best_rank = rank_s
        print(f"Found Strassen rank {rank_s} decomposition")
    else:
        print(f"Strassen rank {rank_s} construction failed verification")
    
    # 2) Try rank n*p = 14
    U14, V14, W14 = _construct_np(n, m, p)
    if _verify_np(U14, V14, W14, n, m, p):
        if 14 < best_rank:
            best_U, best_V, best_W = U14, V14, W14
            best_rank = 14
            print(f"Found rank {14} decomposition")
    
    # 3) Try to improve Strassen by sharing structure between the two 2x2*2x7 blocks
    # Key insight: A1*B1 + A2*B2 where A1,A2 are 2x2 and B1,B2 are 2x7
    # Can we combine some Strassen terms?
    
    # Try a different approach: use the fact that we can choose any linear combinations
    # of the input elements for the intermediate products
    
    # For 2x4 * 4x7, think of it as computing 14 output entries C[i,k] for i=0,1; k=0..6
    # Each C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # The minimal rank is at least max(n*p, m*max(n,p)) = max(14, 4*7) = 28? No.
    # Lower bound: rank >= max(n*p, m*max(n,p), ... ) 
    # Actually rank >= max(n*p, m*max(n,p), p*max(n,m)) = max(14, 28, 14) = 28? 
    # No, that's not right either. The standard lower bound is rank >= max(n*p, m*max(n,p), p*max(n,m))
    # For 2,4,7: max(14, 4*7, 7*4) = max(14, 28, 28) = 28
    
    # So rank >= 28. Our Strassen gives 50. Can we do better?
    
    # Try rank 28-40 with structured approach
    # Use the tensor flattening: fix one index, get a matrix, find its rank
    
    # For fixed i: sum_j A[i,j] B[j,k] for all k -> this is a linear map from B to C[i,:]
    # The tensor can be viewed as a 3-tensor T[i,j,k]
    
    # Let's try a systematic search for rank r from 28 to 50
    # using random W and solving for U, V
    
    import random
    
    time_remaining = DEADLINE - time.time() - 10
    if time_remaining > 5 and best_rank > 30:
        # Try ranks from min(best_rank-1, 40) down to 28
        start_rank = min(best_rank - 1, 45)
        for target_rank in range(start_rank, 27, -1):
            if time.time() > DEADLINE - 10:
                break
            
            # Try multiple random W structures
            found = False
            max_attempts = 200
            for attempt in range(max_attempts):
                if time.time() > DEADLINE - 10:
                    break
                
                # Generate W: each column assigned to one or more (i,k) pairs
                W = np.zeros((p * n, target_rank), dtype=float)
                for c in range(target_rank):
                    # 70% single, 30% double
                    if random.random() < 0.7:
                        k = random.randint(0, p - 1)
                        i = random.randint(0, n - 1)
                        W[k * n + i, c] = 1.0
                    else:
                        k1 = random.randint(0, p - 1)
                        i1 = random.randint(0, n - 1)
                        k2 = random.randint(0, p - 1)
                        i2 = random.randint(0, n - 1)
                        if (i1, k1) != (i2, k2):
                            W[k1 * n + i1, c] = 1.0
                            W[k2 * n + i2, c] = 1.0
                        else:
                            W[k1 * n + i1, c] = 1.0
                
                # Check coverage
                valid = True
                for i in range(n):
                    for k in range(p):
                        if W[k * n + i].sum() == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Solve for U, V using first-column strategy
                U = np.zeros((n * m, target_rank), dtype=float)
                V = np.zeros((m * p, target_rank), dtype=float)
                
                solved = True
                for i in range(n):
                    for k in range(p):
                        S = np.where(W[k * n + i] != 0)[0]
                        if len(S) == 0:
                            solved = False
                            break
                        c_first = S[0]
                        for j in range(m):
                            U[i * m + j, c_first] = 1.0
                            V[j * p + k, c_first] = 1.0
                        for c in S[1:]:
                            for j in range(m):
                                U[i * m + j, c] = 0.0
                                V[j * p + k, c] = 0.0
                    if not solved:
                        break
                
                if not solved:
                    continue
                
                if _verify_np(U, V, W, n, m, p):
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    print(f"Found rank {target_rank} decomposition")
                    found = True
                    break
            
            if found:
                break
    
    # Final fallback: schoolbook
    if best_U is None:
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank
        print(f"Using schoolbook rank {rank}")
    
    # Convert to list format
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    row_list.append(int(round(val)))
                else:
                    row_list.append(float(val))
            result.append(row_list)
        return result
    
    return {
        'U': to_list(best_U),
        'V': to_list(best_V),
        'W': to_list(best_W)
    }
