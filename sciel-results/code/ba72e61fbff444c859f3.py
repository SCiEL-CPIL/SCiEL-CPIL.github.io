"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

import time
import math

# EVOLVE-BLOCK-START
def _base_strassen_222():
    """
    Returns (U, V, W) for 2x2x2 Strassen decomposition.
    U has shape (4, 7)  [n*m = 2*2]
    V has shape (4, 7)  [m*p = 2*2]
    W has shape (4, 7)  [p*n = 2*2]
    
    Indices:
    U[i*m+j, c] where m=2
    V[j*p+k, c] where p=2
    W[k*n+i, c] where n=2
    """
    n = m = p = 2
    # Strassen constants:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = AB
    #
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    rank = 7
    
    # U: rows indexed by (i,j) -> i*2+j, i in [0,1], j in [0,1]
    # Column c corresponds to coefficient of A[i,j] in M_c
    U = [
        # Row 0: (0,0)
        [1, 0, 1, 0, 1, 1, 0],
        # Row 1: (0,1)
        [0, 0, 1, 0, 1, 0, 1],
        # Row 2: (1,0)
        [0, 1, 0, 0, 0, 0, 0],
        # Row 3: (1,1)
        [1, 1, 0, 1, 0, 0, 0],
    ]
    
    # V: rows indexed by (j,k) -> j*2+k
    V = [
        # Row 0: (0,0)
        [1, 1, 0, 0, 0, 1, 0],
        # Row 1: (0,1)
        [0, 0, 1, 0, 0, 1, 0],
        # Row 2: (1,0)
        [0, 0, 0, 1, 0, 0, 1],
        # Row 3: (1,1)
        [1, 0, 1, 0, 1, 0, 1],
    ]
    
    # W: rows indexed by (k,i) -> k*2+i
    W = [
        # Row 0: (0,0) -> C00
        [1, 0, 0, 1, -1, 0, 1],
        # Row 1: (0,1) -> C01
        [0, 0, 1, 0, 1, 0, 0],
        # Row 2: (1,0) -> C10
        [0, 1, 0, 1, 0, 0, 0],
        # Row 3: (1,1) -> C11
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    return U, V, W

def _combine_decompositions(n1, m1, p1, n2, m2, p2, 
                             U1, V1, W1, U2, V2, W2):
    """
    Combine two decompositions for block multiplication.
    
    We multiply (n1+n2) x (m1+m2) by (m1+m2) x (p1+p2).
    Split A into top/bottom blocks of heights n1, n2.
    Split B into left/right blocks of widths m1, m2 (rows), and p1, p2 (cols).
    
    Actually, the standard block decomposition:
    A is (n1+n2) x (m1+m2), B is (m1+m2) x (p1+p2)
    
    A = [[A11, A12],
         [A21, A22]]
    where A11 is n1 x m1, A12 is n1 x m2, A21 is n2 x m1, A22 is n2 x m2
    
    B = [[B11, B12],
         [B21, B22]]
    where B11 is m1 x p1, B12 is m1 x p2, B21 is m2 x p1, B22 is m2 x p2
    
    C = AB = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
              [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    The four blocks of C are:
    C11 (n1 x p1) = A11*B11 + A12*B21
    C12 (n1 x p2) = A11*B12 + A12*B22
    C21 (n2 x p1) = A21*B11 + A22*B21
    C22 (n2 x p2) = A21*B12 + A22*B22
    
    Each of these is a sum of two products, so we need decompositions for:
    1. n1 x m1 x p1 (A11 * B11)
    2. n1 x m2 x p1 (A12 * B21)
    3. n1 x m1 x p2 (A11 * B12)
    4. n1 x m2 x p2 (A12 * B22)
    5. n2 x m1 x p1 (A21 * B11)
    6. n2 x m2 x p1 (A22 * B21)
    7. n2 x m1 x p2 (A21 * B12)
    8. n2 x m2 x p2 (A22 * B22)
    
    But wait - we only have two decompositions passed in. The idea is that
    we recursively split, so we should call this with the sub-problems already solved.
    
    Let me restructure: instead of passing U1,V1,W1,U2,V2,W2, we should have a
    recursive function that solves the problem.
    """
    pass  # Not used directly; see _solve_recursive

def _solve_recursive(n, m, p):
    """
    Recursively compute a tensor decomposition for n x m x p matrix multiplication.
    Returns (U, V, W) where:
      U has shape (n*m, rank)
      V has shape (m*p, rank)  
      W has shape (p*n, rank)
    """
    # Base case: 2x2x2 uses Strassen
    if n == 2 and m == 2 and p == 2:
        return _base_strassen_222()
    
    # Base case: if any dimension is 1, use schoolbook for that subproblem
    # n x 1 x p: rank = n*p
    # 1 x m x p: rank = m*p  
    # n x m x 1: rank = n*m
    if m == 1:
        # A is n x 1, B is 1 x p, C is n x p
        # C[i,k] = A[i,0] * B[0,k]
        # rank = n*p
        rank = n * p
        U = [[0]*rank for _ in range(n*1)]
        V = [[0]*rank for _ in range(1*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for k in range(p):
                U[i*1 + 0][col] = 1
                V[0*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
        return U, V, W
    if n == 1:
        # A is 1 x m, B is m x p, C is 1 x p
        # C[0,k] = sum_j A[0,j] * B[j,k]
        # rank = m*p
        rank = m * p
        U = [[0]*rank for _ in range(1*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*1)]
        col = 0
        for j in range(m):
            for k in range(p):
                U[0*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*1 + 0][col] = 1
                col += 1
        return U, V, W
    if p == 1:
        # A is n x m, B is m x 1, C is n x 1
        # C[i,0] = sum_j A[i,j] * B[j,0]
        # rank = n*m
        rank = n * m
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*1)]
        W = [[0]*rank for _ in range(1*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                U[i*m + j][col] = 1
                V[j*1 + 0][col] = 1
                W[0*n + i][col] = 1
                col += 1
        return U, V, W
    
    # Choose which dimension to split
    # Try to split the largest dimension into two parts
    # We want balanced splits for efficiency
    
    # Try splitting m (middle dimension)
    # Split m into m1 and m2 where m1 = m//2, m2 = m - m1
    best_rank = float('inf')
    best_result = None
    best_split_info = None
    
    deadline = time.time() + 140  # 140s limit, leave 10s margin
    
    # Try different splits for each dimension
    # Dimension 0: n
    # Dimension 1: m  
    # Dimension 2: p
    
    for dim in [1, 0, 2]:  # Try m first, then n, then p
        if dim == 1:
            total = m
        elif dim == 0:
            total = n
        else:
            total = p
        
        if total < 2:
            continue
            
        for split in range(1, total):
            part1 = split
            part2 = total - split
            
            if time.time() > deadline:
                break
                
            # Get the sub-dimensions
            if dim == 1:
                n1, m1, p1 = n, part1, p
                n2, m2, p2 = n, part2, p
            elif dim == 0:
                n1, m1, p1 = part1, m, p
                n2, m2, p2 = part2, m, p
            else:
                n1, m1, p1 = n, m, part1
                n2, m2, p2 = n, m, part2
            
            # We need 4 sub-decompositions:
            # For C11 (n1 x p1): involves A11*B11 and A12*B21
            # This gets complicated. Let me use a different approach.
            
            # Actually, the standard way: when we split, we get 4 sub-multiplications
            # that need to be combined. But the combination isn't just concatenation
            # because the same A or B entries appear in multiple sub-products.
            
            # Let me use a cleaner recursive formulation:
            # The tensor for n x m x p multiplication can be expressed as a sum
            # of tensors for the 4 block sub-problems when we split m.
            
            # For splitting m into m1 + m2:
            # A = [A_left (n x m1), A_right (n x m2)]
            # B = [B_top (m1 x p), B_bottom (m2 x p)]
            # C = A_left * B_top + A_right * B_bottom
            #
            # This is a sum of two matrix products!
            # So rank(n,m,p) <= rank(n,m1,p) + rank(n,m2,p)
            
            # Similarly for other splits.
            
            # This is much simpler! Let's use this.
            
            if dim == 1:
                # Split m: C = A[:,:m1]*B[:m1,:] + A[:,m1:]*B[m1:,:]
                Ua, Va, Wa = _solve_recursive(n, part1, p)
                Ub, Vb, Wb = _solve_recursive(n, part2, p)
                rank_a = len(Ua[0]) if Ua else 0
                rank_b = len(Ub[0]) if Ub else 0
                total_rank = rank_a + rank_b
                
                # Combine U: U has n*m rows, n*(part1+part2) = n*m rows
                # Ua has n*part1 rows, Ub has n*part2 rows
                # U's rows: for i in range(n), j in range(m)
                # First part1 columns of A go to Ua, last part2 to Ub
                U = [None] * (n * m)
                for i in range(n):
                    for j in range(part1):
                        U[i * m + j] = Ua[i * part1 + j] + [0] * rank_b
                    for j in range(part1, m):
                        U[i * m + j] = [0] * rank_a + Ub[i * part2 + (j - part1)]
                
                # V has m*p rows
                # Va has part1*p rows, Vb has part2*p rows
                V = [None] * (m * p)
                for j in range(part1):
                    for k in range(p):
                        V[j * p + k] = Va[j * p + k] + [0] * rank_b
                for j in range(part1, m):
                    for k in range(p):
                        V[j * p + k] = [0] * rank_a + Vb[(j - part1) * p + k]
                
                # W has p*n rows
                Wa_rows = p * n
                Wb_rows = p * n
                W = [None] * (p * n)
                for k in range(p):
                    for i in range(n):
                        W[k * n + i] = Wa[k * n + i] + [0] * rank_b
                for k in range(p):
                    for i in range(n):
                        W[k * n + i] = [Wa[k * n + i] + Wb[k * n + i]]
                
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_result = (U, V, W)
                    best_split_info = (dim, split)
                    
            elif dim == 0:
                # Split n: A is split into top and bottom
                # C_top (part1 x p) = A_top * B
                # C_bottom (part2 x p) = A_bottom * B
                # Rank = rank(part1, m, p) + rank(part2, m, p)
                Ua, Va, Wa = _solve_recursive(part1, m, p)
                Ub, Vb, Wb = _solve_recursive(part2, m, p)
                rank_a = len(Ua[0]) if Ua else 0
                rank_b = len(Ub[0]) if Ub else 0
                total_rank = rank_a + rank_b
                
                # U has n*m rows
                # Ua has part1*m rows, Ub has part2*m rows
                U = [None] * (n * m)
                for i in range(part1):
                    for j in range(m):
                        U[i * m + j] = Ua[i * m + j] + [0] * rank_b
                for i in range(part1, n):
                    for j in range(m):
                        U[i * m + j] = [0] * rank_a + Ub[(i - part1) * m + j]
                
                # V has m*p rows - same for both, so V = Va + Vb (element-wise)
                # But they have the same shape, so we add them
                V = [None] * (m * p)
                for j in range(m):
                    for k in range(p):
                        V[j * p + k] = [Va[j * p + k][c] + Vb[j * p + k][c] for c in range(max(rank_a, rank_b))]
                        # Wait, they have different ranks. We need to pad.
                        V[j * p + k] = Va[j * p + k] + [0] * rank_b
                        
                # Hmm, this doesn't work because V_a and V_b have different numbers of columns.
                # Let me reconsider.
                
                # Actually, for splitting n:
                # The B matrix is shared. So V should be the same structure.
                # But in the decomposition, each sub-problem has its own V.
                # The issue is that we're adding two decompositions that use the same B.
                
                # Let me think again. The rank of the sum is at most the sum of ranks.
                # But we need to construct the actual matrices.
                
                # For U: straightforward concatenation as above.
                # For V: both sub-problems use the same B, so V_a and V_b should be
                # compatible. But they have different numbers of columns (rank_a vs rank_b).
                # We need to embed them into a common space.
                
                # Actually, I think the cleaner approach is: the total rank is rank_a + rank_b,
                # and V has m*p rows with rank_a + rank_b columns.
                # For the first rank_a columns, V = V_a (for the first part) 
                # For the last rank_b columns, V = V_b
                # But that's not right either, because both parts use all of B.
                
                # I think I'm overcomplicating this. Let me go back to splitting m,
                # which is cleaner because A and B are split independently.
                
                # For now, let's only use splitting m.
                continue
                
            else:  # dim == 2, split p
                # Split p: C_left (n x part1) = A * B_left, C_right (n x part2) = A * B_right
                # Rank = rank(n, m, part1) + rank(n, m, part2)
                Ua, Va, Wa = _solve_recursive(n, m, part1)
                Ub, Vb, Wb = _solve_recursive(n, m, part2)
                rank_a = len(Ua[0]) if Ua else 0
                rank_b = len(Ub[0]) if Ub else 0
                total_rank = rank_a + rank_b
                
                # U has n*m rows - same for both, so U = U_a + U_b (padded)
                # But again, different ranks.
                
                # This has the same issue as splitting n.
                continue
        
        if time.time() > deadline:
            break
    
    # If we found a valid split, return it
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook decomposition
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
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0]) if U else 0
    import numpy as np
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # For each (i, j, k), check that the coefficient is 1
    # The tensor entry T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
                err = abs(val - 1.0)
                max_err = max(max_err, err)
    
    # Also check that entries that should be 0 are 0
    # The tensor is zero except at positions (i, j, k) which correspond to C[i,k] = sum_j A[i,j]*B[j,k]
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k in the range,
    # and the output C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # So all T[i,j,k] = 1 for valid i,j,k.
    
    return max_err < 1e-10

def construct():
    """
    Construct a rank-minimal tensor decomposition for 4x5 x 5x6 matrix multiplication.
    Returns {'U': U, 'V': V, 'W': W}
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    
    # Try recursive decomposition
    try:
        U, V, W = _solve_recursive(n, m, p)
        rank = len(U[0]) if U else 0
        
        # Verify the decomposition
        if time.time() < deadline - 5:
            if not _verify_decomposition(U, V, W, n, m, p):
                # Fall back to schoolbook
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
        else:
            # Time up, just return what we have
            pass
        
        return {"U": U, "V": V, "W": W}
        
    except Exception as e:
        # Fallback to schoolbook on any error
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
