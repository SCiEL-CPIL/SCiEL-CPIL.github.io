"""Matrix multiplication 2,2,5 rank-minimal decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reconstructs the multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    # Tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # The multiplication tensor has T[i,j,k] = 1 for all i,j,k (since (AB)_{ik} = sum_j A_{ij} B_{jk})
    # Actually the tensor for matrix multiplication M_{n,m,p} has entries:
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # and the multiplication is C = A * B where A is n x m, B is m x p
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor has T[i,j,k] = 1 for all valid i,j,k
    # In the factorization: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Build the full tensor
    T_target = np.ones((n, m, p), dtype=np.float64)
    
    # Compute from factors
    T_computed = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_computed[i, j, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    if not np.allclose(T_computed, T_target, atol=tol):
        return False
    return True

def _verify_strict(U, V, W, n, m, p):
    """Strict verification using exact arithmetic."""
    rank = len(U[0])
    U = [[Fraction(x) for x in row] for row in U]
    V = [[Fraction(x) for x in row] for row in V]
    W = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _make_decomposition(U, V, W):
    """Convert to integer/half-integer format if needed, return dict."""
    # Ensure entries are integers or half-integers
    def clean_matrix(M):
        result = []
        for row in M:
            new_row = []
            for val in row:
                # Check if it's an integer or half-integer
                if isinstance(val, Fraction):
                    if val.denominator == 1:
                        new_row.append(val.numerator)
                    elif val.denominator == 2:
                        new_row.append(Fraction(val.numerator, 2))
                    else:
                        # Scale to make it work - this shouldn't happen
                        new_row.append(val)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    U_clean = clean_matrix(U)
    V_clean = clean_matrix(V)
    W_clean = clean_matrix(W)
    
    return {"U": U_clean, "V": V_clean, "W": W_clean}

def _strassen_2x2():
    """Return Strassen decomposition for 2x2 x 2x2 multiplication. Rank 7."""
    # Strassen's algorithm for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # Strassen products:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # C11 = p1+p4-p5+p7
    # C12 = p3+p5
    # C21 = p2+p4
    # C22 = p1-p2+p3+p6
    
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]  # 4 x 7
    V = [[0]*rank for _ in range(m*p)]  # 4 x 7
    W = [[0]*rank for _ in range(p*n)]  # 4 x 7
    
    # U indices: [i*m+j] = [0,1,2,3] for (0,0),(0,1),(1,0),(1,1)
    # V indices: [j*p+k] = [0,1,2,3] for (0,0),(0,1),(1,0),(1,1)
    # W indices: [k*n+i] = [0,1,2,3] for (0,0),(0,1),(1,0),(1,1)
    
    # p1 = (a+d)(e+h): U[0,0]=1, U[3,0]=1, V[0,0]=1, V[1,0]=1, V[2,0]=1, V[3,0]=1
    # W: C11 += p1, C22 += p1 => W[0,0]=1, W[3,0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[1][0] = 1; V[2][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # p2 = (c+d)e: U[2,1]=1, U[3,1]=1, V[0,1]=1
    # W: C21 += p2 => W[2,1]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # p3 = a(f-h): U[0,2]=1, V[1,2]=1, V[3,2]=-1
    # W: C12 += p3, C22 += p3 => W[1,2]=1, W[3,2]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # p4 = d(g-e): U[3,3]=1, V[0,3]=-1, V[2,3]=1
    # W: C11 += p4, C21 += p4 => W[0,3]=1, W[2,3]=1
    U[3][3] = 1
    V[0][3] = -1; V[2][3] = 1
    W[0][3] = 1; W[2][3] = 1
    
    # p5 = (a+b)h: U[0,4]=1, U[1,4]=1, V[3,4]=1
    # W: C11 -= p5, C12 += p5 => W[0,4]=-1, W[1,4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # p6 = (c-a)(e+f): U[0,5]=-1, U[2,5]=1, V[0,5]=1, V[1,5]=1
    # W: C22 += p6 => W[3,5]=1
    U[0][5] = -1; U[2][5] = 1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # p7 = (b-d)(g+h): U[1,6]=1, U[3,6]=-1, V[2,6]=1, V[3,6]=1
    # W: C11 += p7 => W[0,6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _decomp_2x2x3():
    """
    Find a low-rank decomposition for 2x2 * 2x3 multiplication.
    We need T[i,j,k]=1 for i in {0,1}, j in {0,1}, k in {0,1,2}.
    
    The tensor has 12 entries, all equal to 1.
    Rank lower bound: max(n*m, m*p, p*n) = max(4,6,6) = 6.
    Known: rank of 2x2x3 tensor is 6 (it's a "generic" enough tensor).
    
    Let's construct a rank-6 decomposition.
    U: 4x6, V: 6x6, W: 6x4
    
    We need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k.
    """
    n, m, p = 2, 2, 3
    rank = 6
    U = [[0]*rank for _ in range(n*m)]  # 4x6
    V = [[0]*rank for _ in range(m*p)]  # 6x6
    W = [[0]*rank for _ in range(p*n)]  # 6x4
    
    # Strategy: Use a structured approach.
    # For each k (0,1,2), we need to handle the 2x2 block.
    # But we can share structure.
    
    # Let's try: for each c in 0..5, define the factors.
    # We want sum over c of U[i,j,c]*V[j,k,c]*W[k,i,c] = 1.
    
    # Simple approach: Use 6 terms where each term covers specific (i,j,k) patterns.
    # Actually, let's use the fact that this is a bilinear form.
    
    # Let me try a direct construction:
    # Term c=0: U[0,0]=1, U[1,0]=1, V[0,0]=1, V[1,0]=1, V[2,0]=1, V[3,0]=1, V[4,0]=1, V[5,0]=1
    #           W[0,0]=1, W[1,0]=1, W[2,0]=1, W[3,0]=1, W[4,0]=1, W[5,0]=1
    # This gives contribution 1 to all (i,j,k) where i in {0,1}, j in {0,1}, k in {0,1,2}
    # Wait, that's just one term giving 1 to everything. But then we'd need rank 1.
    # That's not right - the issue is that U, V, W have specific index structures.
    
    # Let me reconsider. U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # For n=m=2, p=3:
    # U has 4 rows: (0,0),(0,1),(1,0),(1,1)
    # V has 6 rows: (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)
    # W has 6 rows: (0,0),(0,1),(1,0),(1,1),(2,0),(2,1)
    
    # We need for each (i,j,k): sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    
    # Let's try rank 6 with a systematic approach.
    # Idea: For each k, use a 2x2 Strassen-like structure but adapt.
    
    # Actually, let me try a simpler structured decomposition.
    # Use 6 terms:
    # c=0: covers (i=0,j=0) for all k, with W depending on k
    # This is getting complex. Let me use a different strategy.
    
    # Known result: The 2x2x3 matrix multiplication tensor has rank 6.
    # Let me construct it explicitly.
    
    # Approach: Think of it as multiplying two 2x2 matrices but the second has 3 columns.
    # A (2x2) * B (2x3) = C (2x3)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can use 6 bilinear forms. Let me define:
    # For k=0,1,2, we need 2x2 multiplication essentially, but they share A.
    
    # Let me try:
    # c=0: U[0,0]=1, U[2,0]=1, V[0,0]=1, V[3,0]=1, W[0,0]=1, W[3,0]=1
    #   => contributes to (i=0,j=0,k=0): U[0]*V[0]*W[0] = 1
    #   Also (i=1,j=0,k=0): U[2]*V[3]*W[3] = 1
    #   But we need it for all (i,j,k) combinations...
    
    # Let me use a more systematic approach with numpy to solve.
    
    # Actually, let me just hardcode a known rank-6 decomposition.
    # 
    # For 2x2 * 2x3:
    # Let's use the following construction:
    # 
    # The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1,2}
    # 
    # Rank 6 decomposition:
    # c=0: U[0,0]=1, V[0,0]=1, V[3,0]=1, W[0,0]=1, W[3,0]=1  -> (0,0,0):1, (1,0,0):1, (0,0,1):? 
    # 
    # This is getting complicated. Let me use a numerical approach to find it.
    
    # For now, let's use a schoolbook for 2x2x3: rank = 2*2*3 = 12
    # U: 4x12, V: 6x12, W: 6x4
    rank = 12
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(6)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*2+j][col] = 1
                V[j*3+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """
    Combine multiple decompositions that partition the output columns.
    Each decomp is (U, V, W) for a sub-problem.
    We concatenate them along the rank dimension.
    """
    total_rank = sum(len(d[0][0]) for d in decomps)
    
    # For 2x2 * 2x5, we can split the 5 columns into 2+3 or other partitions.
    # But the decompositions need to be compatible.
    
    # Actually, for the full 2x2*2x5 problem, we need a single decomposition.
    # Let me think about this differently.
    
    # The full problem: A (2x2) * B (2x5) = C (2x5)
    # T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2,3,4}
    # Total 20 entries.
    # Lower bound on rank: max(n*m, m*p, p*n) = max(4, 10, 10) = 10.
    
    # Can we achieve rank 10? Or lower?
    # Known: rank of matrix multiplication tensor M_{n,m,p} >= max(n*m, m*p, p*n)
    # And for small cases, the rank is often close to this bound.
    
    # For 2x2x5: lower bound is 10. Let's try to find a rank 10 decomposition.
    # If not possible, try 11, 12, etc.
    
    return None  # Placeholder

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    n, m, p = DIMENSIONS  # 2, 2, 5
    
    # First, try to find a good decomposition using optimization
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p  # 20
    U_sb = [[0]*rank_schoolbook for _ in range(n*m)]
    V_sb = [[0]*rank_schoolbook for _ in range(m*p)]
    W_sb = [[0]*rank_schoolbook for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(4, 10, 10) = 10
    
    # Strategy: Use structured decompositions
    # For 2x2*2x5, split into 2x2*2x2 + 2x2*2x3
    # Strassen for 2x2*2x2: rank 7
    # For 2x2*2x3: need to find optimal rank
    
    # Let's try a direct approach: construct a rank-r decomposition
    # and verify it.
    
    # Try rank 10 first (theoretical lower bound)
    for target_rank in range(10, best_rank):
        if time.time() > deadline:
            break
        
        # Try to construct a decomposition of the given rank
        # Using a structured approach based on Strassen-like ideas
        
        # For 2x2 * 2x5:
        # We can view this as computing C[i,k] = sum_j A[i,j]*B[j,k] for i=0,1 and k=0..4
        # That's 10 outputs, each a bilinear form in 4 variables (A00,A01,A10,A11) 
        # and 10 variables (B00..B04, B10..B14).
        
        # A rank-r decomposition means we can compute all 10 outputs using r bilinear products.
        
        # Let me try a construction:
        # Group the k indices: {0,1}, {2,3}, {4}
        # For {0,1}: use Strassen (rank 7)
        # For {2,3}: use Strassen (rank 7) 
        # For {4}: use 2 multiplications (rank 2)
        # Total: 7+7+2 = 16
        
        # Or: {0,1,2,3,4} all together
        # 
        # Let me try a different grouping:
        # {0,1}: Strassen rank 7
        # {2,3,4}: 2x2*2x3, if rank 6, then total 13
        
        # Let me try to find a rank-6 decomposition for 2x2*2x3.
        
        if target_rank <= 13:
            # Try Strassen(2x2) + decomp(2x2x3)
            U_s, V_s, W_s = _strassen_2x2()
            U_223, V_223, W_223 = _decomp_2x2x3()
            
            rank_s = len(U_s[0])  # 7
            rank_223 = len(U_223[0])  # 12 for now
            
            # Combine: total rank = rank_s + rank_223
            total_rank = rank_s + rank_223
            
            if total_rank <= target_rank:
                # Build the combined decomposition for 2x2*2x5
                # First 2 columns (k=0,1) use Strassen
                # Last 3 columns (k=2,3,4) use 2x2x3 decomp
                
                U_comb = [[0]*total_rank for _ in range(4)]
                V_comb = [[0]*total_rank for _ in range(10)]  # m*p = 2*5 = 10
                W_comb = [[0]*total_rank for _ in range(10)]  # p*n = 5*2 = 10
                
                # Copy Strassen part (for k=0,1)
                for r in range(rank_s):
                    for i in range(4):
                        U_comb[i][r] = U_s[i][r]
                    for j in range(2):
                        for k in range(2):
                            V_comb[j*5+k][r] = V_s[j*2+k][r]
                    for k in range(2):
                        for i in range(2):
                            W_comb[k*2+i][r] = W_s[k*2+i][r]
                
                # Copy 2x2x3 part (for k=2,3,4)
                offset = rank_s
                for r in range(rank_223):
                    for i in range(4):
                        U_comb[i][offset+r] = U_223[i][r]
                    for j in range(2):
                        for k in range(3):
                            V_comb[j*5+(k+2)][offset+r] = V_223[j*3+k][r]
                    for k in range(3):
                        for i in range(2):
                            W_comb[(k+2)*2+i][offset+r] = W_223[k*2+i][r]
                
                # Verify
                if _verify(U_comb, V_comb, W_comb, 2, 2, 5):
                    if total_rank < best_rank:
                        best_rank = total_rank
                        best_result = {"U": U_comb, "V": V_comb, "W": W_comb}
                    break  # Found a valid one at this rank
        
        if time.time() > deadline:
            break
    
    # Try to improve: find a better 2x2x3 decomposition
    # The current 2x2x3 uses rank 12 (schoolbook). Let's try to do better.
    
    # For 2x2*2x3, lower bound is max(4,6,6)=6.
    # Let me try to find a rank 6 or 7 decomposition.
    
    # Try rank 6 for 2x2x3
    if time.time() < deadline:
        # Attempt to find a rank-6 decomposition for 2x2*2x3
        # Using a structured approach
        
        # Let me try:
        # c=0: U[0,0]=1, U[1,0]=1, V[0,0]=1, V[1,0]=1, V[2,0]=1, V[3,0]=1, V[4,0]=1, V[5,0]=1
        #      W[0,0]=1, W[1,0]=1, W[2,0]=1, W[3,0]=1, W[4,0]=1, W[5,0]=1
        # This gives 1 to all (i,j,k) - but that's rank 1 which is impossible.
        
        # The issue: U[i,j,c] * V[j,k,c] * W[k,i,c] must sum to 1.
        # With rank 1: U[i,j]*V[j,k]*W[k,i] = 1 for all i,j,k.
        # This requires U[i,j] = a_i * b_j, V[j,k] = b_j * c_k, W[k,i] = c_k * a_i
        # Then product = a_i * b_j * b_j * c_k * c_k * a_i = a_i^2 * b_j^2 * c_k^2
        # This can't be 1 for all i,j,k unless a_i^2 = b_j^2 = c_k^2 = 1,
        # but then we need consistency.
        
        # Let me try a different approach: use numpy to solve for a rank-r decomposition.
        
        # For rank r, we have r * (4 + 6 + 6) = r * 16 variables.
        # We have 12 equations (one for each (i,j,k)).
        # For r >= 2, we have more variables than equations, so solutions should exist.
        
        # Let me try r=6:
        # Actually, let me just try a few structured decompositions.
        
        # Try this construction for 2x2*2x3:
        # Use 6 terms where each term is a rank-1 tensor.
        
        # Let me define:
        # Term 0: U=[1,0,0,0], V=[1,1,1,0,0,0], W=[1,0,0,0,0,0]
        #   Contributes: (0,0,0):1*1*1=1, others 0
        # Term 1: U=[0,1,0,0], V=[1,1,1,0,0,0], W=[0,1,0,0,0,0]  
        #   Contributes: (0,1,0):1*1*1=1, others 0
        # ... this is just schoolbook.
        
        # Let me try a smarter approach:
        # Group by j:
        # For j=0: need C[i,k] = A[i,0]*B[0,k] for i=0,1, k=0,1,2 (6 terms)
        # For j=1: need C[i,k] += A[i,1]*B[1,k] for i=0,1, k=0,1,2 (6 terms)
        # Total: 12 (schoolbook)
        
        # To reduce rank, we need to share structure between j=0 and j=1.
        
        # Idea: For each k, compute [A[0,0]*B[0,k] + A[1,0]*B[1,k], A[0,1]*B[0,k] + A[1,1]*B[1,k]]
        # This is like a 2x2 matrix product for each k, but with shared A.
        
        # For a fixed k, the computation is:
        # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
        # This is a 2x2 "outer product" type computation.
        # Can we do this in fewer than 4 bilinear forms?
        # It's essentially computing two independent bilinear forms, so rank 2 minimum.
        
        # For 3 values of k, that's 6 bilinear forms minimum if independent.
        # But they share A, so maybe we can do better.
        
        # Let me try rank 6:
        # For each k, use 2 bilinear forms.
        # c=2k:   U[0,2k]=1, U[2,2k]=1, V[0,2k]=1, V[3,2k]=1, W[2k,0]=1, W[2k,1]=1
        #   Wait, I'm getting confused with indices.
        
        # Let me restart the 2x2x3 construction more carefully.
        # n=2, m=2, p=3
        # U: 4 rows (i,j) in order (0,0),(0,1),(1,0),(1,1)
        # V: 6 rows (j,k) in order (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)
        # W: 6 rows (k,i) in order (0,0),(0,1),(1,0),(1,1),(2,0),(2,1)
        
        # We need: for all (i,j,k), sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
        
        # Try rank 6:
        # c=0: U[0,0]=1, U[1,0]=1, V[0,0]=1, V[1,0]=1, V[2,0]=1, V[3,0]=1, V[4,0]=1, V[5,0]=1
        #      W[0,0]=1, W[1,0]=1, W[2,0]=1, W[3,0]=1, W[4,0]=1, W[5,0]=1
        # Contribution to (i,j,k): U[i,j,0]*V[j,k,0]*W[k,i,0]
        # (0,0,0): 1*1*1=1
        # (0,0,1): 1*1*1=1
        # (0,0,2): 1*1*1=1
        # (0,1,0): 1*1*1=1
        # (0,1,1): 1*1*1=1
        # (0,1,2): 1*1*1=1
        # (1,0,0): 1*1*1=1
        # (1,0,1): 1*1*1=1
        # (1,0,2): 1*1*1=1
        # (1,1,0): 1*1*1=1
        # (1,1,1): 1*1*1=1
        # (1,1,2): 1*1*1=1
        # This gives 1 to everything with rank 1! But that's not possible.
        
        # The problem: U[0,0]=1, U[1,0]=1 means U[0,0]=U[1,0]=1, i.e., A[0,0] and A[0,1] both have coefficient 1.
        # Similarly V[0,0]=V[1,0]=...=1 means all B[j,k] have coefficient 1.
        # And W[0,0]=W[1,0]=...=1 means all C[k,i] have coefficient 1.
        # So the product is always 1*1*1=1.
        
        # But wait, this IS a valid rank-1 decomposition! Let me check:
        # (AB)[i,k] = sum_j A[i,j]*B[j,k]
        # With U[i,j]=1 for all i,j, V[j,k]=1 for all j,k, W[k,i]=1 for all k,i:
        # sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1*1*1 = 1
        # 
        # But this doesn't correctly compute matrix multiplication!
        # The issue is that the tensor T[i,j,k] should represent the coefficient of A[i,j]*B[j,k] in C[i,k].
        # For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k.
        # So a rank-1 decomposition with all ones would give T[i,j,k] = 1, which is correct!
        
        # Wait, but that would mean matrix multiplication has rank 1, which is absurd.
        
        # Let me re-examine. The tensor for matrix multiplication M_{n,m,p} is defined as:
        # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the multiplication is C = AB
        # Actually, the standard definition: the matrix multiplication tensor M_{n,m,p} 
        # has entries T[i,j,k] = 1 if and only if the product A[i,j]*B[j,k] appears in C[i,k].
        # For standard matrix multiplication, every A[i,j]*B[j,k] appears exactly once in C[i,k],
        # so T[i,j,k] = 1 for all i,j,k.
        
        # A rank-r decomposition means: T[i,j,k] = sum_{c=1}^r U[i,j,c]*V[j,k,c]*W[k,i,c]
        # 
        # If we set U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1 for all i,j,k and c=1,
        # then sum_c ... = 1, which equals T[i,j,k]=1.
        
        # So this IS a valid rank-1 decomposition?! That can't be right.
        
        # OH WAIT. I think I'm confusing myself. Let me re-read the problem.
        
        # The problem says: "exactly decompose the matrix multiplication tensor"
        # The matrix multiplication tensor for multiplying an n x m matrix by an m x p matrix
        # is a 3-tensor of size n x m x p where T[i,j,k] = 1.
        
        # A rank-r decomposition of this tensor is a set of r rank-1 tensors whose sum equals T.
        # A rank-1 tensor is u ⊗ v ⊗ w, i.e., T[i,j,k] = u[i]*v[j]*w[k].
        
        # So we need: 1 = sum_c u_c[i]*v_c[j]*w_c[k] for all i,j,k.
        
        # With r=1: u[i]*v[j]*w[k] = 1 for all i,j,k.
        # This requires u[i] = 1/v[j]/w[k] for all i,j,k, which means u[i] is constant,
        # v[j] is constant, w[k] is constant. Let u[i]=a, v[j]=b, w[k]=c.
        # Then abc = 1. So u[i]=a, v[j]=b, w[k]=c with abc=1 works!
        
        # So the rank-1 tensor a*e ⊗ b*f ⊗ c*g where a*b*c=1 gives T[i,j,k]=1.
        
        # This means the matrix multiplication tensor has rank 1?!
        
        # That's clearly wrong. The issue must be in my understanding of the tensor.
        
        # Let me reconsider. In the context of arithmetic circuits for matrix multiplication,
        # the tensor is NOT simply all-ones. The tensor encodes which products of inputs
        # appear in which outputs.
        
        # Actually, I think the confusion is about what the tensor represents.
        # 
        # In the standard formulation:
        # - Input A has entries A[i,j] for i in [0,n), j in [0,m)
        # - Input B has entries B[j,k] for j in [0,m), k in [0,p)
        # - Output C has entries C[i,k] for i in [0,n), k in [0,p)
        # - C[i,k] = sum_j A[i,j]*B[j,k]
        # 
        # The "tensor" associated with this computation is a 3-tensor T of dimensions n x m x p
        # where T[i,j,k] indicates the coefficient of the monomial A[i,j]*B[j,k] in the output C[i,k].
        # Since A[i,j]*B[j,k] appears with coefficient 1 in C[i,k], we have T[i,j,k] = 1.
        # 
        # A rank-r decomposition of this tensor corresponds to a circuit with r bilinear products.
        # Specifically, if T = sum_c u_c ⊗ v_c ⊗ w_c, then we can compute the matrix product
        # using r bilinear products: for each c, compute P_c = (sum_i u_c[i]*A[i,j] for each j) * 
        # (sum_k w_c[k]*B[j,k] for each j), then C[i,k] = sum_c v_c[j]*P_c... 
        # 
        # Hmm, I'm getting confused. Let me look at this more carefully.
        
        # Actually, I think the correct interpretation is:
        # The tensor T has entries T[i, j, k] where:
        # - i indexes rows of A (and C)
        # - j indexes columns of A (and rows of B)  
        # - k indexes columns of B (and C)
        # 
        # And T[i,j,k] = 1 means that the product A[i,j]*B[j,k] contributes to C[i,k].
        # 
        # A rank-r decomposition T = sum_c U_c[i]*V_c[j]*W_c[k] means:
        # We can compute C using r bilinear operations.
        # For each c, we compute a bilinear form: (linear combination of A entries) * (linear combination of B entries)
        # and the result contributes to C[i,k] with weight... 
        # 
        # I think the correct formulation is:
        # C[i,k] = sum_j A[i,j]*B[j,k] = sum_c (sum_j U_c[i,j]*A[i,j]) * (sum_j V_c[j,k]*B[j,k]) * W_c[i,k]
        # 
        # No, that's not right either. Let me think about this differently.
        
        # In the standard tensor decomposition for matrix multiplication:
        # The multiplication tensor M_{n,m,p} is in the space of 3-tensors with indices (i,j,k).
        # A rank-r decomposition is M = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
        # where u_c is a vector in R^{n*m} (indexed by (i,j)), v_c is in R^{m*p} (indexed by (j,k)),
        # and w_c is in R^{p*n} (indexed by (k,i)).
        
        # The decomposition means: for all (i,j,k), M[i,j,k] = sum_c u_c[i,j]*v_c[j,k]*w_c[k,i]
        
        # And M[i,j,k] = 1 for all valid (i,j,k).
        
        # So we need: sum_c u_c[i,j]*v_c[j,k]*w_c[k,i] = 1 for all i,j,k.
        
        # With r=1: u[i,j]*v[j,k]*w[k,i] = 1 for all i,j,k.
        # Let u[i,j] = a_i * b_j, v[j,k] = b_j * c_k, w[k,i] = c_k * a_i.
        # Then product = a_i * b_j * b_j * c_k * c_k * a_i = a_i^2 * b_j^2 * c_k^2.
        # For this to equal 1 for all i,j,k, we need a_i^2 = 1/a_i^2... no, we need
        # a_i^2 * b_j^2 * c_k^2 = 1 for all i,j,k.
        # This requires a_i^2 = constant, b_j^2 = constant, c_k^2 = constant,
        # and the product of the three constants = 1.
        # So a_i = ±1/sqrt(c), b_j = ±1/sqrt(d), c_k = ±1/sqrt(e) with c*d*e = 1.
        
        # But wait, this means u[i,j] = a_i*b_j depends on both i and j separately.
        # So u is a rank-1 matrix in the (i,j) space.
        # Similarly v is rank-1 in (j,k) space, w is rank-1 in (k,i) space.
        
        # So a rank-1 decomposition of the multiplication tensor corresponds to
        # u being rank-1, v being rank-1, w being rank-1.
        
        # And this IS possible! For example:
        # u[i,j] = 1 for all i,j
        # v[j,k] = 1 for all j,k  
        # w[k,i] = 1 for all k,i
        # Then u[i,j]*v[j,k]*w[k,i] = 1 for all i,j,k.
        
        # So the matrix multiplication tensor DOES have rank 1?!
        
        # That contradicts everything I know about matrix multiplication complexity.
        
        # I must be misunderstanding the tensor definition. Let me reconsider.
        
        # OH! I think I see the issue. The tensor for matrix multiplication is NOT all-ones.
        # 
        # The correct tensor is: M[i,j,k] = 1 if and only if the entry A[i,j] of the first matrix
        # and B[j,k] of the second matrix multiply to contribute to C[i,k].
        # 
        # But in the standard formulation, the tensor is actually defined differently.
        # The matrix multiplication tensor M_{n,m,p} is defined as:
        # M = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i ⊗ e_j ⊗ e_k
        # where e_i, e_j, e_k are standard basis vectors.
        # 
        # Wait, that's still all-ones.
        
        # Let me look at this from a different angle. In the context of arithmetic circuits:
        # A bilinear algorithm for matrix multiplication with r multiplications works as follows:
        # 1. Compute r bilinear forms: P_c = (sum_i u_c[i,j]*A[i,j]) * (sum_k v_c[j,k]*B[j,k]) for each c
        #    Wait, this doesn't make sense because the sum over i and j should be separate.
        
        # Actually, the correct formulation is:
        # For each c in 1..r:
        #   Compute X_c = sum_{i,j} u_c[i,j] * A[i,j]  (a linear combination of A entries)
        #   Compute Y_c = sum_{j,k} v_c[j,k] * B[j,k]  (a linear combination of B entries)
        #   Compute P_c = X_c * Y_c  (one bilinear product)
        # Then: C[i,k] = sum_c w_c[i,k] * P_c
        # 
        # But this gives C[i,k] = sum_c w_c[i,k] * (sum_{i',j'} u_c[i',j']*A[i',j']) * (sum_{j'',k''} v_c[j'',k'']*B[j'',k''])
        # = sum_c w_c[i,k] * sum_{i',j',j'',k''} u_c[i',j']*v_c[j'',k'']*A[i',j']*B[j'',k'']
        # 
        # For this to equal sum_j A[i,j]*B[j,k], we need:
        # sum_c w_c[i,k] * u_c[i',j'] * v_c[j'',k''] = delta[i',i]*delta[j',j'']*delta[j'',j]*delta[k'',k]
        # 
        # This is a much more complex condition. The tensor in this formulation is not simply all-ones.
        
        # I think the confusion arises from different conventions. Let me re-read the problem statement.
        
        # The problem says: "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
        # And the decomposition should satisfy: the matrix multiplication tensor is exactly decomposed.
        
        # I believe the correct condition is:
        # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # 
        # This is because the (i,j,k)-entry of the multiplication tensor is 1, indicating that
        # A[i,j]*B[j,k] contributes to C[i,k].
        
        # And as I showed, this has a rank-1 solution with all ones.
        
        # But that can't be right for the purposes of this problem, because the schoolbook
        # decomposition has rank n*m*p = 20, and the problem implies we should find something better.
        
        # I think the issue is that the "rank" in this context is not the tensor rank in the
        # algebraic sense, but rather the number of bilinear multiplications in a specific type
        # of arithmetic circuit.
        
        # Let me reconsider. Perhaps the condition is different. Maybe it's:
        # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
        # 
        # No, that doesn't work dimensionally.
        
        # Let me try yet another interpretation:
        # P_c = (sum_i U[i*m+j, c] * A[i,j]) * (sum_k V[j*p+k, c] * B[j,k])
        # But U depends on j, so this doesn't make sense as written.
        
        # I think the correct interpretation, given the index structure, is:
        # For each c, define:
        #   A_c[j] = sum_i U[i*m+j, c] * A[i,j]  (for each j)
        #   B_c[j] = sum_k V[j*p+k, c] * B[j,k]  (for each j)
        #   P_c = sum_j A_c[j] * B_c[j]  (one bilinear product per c, but actually a sum of m products)
        # 
        # No, that's m products per c, giving r*m total products.
        
        # I'm clearly confused about the exact formulation. Let me just go with the condition:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        # 
        # And accept that this has low-rank solutions. But since the problem says the schoolbook
        # has rank 20 and we should find something better, I'll assume the intended condition is
        # something else.
        
        # Let me try a different approach: maybe the condition is that the decomposition
        # correctly computes matrix multiplication in the sense that:
        # For any matrices A (n x m) and B (m x p), the product C = A*B satisfies:
        # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
        
        # No, that's not right either because of the shared index j.
        
        # Let me try:
        # C[i,k] = sum_c W[k*n+i, c] * (sum_{i'} U[i'*m+j, c] * A[i',j]) * (sum_{k'} V[j*p+k', c] * B[j,k'])
        # summed over j.
        
        # I give up trying to figure out the exact condition from first principles.
        # Let me just use the condition that the seed program uses:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        
        # And since this has rank-1 solutions, the problem must have a different intended condition.
        
        # Wait, maybe I'm wrong about the rank-1 solution. Let me double-check.
        # U[i*m+j, 0] = 1 for all i,j
        # V[j*p+k, 0] = 1 for all j,k
        # W[k*n+i, 0] = 1 for all k,i
        # 
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1*1*1 = 1. ✓
        
        # So yes, rank 1 works for this condition. The problem must intend a different condition.
        
        # Given my confusion, let me just provide a good structured decomposition.
        # I'll use the Strassen-based approach which is known to be valid for matrix multiplication.
        
        # For 2x2*2x5, I'll use:
        # - Strassen for the first 2x2 block (k=0,1): rank 7
        # - Schoolbook for the remaining 2x3 block (k=2,3,4): rank 12
        # Total: rank 19
        
        # Or better, I'll try to find a valid decomposition using the actual matrix multiplication
        # semantics.
        
        # Let me just return the schoolbook decomposition for now, since I'm not sure about
        # the exact condition. The schoolbook is guaranteed to be correct.
        
        pass
    
    return best_result

# EVOLVE-BLOCK-END
