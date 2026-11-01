"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time
import itertools
from scipy.optimize import linprog

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that U,V,W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build expected tensor Texp
    Texp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                Texp[a, b, c] = 1.0
    
    # Compute actual tensor T = sum_r U[:,r] V[:,r] W[:,r]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    err = np.max(np.abs(T - Texp))
    return err < tol

def _schoolbook(n, m, p):
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

def _strassen_2x2():
    """
    Strassen's algorithm for 2x2 matrix multiplication.
    Returns U, V, W matrices for the tensor decomposition.
    """
    # For 2x2 matrices: A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    #
    # Strassen's 7 multiplications:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-c)
    # m5 = (a+b)h
    # m6 = (c-d)(f+h)
    # m7 = (b-c)(e+g)
    #
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Input indices: a=0, b=1, c=2, d=3 (A is 2x2)
    #                e=0, f=1, g=2, h=3 (B is 2x2)
    # Output indices: C11=0, C12=1, C21=2, C22=3 (C is 2x2)
    
    rank = 7
    U = np.zeros((4, rank))  # A has 4 entries
    V = np.zeros((4, rank))  # B has 4 entries
    W = np.zeros((4, rank))  # C has 4 entries
    
    # m1 = (a+d)(e+h)
    U[0, 0] = 1  # a
    U[3, 0] = 1  # d
    V[0, 0] = 1  # e
    V[3, 0] = 1  # h
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # m2 = (c+d)e
    U[2, 1] = 1  # c
    U[3, 1] = 1  # d
    V[0, 1] = 1  # e
    W[2, 1] = 1  # C21
    W[3, 1] = -1 # C22
    
    # m3 = a(f-h)
    U[0, 2] = 1  # a
    V[1, 2] = 1  # f
    V[3, 2] = -1 # h
    W[1, 2] = 1  # C12
    W[3, 2] = 1  # C22
    
    # m4 = d(g-c)
    U[3, 3] = 1  # d
    V[2, 3] = 1  # g
    V[0, 3] = -1 # e... wait, let me reconsider
    
    # Let me redo this more carefully.
    # Actually, let me just use a known Strassen decomposition.
    
    # For 2x2 x 2x2 = 2x2:
    # The tensor has 8 non-zero entries (one for each (i,j,k) where C[i,k] += A[i,j]*B[j,k])
    # We want to decompose into 7 rank-1 tensors.
    
    # Let me use a different approach: just verify the schoolbook and try to find
    # a lower rank using the structure.
    
    return None  # Placeholder, we'll construct it differently

def _block_decomposition(n, m, p):
    """
    Use block decomposition with Strassen for 2x2 blocks.
    
    For 2x4 x 4x7:
    - Split A (2x4) into two 2x2 blocks: A = [A1 | A2]
    - Split B (4x7) into two 2x7 blocks: B = [B1; B2] where B1 is 2x7, B2 is 2x7
    - Then C = A1*B1 + A2*B2
    
    For each 2x2 x 2x7 multiplication, we can further decompose.
    Actually, let's think about this differently.
    
    A is 2x4, B is 4x7.
    We can view this as:
    C = A * B where A is 2x4, B is 4x7.
    
    Split B into columns: B = [B_left | B_right] where B_left is 4x3, B_right is 4x4.
    Then C = [A*B_left | A*B_right] where C_left is 2x3, C_right is 2x4.
    
    For A*B_right (2x4 x 4x4):
    Split A into 2x2 blocks: A = [A1 | A2]
    Split B_right into 2x2 blocks: B_right = [[B11, B12], [B21, B22]]
    Then C_right = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    Each 2x2 x 2x2 = 2x2 can be done with Strassen (rank 7).
    So we need 4 Strassen blocks for the 2x4 x 4x4 part, giving rank 28.
    
    For A*B_left (2x4 x 4x3):
    This is trickier. Let's split differently.
    
    Actually, let's use a simpler approach:
    - Use schoolbook for most of it but try to find savings.
    
    The schoolbook rank is 2*4*7 = 56.
    
    Let's try:
    - Split the 4x7 matrix B into B1 (4x4) and B2 (4x3).
    - Compute C1 = A*B1 (2x4) and C2 = A*B2 (2x3).
    - For C1 = A*B1: both are 2x4 x 4x4. Split A into [A1|A2] (each 2x2), 
      B1 into 2x2 blocks. Use Strassen for each 2x2x2x2=2x2.
      That's 4 Strassen blocks, rank 28.
    - For C2 = A*B2: A is 2x4, B2 is 4x3. 
      Split A into [A1|A2] (each 2x2), B2 into [B21; B22] where B21 is 2x3, B22 is 2x3.
      C2 = A1*B21 + A2*B22.
      Each 2x2 x 2x3 = 2x3 has rank 2*2*3 = 12 (schoolbook).
      So total rank for C2 is 24.
    - Total rank: 28 + 24 = 52.
    
    That's a saving of 4 over schoolbook (56).
    
    Let me implement this.
    """
    n, m, p = 2, 4, 7
    
    # We'll build the decomposition manually.
    # C = A * B, A is 2x4, B is 4x7, C is 2x7.
    
    # Split B into B1 (4x4) and B2 (4x3).
    # C1 = A * B1 (2x4), C2 = A * B2 (2x3).
    
    # For C1 = A * B1:
    # A = [A1 | A2] where A1 is 2x2, A2 is 2x2.
    # B1 = [[B11, B12], [B21, B22]] where each is 2x2.
    # C1 = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    # Each term is 2x2 x 2x2 = 2x2, use Strassen (rank 7).
    # 4 Strassen blocks, total rank 28.
    
    # For C2 = A * B2:
    # A = [A1 | A2] where A1 is 2x2, A2 is 2x2.
    # B2 = [B2_top; B2_bottom] where B2_top is 2x3, B2_bottom is 2x3.
    # C2 = A1*B2_top + A2*B2_bottom
    # Each term is 2x2 x 2x3 = 2x3, schoolbook rank 12.
    # 2 blocks, total rank 24.
    
    # Total rank: 28 + 24 = 52.
    
    rank = 52
    U = np.zeros((n*m, rank))  # 8 x 52
    V = np.zeros((m*p, rank))  # 28 x 52
    W = np.zeros((p*n, rank))  # 14 x 52
    
    col = 0
    
    # Helper: add a Strassen 2x2x2x2=2x2 block
    def add_strassen_block(A_offset, B_offset, C_offset):
        """
        A_offset: starting row index in A for the 2x2 block
        B_offset: starting column index in B for the 2x2 block  
        C_offset: starting column index in C for the 2x2 block
        """
        nonlocal col
        
        # Strassen's 7 multiplications for 2x2 x 2x2 = 2x2
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
        
        # m1 = (a+d)(e+h)
        U[A_offset, col] = 1      # a
        U[A_offset+2, col] = 1    # d (row 2 in A block)
        V[B_offset, col] = 1      # e
        V[B_offset+2, col] = 1    # h (row 2 in B block)
        W[C_offset, col] = 1      # C11
        W[C_offset+2, col] = 1    # C22
        col += 1
        
        # m2 = (c+d)e
        U[A_offset+1, col] = 1    # c
        U[A_offset+2, col] = 1    # d
        V[B_offset, col] = 1      # e
        W[C_offset+1, col] = 1    # C21
        W[C_offset+2, col] = -1   # C22
        col += 1
        
        # m3 = a(f-h)
        U[A_offset, col] = 1      # a
        V[B_offset+1, col] = 1    # f
        V[B_offset+2, col] = -1   # h
        W[C_offset+1, col] = 1    # C12... wait, C12 is at offset+1? 
        # C is 2x2: C[0,0]=C11, C[0,1]=C12, C[1,0]=C21, C[1,1]=C22
        # In our indexing, C_offset is the starting index for C11.
        # C11 is at C_offset, C12 is at C_offset+1, C21 is at C_offset+2, C22 is at C_offset+3.
        # Wait, that's not right. Let me reconsider the indexing.
        
        # Actually, for a 2x2 matrix stored in row-major order:
        # C[0,0] = C11, C[0,1] = C12, C[1,0] = C21, C[1,1] = C22
        # So if C_offset is the index of C11, then:
        # C11 is at C_offset
        # C12 is at C_offset+1
        # C21 is at C_offset+2
        # C22 is at C_offset+3
        
        # Let me redo the Strassen block with correct indexing.
        # I'll reset and do it properly.
        
        return col
    
    # The above is getting complicated. Let me just use the schoolbook for now
    # and try a different approach to find savings.
    
    # Actually, let me just return the schoolbook and try to find a better
    # decomposition through a more systematic search.
    
    return _schoolbook(n, m, p)

def _find_decomposition_via_ilp(n, m, p, max_rank=None):
    """
    Try to find a low-rank decomposition using integer linear programming.
    
    The idea: the tensor T has entries T[a,b,c] = 1 if (a,b,c) corresponds to
    a valid (i,j,k) triple, else 0.
    
    We want to find rank r and matrices U, V, W such that:
    T[a,b,c] = sum_{r=1}^R U[a,r] * V[b,r] * W[c,r]
    
    This is equivalent to finding r rank-1 tensors that sum to T.
    
    For small tensors, we can try to enumerate possible rank-1 tensors and
    use ILP to find a subset that sums to T.
    
    However, the number of possible rank-1 tensors is huge. Instead, we can
    try a different approach: fix U and V, then W is determined by the equations.
    
    For a given U (n*m x r) and V (m*p x r), we need to find W (p*n x r) such that:
    sum_{r=1}^R U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    This is a linear system in W. If U and V are fixed, we can solve for W.
    
    So the problem reduces to: find U and V such that the linear system has a solution.
    
    This is still hard, but for small r, we can try random U and V and check.
    """
    if max_rank is None:
        max_rank = n * m * p
    
    Texp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                Texp[a, b, c] = 1.0
    
    best_rank = max_rank
    best_result = _schoolbook(n, m, p)
    
    # Try random initializations for decreasing ranks
    for r in range(max_rank - 1, max(20, max_rank - 30), -1):
        if time.time() > 140:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            
            # For each c, we have:
            # sum_{r=1}^R U[a,r] * V[b,r] * W[c,r] = Texp[a,b,c]
            # This is a linear system in W[:,c] for each c.
            
            W = np.zeros((p*n, r))
            success = True
            
            for c in range(p*n):
                # Build the linear system for W[:,c]
                # For each (a,b), we have: sum_r U[a,r] * V[b,r] * W[c,r] = Texp[a,b,c]
                # This is: (U * V^T)_{a,b} * W[c,:] = Texp[a,b,c]
                # But that's not quite right. Let me think again.
                
                # Actually, for fixed c, the equation is:
                # sum_{r=1}^R U[a,r] * V[b,r] * W[c,r] = Texp[a,b,c] for all (a,b)
                # Let M = U * V^T (n*m x m*p matrix)
                # Then: sum_r M[a,b] * W[c,r] = Texp[a,b,c]
                # This is: M @ W[c,:] = Texp[:,:,c]
                
                # So we need to solve: M @ W[c,:] = Texp[:,:,c]
                # where M is n*m x m*p
                
                M = U @ V.T  # (n*m) x (m*p)
                rhs = Texp[:, :, c].flatten()  # (n*m*m*p,)
                
                # Flatten M to get a linear system
                M_flat = M.flatten()  # (n*m*m*p,)
                # The equation is: sum_{a,b} M[a,b] * W[c,r] ... no, that's not right.
                
                # Let me reconsider. The tensor equation is:
                # Texp[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For fixed c, this is: Texp[:, :, c] = sum_r (U[:,r] * V[:,r]) * W[c,r]
                # where U[:,r] * V[:,r] is the outer product (n*m x m*p).
                
                # So for fixed c, we have a linear combination of rank-1 matrices.
                # Let A_r = U[:,r] * V[:,r] (outer product, n*m x m*p)
                # Then Texp[:,:,c] = sum_r W[c,r] * A_r
                
                # This is a linear system in W[c,:]:
                # [A_1, A_2, ..., A_r] @ W[c,:] = Texp[:,:,c].flatten()
                
                A = np.zeros((n*m * m*p, r))
                for rr in range(r):
                    A[:, rr] = (U[:, rr] * V[:, rr]).flatten()
                
                try:
                    # Solve A @ W[c,:] = rhs
                    W_c, residuals, rank_sys, sv = np.linalg.lstsq(A, rhs, rcond=None)
                    W[c, :] = W_c
                except:
                    success = False
                    break
            
            if success:
                # Verify the decomposition
                if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    best_rank = r
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
        
        if best_rank < max_rank - 1:
            break
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    
    # Try ILP-based search if time permits
    if time.time() < deadline - 30:
        try:
            ilp_result = _find_decomposition_via_ilp(n, m, p, max_rank=56)
            if ilp_result['U'] is not None:
                if _verify(ilp_result['U'], ilp_result['V'], ilp_result['W'], n, m, p):
                    result = ilp_result
        except Exception as e:
            pass
    
    return result

# EVOLVE-BLOCK-END
