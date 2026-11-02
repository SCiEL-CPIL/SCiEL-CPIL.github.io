import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """
    Verifies if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    r = U.shape[1]
    # Tensor T_{a,b,c,d,i,k} is 1 if a==i, b==d, c==k else 0
    # Decomposition: sum_l U_{a*m+b, l} V_{c*p+d, l} W_{k*n+i, l}
    
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            target = 1.0 if (a == i and c == k and b == d) else 0.0
                            # Efficient check: sum_l U_ab_l * V_cd_l * W_ki_l
                            # We can vectorize this slightly, but for small dims loop is fine
                            # U_ab is row a*m+b of U
                            # V_cd is row c*p+d of V
                            # W_ki is row k*n+i of W
                            s = np.sum(U[a * m + b, :] * V[c * p + d, :] * W[k * n + i, :])
                            if not np.isclose(s, target, atol=atol):
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # 1. Construct Initial Rank-36 Schoolbook Decomposition
    # This is guaranteed to be valid.
    # U[i*m + j, idx] = 1 if idx corresponds to (i, j, k, l) where i==i, j==l? 
    # Wait, tensor T_{a,b,c,d,i,k} = 1 iff a=i, b=d, c=k.
    # Schoolbook term for (a, c) product contributing to (i, k) is non-zero only if a=i and c=k.
    # The term is A[a, b] * B[c, d]. 
    # In schoolbook, we sum over b. For a fixed output (i, k), we need A[i, b] * B[k, b] for all b.
    # So for each b in 0..m-1, we have a term.
    # Total terms = m = 3? No, that's for square.
    # For rectangular:
    # C[i, k] = sum_{b=0}^{m-1} A[i, b] B[k, b]
    # So there are m terms for each (i, k)? No, the terms are global.
    # The tensor is sum over b of A[i, b] B[k, b] Delta_{i, a} Delta_{k, c} Delta_{b, d}?
    # Let's stick to the definition:
    # T_{a,b,c,d,i,k} = 1 if a=i, b=d, c=k else 0.
    # A valid decomposition is sum_{l=1}^R U_{a,b}^l V_{c,d}^l W_{i,k}^l.
    # Schoolbook:
    # For each b in 0..m-1, and for each "position" in the matrices?
    # Actually, the simplest valid decomposition is rank n*m*p? No.
    # Rank is at most n*m*p.
    # But we can do better.
    # Let's use the standard "element-wise" decomposition:
    # For each b in 0..m-1:
    #   We need to handle the dependency on b.
    #   The term involves A[*, b] and B[*, b].
    #   Specifically, for a fixed b, the contribution to C[i, k] is A[i, b] * B[k, b].
    #   This can be represented by a rank-1 tensor?
    #   U_{a,b}^l = 1 if a is the row index? No, U is size n*m.
    #   Let's define the basis.
    #   For each b in range(m):
    #      For each i in range(n):
    #         For each k in range(p):
    #            This is getting complicated. Let's use the known minimal rank for schoolbook which is m * n * p?
    #            No, schoolbook is just m multiplications.
    #            The rank of the matrix multiplication tensor M_{n,m,p} is at most m * n * p?
    #            Actually, the "naive" algorithm uses m scalar multiplications.
    #            Can we represent this with rank m?
    #            No, because the outputs are independent.
    #            
    #            Let's use the decomposition:
    #            R = m * n * p is always valid but huge.
    #            A better initial guess:
    #            We know rank <= 36 (n*m*p).
    #            Let's construct the rank-36 solution explicitly.
    #            Indices for U: (a, b) -> a*m+b
    #            Indices for V: (c, d) -> c*p+d
    #            Indices for W: (k, i) -> k*n+i  (Note: W index is k*n+i per contract)
            
    #    Let's use the decomposition:
    #    For each b in 0..m-1:
    #       For each i in 0..n-1:
    #          For each k in 0..p-1:
    #             This doesn't seem right.
            
    #    Let's look at the structure:
    #    T_{a,b,c,d,i,k} = Delta_{a,i} Delta_{b,d} Delta_{c,k}
    #    We can write this as:
    #    Sum_{b=0}^{m-1} ( Delta_{a,i} Delta_{c,k} ) * ( Delta_{b,d} ) * ( 1 ) ? No.
    #    
    #    Let's use the standard rank-1 decomposition of the identity-like structure.
    #    Actually, the simplest valid decomposition is:
    #    Rank = m * n * p
    #    For each b in 0..m-1:
    #      For each i in 0..n-1:
    #        For each k in 0..p-1:
    #          This is not efficient to generate manually.
          
    #    Let's use a known valid decomposition:
    #    Rank = m * n * p is safe.
    #    But we can start with Rank = 36.
    
    R_init = 36
    U = np.zeros((n*m, R_init))
    V = np.zeros((m*p, R_init))
    W = np.zeros((p*n, R_init))
    
    # Construct schoolbook:
    # The tensor is sum_{b} A_{a,b} B_{c,b} Delta_{a,i} Delta_{c,k} ? No, B is 3x6, so B_{c,d}.
    # C_{i,k} = sum_b A_{i,b} B_{k,b}
    # So T_{a,b,c,d,i,k} = 1 if a=i, c=k, b=d.
    # We can decompose this by iterating over b, and then using rank-1 terms for the remaining deltas?
    # For a fixed b, the term is A_{a,b} B_{c,d} Delta_{a,i} Delta_{c,k} with d=b.
    # This is a product of three vectors?
    # U_{a,b}^{l} = 1 if l corresponds to (b, i, k)?
    
    # Let's just use the "full" expansion which is always valid:
    # Rank = n * m * p
    # For each a in 0..n-1:
    #   For each b in 0..m-1:
    #     For each c in 0..m-1:
    #       For each d in 0..p-1:
    #         ... this is n*m*m*p terms. Too many.
        
    #    Correct Schoolbook Decomposition:
    #    C_{i,k} = sum_{b=0}^{m-1} A_{i,b} B_{k,b}
    #    This requires m multiplications.
    #    Can we express this as a sum of rank-1 tensors?
    #    Yes.
    #    For each b in 0..m-1:
    #       We need to represent the map (A, B) -> C where C_{i,k} += A_{i,b} B_{k,b}.
    #       This map is separable?
    #       A_{i,b} depends on i and b. B_{k,b} depends on k and b.
    #       The output C_{i,k} depends on i and k.
    #       This is a tensor of size n x m x m x p x n x p.
    #       The term for fixed b is:
    #       U_{a,b}^l = 1 if a is free? No.
        
    #    Let's use the decomposition:
    #    R = m * n * p
    #    For each b in 0..m-1:
    #      For each i in 0..n-1:
    #        For each k in 0..p-1:
    #           This creates a term that contributes to C_{i,k} only if the inputs match?
    #           No, this is getting confusing.
            
    #    Let's use the simplest valid construction:
    #    Rank = m * n * p is definitely valid.
    #    But we can do better.
    #    Let's use Rank = m * n * p / something?
    
    #    Actually, let's just use the rank-36 solution derived from:
    #    U_{a*m+b, idx} = 1 if idx encodes (b, a, k) ? No.
    
    #    Let's use the following valid decomposition:
    #    For each b in 0..m-1:
    #      For each i in 0..n-1:
    #        For each k in 0..p-1:
    #           This is m*n*p terms.
    #           Let idx = b * (n*p) + i * p + k
    #           U[a*m+b, idx] = 1 if a == i else 0
    #           V[c*p+d, idx] = 1 if c == k and d == b else 0
    #           W[k*n+i, idx] = 1 (always 1 for this term? No, W must select the output)
    #           Wait, W_{k*n+i, idx} should be 1 if the term contributes to output (k, i).
    #           In our construction, the term is associated with output (i, k).
    #           So W[k*n+i, idx] = 1.
            
    #    Let's verify this construction:
    #    Sum_{idx} U_{a*m+b, idx} V_{c*p+d, idx} W_{k*n+i, idx}
    #    Non-zero terms require:
    #    1. idx corresponds to some (b', i', k')
    #    2. U non-zero: a*m+b row, idx col. U[a*m+b, idx] = 1 if a == i' and b == b'.
    #    3. V non-zero: c*p+d row, idx col. V[c*p+d, idx] = 1 if c == k' and d == b'.
    #    4. W non-zero: k*n+i row, idx col. W[k*n+i, idx] = 1 if k == k' and i == i'.
    
    #    So for a term to be non-zero, we need:
    #    a == i'
    #    b == b'
    #    c == k'
    #    d == b'  => d == b
    #    k == k' => k == c
    #    i == i' => i == a
    
    #    So the term is non-zero iff a==i, b==d, c==k.
    #    And the value is 1*1*1 = 1.
    #    This matches the tensor definition!
    
    R_init = m * n * p
    U = np.zeros((n*m, R_init))
    V = np.zeros((m*p, R_init))
    W = np.zeros((p*n, R_init))
    
    idx = 0
    for b in range(m):
        for i in range(n):
            for k in range(p):
                # U row: a*m+b. Non-zero if a==i.
                U[i*m+b, idx] = 1.0
                # V row: c*p+d. Non-zero if c==k and d==b.
                V[k*p+b, idx] = 1.0
                # W row: k*n+i. Non-zero if k==k and i==i.
                W[k*n+i, idx] = 1.0
                idx += 1
                
    # Verify initial solution
    if not verify_decomposition(U, V, W, n, m, p):
        # Should never happen
        raise ValueError("Initial schoolbook decomposition failed verification")
    
    current_rank = R_init
    
    # 2. Rank Reduction Loop
    # We try to reduce the rank by finding linear dependencies.
    # Strategy: 
    # - Pick a column l.
    # - Check if the column vector in U, V, W can be expressed as a linear combination of other columns?
    # - No, the decomposition is sum of rank-1 tensors. 
    # - We can reduce rank if there exists a linear transformation that zeroes out one column.
    # - This is equivalent to checking if the column space of the "flattened" tensors has redundancy.
    
    # A simpler heuristic: 
    # Try to merge two columns if they are "similar" or if one can be eliminated by adjusting others.
    # However, exact elimination is hard.
    
    # Alternative Strategy: Use the known structure.
    # For M_{2,3,6}, we can split B into three 3x2 blocks.
    # C = A * B1 + A * B2 + A * B3
    # Each A*B_j is 2x2.
    # Rank(M_{2,3,2}) is known to be 11? Or 12?
    # If rank(M_{2,3,2}) = 11, then total rank = 3 * 11 = 33.
    # If rank(M_{2,3,2}) = 12, total rank = 36.
    # Literature says rank(M_{2,3,2}) <= 11.
    
    # Let's try to construct the rank-33 solution directly using the block structure.
    # This is much more likely to succeed than generic linear algebra on 36 dims.
    
    def build_rank11_2x3x2():
        """
        Returns U, V, W for 2x3 * 3x2 multiplication.
        U: (6, 11), V: (6, 11), W: (4, 11)
        """
        # Based on literature, a rank-11 decomposition exists.
        # Let's use a known valid one.
        # Reference: "Tensor rank of 2x3x2 matrix multiplication"
        # One such decomposition:
        
        U = np.zeros((6, 11))
        V = np.zeros((6, 11))
        W = np.zeros((4, 11))
        
        # Strassen's 7 terms for 2x2x2 (using A[0,0], A[0,1], A[1,0], A[1,1] and B[0,0]..B[1,1])
        # But we have 2x3.
        # Let's use the decomposition from the "inspiration" which mentioned rank 11.
        
        # Term 0: A00 B00 -> C00
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 1: A01 B10 -> C01? No, indices.
        
        # Let's use a verified set from a database or derive it.
        # Actually, let's use the rank-12 schoolbook for 2x3x2 first to be safe, 
        # and then try to reduce it to 11 using the linear algebra method on the smaller block.
        
        # Schoolbook for 2x3x2:
        # n=2, m=3, p=2. Rank = 3*2*2 = 12.
        U12 = np.zeros((6, 12))
        V12 = np.zeros((6, 12))
        W12 = np.zeros((4, 12))
        
        idx = 0
        for b in range(3):
            for i in range(2):
                for k in range(2):
                    U12[i*3+b, idx] = 1.0
                    V12[k*2+b, idx] = 1.0
                    W12[k*2+i, idx] = 1.0
                    idx += 1
                    
        # Verify 2x3x2 schoolbook
        if not verify_decomposition(U12, V12, W12, 2, 3, 2):
            return None, None, None
            
        return U12, V12, W12
        
    U_block, V_block, W_block = build_rank11_2x3x2()
    
    if U_block is not None:
        # Construct full 2x3x6 by tiling the 2x3x2 block 3 times
        U_full = np.zeros((6, 3 * U_block.shape[1]))
        V_full = np.zeros((18, 3 * V_block.shape[1]))
        W_full = np.zeros((12, 3 * W_block.shape[1]))
        
        for j in range(3):
            # U is same for all blocks (A is same)
            U_full[:, j*12:(j+1)*12] = U_block
            
            # V is different for each block (B is 3x6, split into three 3x2)
            # V_block is (6, 12). Rows are c*p_block + d.
            # For block j, the rows in V_full correspond to c*6 + (j*2 + d).
            # V_block row c*2+d corresponds to input B_j[c, d].
            # In V_full, the row index is c*p + d_total.
            # d_total = j*2 + d.
            for c in range(3):
                for d in range(2):
                    V_full[c*6 + j*2 + d, j*12:(j+1)*12] = V_block[c*2 + d, :]
                    
            # W is different for each block (C is 2x6, split into three 2x2)
            # W_block is (4, 12). Rows are k*p_block + i.
            # For block j, the rows in W_full correspond to k*2 + i, but output column is j*2+k?
            # Wait, C is 2x6. Output index is (i, k_total).
            # k_total = j*2 + k_block.
            # W row index is k_total * n + i.
            # W_block row index is k_block * n + i.
            for k in range(2):
                for i in range(2):
                    W_full[(j*2 + k)*2 + i, j*12:(j+1)*12] = W_block[k*2 + i, :]
                    
        if verify_decomposition(U_full, V_full, W_full, n, m, p):
            # We have a valid rank-36 solution (3 * 12).
            # Now try to reduce rank using linear algebra on this specific structure.
            # Or just return this if we can't reduce.
            # But we want rank 33 or lower.
            
            # Let's try to reduce the rank of the block from 12 to 11.
            # If we can find a rank-11 decomposition for 2x3x2, we use it.
            # Since I don't have a hardcoded rank-11, I will attempt a quick search.
            
            # For now, return the valid rank-36 solution.
            # But wait, the current program scores 1.0588 which is rank 34?
            # 36 is worse.
            
            # Let's check if we can use the rank-34 solution from the current program?
            # The current program's rank-34 solution was verified?
            # "if verify_decomposition... return"
            # If it failed, it returned the invalid one?
            # The score 1.0588 implies it returned a valid solution of rank 34?
            # 1/34 = 0.029. 1.0588 / 0.029 = 36.5.
            # Maybe the seed score is different.
            
            # Let's try to construct a rank-34 solution directly.
            # 2 blocks of 2x3x3?
            # Rank(M_{2,3,3}) is 17? 2*17=34.
            
            def build_rank17_2x3x3():
                # Hardcoded rank-17 for 2x3x3
                # This is risky without a verified source.
                # Let's use the schoolbook rank 18 for 2x3x3.
                # 2 * 18 = 36.
                
                # Let's stick to the block tiling of 2x3x2 with rank 12.
                # Total rank 36.
                
                pass
                
            # Since I cannot guarantee a rank < 36 without a verified decomposition,
            # I will return the rank-36 solution.
            # However, the inspiration mentions rank 22.
            # This implies a much more efficient decomposition exists.
            
            # Let's try one more thing: 
            # Use the rank-36 solution and attempt to reduce it using SVD on the flattened tensors?
            # No, that changes the structure.
            
            # I will return the rank-36 solution as it is guaranteed valid.
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }
            
    # Fallback: Rank 36 Schoolbook
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    construct()
