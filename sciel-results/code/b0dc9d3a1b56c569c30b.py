"""Find a rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 50

    # For 2x5 * 5x5, we try to find a lower rank decomposition.
    # The tensor T[i,j,k] = delta_{i,k} (output index i from first matrix row, 
    # but wait - let me think about the tensor structure more carefully).
    
    # Matrix multiplication: C = A * B where A is n x m (2x5), B is m x p (5x5), C is n x p (2x5)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor for matrix multiplication has entries T[i,j,k] where:
    # - i indexes rows of A (and C): 0..n-1
    # - j indexes columns of A / rows of B: 0..m-1  
    # - k indexes columns of B (and C): 0..p-1
    # 
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k], i.e., always 1 for valid indices
    # Actually, the standard tensor is T[i,j,k] = delta_{i,k}... no.
    
    # Let me reconsider. The matrix multiplication tensor T in U ⊗ V ⊗ W where:
    # U has dimension n*m (indices (i,j) for A)
    # V has dimension m*p (indices (j,k) for B)  
    # W has dimension n*p (indices (i,k) for C)
    # 
    # T[(i,j),(j',k),(i',k')] = 1 if i=i', j=j', k=k' ... no that's not right either.
    
    # The correct tensor: T is a 3-tensor with T[a,b,c] where:
    # a = i*m + j (row i, col j of A), a in 0..n*m-1
    # b = j'*p + k (row j', col k of B), b in 0..m*p-1
    # c = i'*n + k' (row i', col k' of C), c in 0..n*p-1
    # 
    # T[a,b,c] = 1 if the entry A[i,j] * B[j',k] contributes to C[i',k']
    # This happens when i=i', j=j', k=k'
    # So T[i*m+j, j*p+k, i'*n+k'] = 1 if i=i' and j=j' and k=k'
    
    # Wait, but the W factor uses trace convention W[k*n+i, r]. Let me re-read.
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed as W[k*n+i, r] where k is the column of C and i is the row of C.
    # So c = k*n + i (not i*n + k as I assumed).
    
    # Let me redo: W has dimension p*n, indexed by (k,i) -> k*n+i
    # So the tensor is:
    # T[i*m+j, j*p+k, k'*n+i'] = 1 if i=i' and j=j' and k=k'
    
    # For the schoolbook decomposition, each term (i,j,k) gives:
    # U[i*m+j, col] = 1
    # V[j*p+k, col] = 1
    # W[k*n+i, col] = 1
    # (Note: W index is k*n+i, matching the trace convention)
    
    # Now, can we do better than rank 50?
    # For 2x5*5x5, the lower bounds are:
    # - n*m*p / (n+m+p-2) is one bound (but this is weak)
    # - The standard lower bound is max(n*m, m*p, n*p) = max(10, 25, 10) = 25
    # - Another bound: n*m*p - (n-1)(m-1)(p-1) ... various bounds
    
    # Let's try to find a decomposition of rank less than 50.
    # One approach: use the fact that for small matrices, there are known
    # decompositions that beat schoolbook.
    
    # For 2x2*2x2, rank 7 (Strassen) beats schoolbook rank 8.
    # For 2x5*5x5, we can try to exploit structure.
    
    # Strategy: Try to find a rank-r decomposition by solving a system of equations.
    # We need U (10 x r), V (25 x r), W (10 x r) such that:
    # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = 1 if (i,j,k)=(i',j',k') else 0
    
    # This is a system of (n*m*p)^2 = 125000 equations in r*(10+25+10) = 45r unknowns.
    # For r=40, that's 1800 unknowns and 125000 equations - overdetermined.
    
    # Alternative: use bilinear rank / tensor rank algorithms.
    # For small problems, we can try random initialization + gradient descent or
    # alternating least squares on the tensor.
    
    # Let me try a different approach: exploit the structure.
    # The tensor T[i,j,k; i',j',k'] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    # Actually, I realize the tensor is simpler. Let me define it properly.
    # T is a 6-dimensional tensor (or 3-tensor with multi-indices):
    # T[(i,j),(j',k),(i',k')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    # This tensor has a special structure. It's essentially the "identity" in some sense.
    
    # Let's try to find a low-rank decomposition using the following idea:
    # Group terms by j (the shared index). For each j, we have a sub-tensor
    # that maps (i,k) -> (i',k') with the condition i=i', k=k'.
    # This sub-tensor for fixed j is just the identity on n*p = 10 elements.
    # The identity tensor of size 10x10 has rank 10 (schoolbook) but can it be lower?
    # The identity tensor I_{10} in C^{10} ⊗ C^{10} ⊗ C^{10}... wait, this isn't quite right.
    
    # Let me think differently. The full tensor T can be written as:
    # T = sum_j T_j where T_j is the tensor for fixed j.
    # T_j[(i,j),(j',k),(i',k')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    
    # For fixed j, T_j is nonzero only when j'=j, and then it's delta_{i,i'} * delta_{k,k'}.
    # So T_j = e_j ⊗ (sum_i e_i ⊗ e_i) ⊗ (sum_k e_k ⊗ e_k) in some sense.
    
    # This is getting complicated. Let me just try to solve it numerically.
    
    # Approach: Try ranks from a lower bound up to schoolbook, and for each rank,
    # try to find a decomposition using random initialization and optimization.
    
    # Lower bound: For n x m * m x p, the rank is at least max(n*m, m*p, n*p) = 25.
    # But better bounds exist. For 2x5*5x5, let's try ranks from 25 to 49.
    
    # Actually, let me try a smarter approach. The tensor has a Kronecker-like structure.
    # T[(i,j),(j',k),(i',k')] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    # We can write this as:
    # T = (sum_i e_i ⊗ e_i) ⊗_? ... hmm, not quite a Kronecker product.
    
    # Let me just try to find a decomposition by solving linear systems.
    # For a given rank r, we have r columns in U, V, W.
    # The constraint is: for all (i,j,j',k,i',k'):
    # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    # This is a system of nonlinear equations (cubic in the variables).
    # Hard to solve directly.
    
    # Alternative: use the fact that the tensor is "simple" and try to find
    # a decomposition by exploiting symmetry or using known results.
    
    # For now, let's try to see if we can find a rank-40 or so decomposition
    # using a heuristic search.
    
    # Actually, let me try a completely different approach: use the structure
    # of the problem. The matrix multiplication tensor for 2x5*5x5 can be
    # decomposed by splitting the 5x5 into blocks.
    
    # Split the 5 columns of A and 5 rows of B into groups.
    # For example, split into groups of sizes 2,2,1 or 3,2 or 2,3 etc.
    
    # If we split j into groups J1, J2, ..., the computation becomes:
    # C = A1*B1 + A2*B2 + ...
    # where Ai is n x |Ji| and Bi is |Ji| x p.
    
    # For each group, we can use a low-rank decomposition of that sub-problem.
    # For 2x2*2x2, we can use Strassen (rank 7).
    # For 2x1*1x5 or 2x5*5x1, the rank is just 2*5=10 or 5*1*5... 
    # Actually for 2x1*1xp, the rank is 2*p (each output C[i,k] = A[i,0]*B[0,k]).
    
    # Let's try splitting 5 into 2+2+1:
    # Group 1: 2x2*2x2 -> rank 7 (Strassen)
    # Group 2: 2x2*2x2 -> rank 7 (Strassen)  
    # Group 3: 2x1*1x5 -> rank 10
    # Total: 7 + 7 + 10 = 24
    
    # Wait, that's below the lower bound of 25! That can't be right.
    # Let me check: for 2x1*1x5, C[i,k] = A[i,0]*B[0,k].
    # This is an outer product of a 2-vector and a 5-vector, so rank 10? 
    # No, the tensor for 2x1*1x5 has T[i*1+0, 0*5+k, k'*2+i'] = delta_{i,i'}delta_{k,k'}
    # = 1 if i=i' and k=k'.
    # This tensor has entries T[a,b,c] where a=i (0,1), b=k (0..4), c=k'*2+i' (0..9)
    # T[i,k,k'*2+i'] = 1 if i=i' and k=k'
    # So for each (i,k), there's exactly one nonzero entry in W.
    # The rank of this tensor: we need r such that sum_c U[i,c]V[k,c]W[k'*2+i',c] = delta_{i,i'}delta_{k,k'}
    # For fixed i, k: sum_c U[i,c]V[k,c]W[c_idx,c] where c_idx = k*2+i
    # This must equal 1 if (i',k')=(i,k) and 0 otherwise.
    # So U[i,:] * V[k,:] (elementwise with W rows) must give the indicator.
    # This is essentially a rank-10 tensor (each of the 10 combinations (i,k) needs its own term).
    # So rank 10 for 2x1*1x5 is correct.
    
    # Total for 2+2+1 split: 7+7+10 = 24. But lower bound says >= 25. Contradiction?
    # Let me re-examine the Strassen decomposition for 2x2*2x2.
    # Strassen: rank 7 for 2x2*2x2. The lower bound for 2x2*2x2 is 7 (it's tight).
    # For 2x2*2x2: n*m=4, m*p=4, n*p=4, lower bound = max(4,4,4)=4, but actual min rank is 7.
    
    # So my split gives 7+7+10=24, but the true minimum might be higher.
    # The issue is that when we split, the decompositions of the parts might not
    # combine optimally. The total rank is the sum of the ranks of the parts,
    # but there might be a way to do better by not splitting.
    
    # Anyway, let's try to implement the 2+2+1 split with Strassen for the 2x2 parts.
    
    # Strassen decomposition for 2x2*2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+g)
    # M7 = (a-c)(e+g)
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Let me encode this.
    
    def strassen_2x2():
        """Return U, V, W for 2x2*2x2 Strassen decomposition.
        U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
        Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        # 7 terms
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 1: M1 = (a+d)(e+h)
        # a=A[0,0]=U[0], d=A[1,1]=U[3]
        # e=B[0,0]=V[0], h=B[1,1]=V[3]
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # Contributes to C[0,0] and C[1,1]
        # C[0,0]: W[0*2+0, 0] = W[0, 0] += 1
        # C[1,1]: W[1*2+1, 0] = W[3, 0] += 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # Term 2: M2 = (c+d)e
        # c=A[1,0]=U[2], d=A[1,1]=U[3], e=B[0,0]=V[0]
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # Contributes to C[1,0]: W[0*2+1, 1] = W[2, 1] += 1
        W[2, 1] = 1
        
        # Term 3: M3 = a(f-h)
        # a=A[0,0]=U[0], f=B[0,1]=V[1], h=B[1,1]=V[3]
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        # Contributes to C[0,1]: W[1*2+0, 2] = W[2, 2] += 1
        # And C[1,1]: W[3, 2] += 1
        W[2, 2] = 1; W[3, 2] = 1
        
        # Term 4: M4 = d(g-e)
        # d=A[1,1]=U[3], g=B[1,0]=V[2], e=B[0,0]=V[0]
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        # Contributes to C[0,0]: W[0, 3] += -1
        # And C[1,0]: W[2, 3] += 1
        W[0, 3] = -1; W[2, 3] = 1
        
        # Term 5: M5 = (a+b)h
        # a=A[0,0]=U[0], b=A[0,1]=U[1], h=B[1,1]=V[3]
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # Contributes to C[0,0]: W[0, 4] += 1
        # And C[0,1]: W[2, 4] += 1
        W[0, 4] = 1; W[2, 4] = 1
        
        # Term 6: M6 = (c-d)(f+g)
        # c=A[1,0]=U[2], d=A[1,1]=U[3], f=B[0,1]=V[1], g=B[1,0]=V[2]
        U[2, 5] = 1; U[3, 5] = -1
        V[1, 5] = 1; V[2, 5] = 1
        # Contributes to C[1,1]: W[3, 5] += 1
        W[3, 5] = 1
        
        # Term 7: M7 = (a-c)(e+g)
        # a=A[0,0]=U[0], c=A[1,0]=U[2], e=B[0,0]=V[0], g=B[1,0]=V[2]
        U[0, 6] = 1; U[2, 6] = -1
        V[0, 6] = 1; V[2, 6] = 1
        # Contributes to C[0,0]: W[0, 6] += -1
        W[0, 6] = -1
        
        return U, V, W
    
    # Now let's build the full 2x5*5x5 decomposition using the 2+2+1 split.
    # Columns 0,1 -> Strassen 1
    # Columns 2,3 -> Strassen 2
    # Column 4 -> schoolbook 2x1*1x5
    
    U_full = np.zeros((10, 24))  # 2*5=10 rows
    V_full = np.zeros((25, 24))  # 5*5=25 rows
    W_full = np.zeros((10, 24))  # 5*2=10 rows (k*n+i)
    
    # Strassen 1: columns 0,1 of A, rows 0,1 of B
    U1, V1, W1 = strassen_2x2()
    U_full[0:4, 0:7] = U1  # A[0:2, 0:2] -> rows 0,1 of U_full
    V_full[0:4, 0:7] = V1  # B[0:2, 0:2] -> rows 0,1 of V_full (j*p+k for j=0,1; k=0,1)
    # W index: k*n+i for k in 0,1 and i in 0,1 -> k*2+i
    W_full[0:4, 0:7] = W1
    
    # Strassen 2: columns 2,3 of A, rows 2,3 of B
    U2, V2, W2 = strassen_2x2()
    U_full[4:8, 7:14] = U2  # A[0:2, 2:4] -> rows 4,5,6,7 of U_full
    # V index: j*p+k for j=2,3 and k=0,4? No, k is column of B.
    # B[j,k] where j=2,3 and k=0,1,2,3,4. We only use j=2,3 and all k? 
    # Wait, in the split, B1 is rows 0,1 of B (all 5 columns), B2 is rows 2,3 of B (all 5 columns).
    # But Strassen 2x2*2x2 only handles 2x2. I need to handle 2x5*5x2 or something.
    
    # I made an error. When splitting the middle dimension j into groups,
    # A becomes [A_J1 | A_J2 | A_J3] and B becomes [B_J1; B_J2; B_J3]
    # where A_Ji is n x |Ji| and B_Ji is |Ji| x p.
    # So for group 1 (j=0,1): A_J1 is 2x2, B_J1 is 2x5.
    # The sub-problem is 2x2 * 2x5, not 2x2*2x2!
    
    # I need a decomposition for 2x2 * 2x5, not 2x2*2x2.
    # This is getting complicated. Let me reconsider.
    
    # For 2x2 * 2x5:
    # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k] for i=0,1 and k=0..4
    # This is 10 outputs, each a sum of 2 products.
    # Schoolbook rank: 2*2*5 = 20.
    # Can we do better? For 2x2*2x2 we can do 7, so for 2x2*2x5 we might do 7*5/2 or something.
    # Actually, each column k of C depends on A and column k of B.
    # C[:,k] = A * B[:,k] which is 2x2 * 2x1.
    # For 2x2*2x1, the rank is... let's see. C[i] = A[i,0]*B[0,k] + A[i,1]*B[1,k].
    # This is 2 outputs, each a bilinear form in 4 variables.
    # The tensor has T[i,j,k; i',j',k'] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
    # For 2x2*2x1: n=2,m=2,p=1. Rank at least max(4,2,2)=4.
    # Schoolbook rank = 4. Can we do 3? Probably not for this small case.
    
    # This approach is getting too complex. Let me just return the schoolbook
    # decomposition for now, which is guaranteed correct.
    
    # Actually, let me try one more thing: a simple rank reduction by combining
    # terms that share structure.
    
    # For now, return schoolbook.
    rank = schoolbook_rank
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
