"""Combined approach: Strassen-based block decomposition for 2x2 * 2x13 matmul."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit
    
    n, m, p = 2, 2, 13
    
    # The matrix multiplication tensor T has shape (n*m, m*p, p*n) = (4, 26, 26)
    # T[a, b, c] = 1 iff the (i,k) from a, (k',j) from b, (j',i') from c satisfy k=k', j=j', i=i'
    # where a = i*m+k, b = k'*p+j, c = j'*n+i
    
    # Strategy: Use Strassen's algorithm for 2x2 blocks to reduce rank.
    # For 2x2 * 2x2, Strassen gives rank 7 instead of 8.
    # For 2x2 * 2x13, we can split the 13 columns into 6 blocks of 2 + 1 block of 1.
    # Each 2-column block can use a rank-7 decomposition (Strassen-like).
    # The 1-column block uses rank 4 (schoolbook for 2x2 * 2x1).
    # Total: 6*7 + 4 = 46
    
    # But we need to be careful: the tensor structure means we can't simply apply
    # independent Strassen to each block because the A matrix is shared.
    
    # Actually, let's think more carefully. The tensor is:
    # T[(i,k), (k',j), (j',i')] = 1 if k=k', j=j', i=i'
    # 
    # A rank-1 term is U[:,t] ⊗ V[:,t] ⊗ W[:,t] where:
    # U has support on some set of (i,k) indices
    # V has support on some set of (k',j) indices  
    # W has support on some set of (j',i') indices
    # 
    # The product contributes to T[a,b,c] iff a is in supp(U[:,t]), b in supp(V[:,t]), c in supp(W[:,t])
    # and the values multiply to give the correct tensor entry.
    
    # For Strassen on 2x2*2x2, the 7 rank-1 terms have specific structures.
    # Let's derive them explicitly.
    
    # For a 2x2 block with columns j0, j1:
    # U indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V indices: (0,j0)=j0, (0,j1)=j1, (1,j0)=m+j0, (1,j1)=m+j1
    # W indices: (j0,0)=j0*n, (j0,1)=j0*n+1, (j1,0)=j1*n, (j1,1)=j1*n+1
    
    # The 8 tensor entries that must be 1:
    # T[0, j0, j0*n] = 1      (i=0,k=0,k'=0,j=j0,j'=j0,i'=0)
    # T[0, j1, j1*n] = 1      (i=0,k=0,k'=0,j=j1,j'=j1,i'=0)
    # T[1, m+j0, j0*n+1] = 1  (i=0,k=1,k'=1,j=j0,j'=j0,i'=0)
    # T[1, m+j1, j1*n+1] = 1  (i=0,k=1,k'=1,j=j1,j'=j1,i'=0)
    # T[2, j0, j0*n] = 1      (i=1,k=0,k'=0,j=j0,j'=j0,i'=1)
    # T[2, j1, j1*n] = 1      (i=1,k=0,k'=0,j=j1,j'=j1,i'=1)
    # T[3, m+j0, j0*n+1] = 1  (i=1,k=1,k'=1,j=j0,j'=j0,i'=1)
    # T[3, m+j1, j1*n+1] = 1  (i=1,k=1,k'=1,j=j1,j'=j1,i'=1)
    
    # Wait, I need to re-examine. Let me be more careful.
    # a = i*m + k, so for n=2,m=2: a=0:(i=0,k=0), a=1:(i=0,k=1), a=2:(i=1,k=0), a=3:(i=1,k=1)
    # b = k'*p + j, so for p=13: b ranges over 0..25
    # c = j'*n + i', so for n=2: c = j'*2 + i'
    
    # For the sub-block with columns j0, j1:
    # Nonzero entries of T restricted to these columns:
    # (a=0, b=j0, c=j0*2+0=2*j0) : i=0,k=0, k'=0,j=j0, j'=j0,i'=0 ✓
    # (a=0, b=j1, c=j1*2+0=2*j1) : i=0,k=0, k'=0,j=j1, j'=j1,i'=0 ✓
    # (a=1, b=m+j0=2+j0, c=j0*2+1=2*j0+1) : i=0,k=1, k'=1,j=j0, j'=j0,i'=0 ✓
    # (a=1, b=m+j1=2+j1, c=j1*2+1=2*j1+1) : i=0,k=1, k'=1,j=j1, j'=j1,i'=0 ✓
    # (a=2, b=j0, c=j0*2+0=2*j0) : i=1,k=0, k'=0,j=j0, j'=j0,i'=1... wait c=j'*n+i'=j0*2+1
    # Let me redo: c = j'*n + i'. For i'=1: c = j'*2 + 1
    # (a=2, b=j0, c=j0*2+1=2*j0+1) : i=1,k=0, k'=0,j=j0, j'=j0,i'=1 ✓
    # (a=2, b=j1, c=j1*2+1=2*j1+1) : i=1,k=0, k'=0,j=j1, j'=j1,i'=1 ✓
    # (a=3, b=m+j0=2+j0, c=j0*2+1=2*j0+1) : i=1,k=1, k'=1,j=j0, j'=j0,i'=1 ✓
    # (a=3, b=m+j1=2+j1, c=j1*2+1=2*j1+1) : i=1,k=1, k'=1,j=j1, j'=j1,i'=1 ✓
    
    # So the 8 nonzero entries for block (j0,j1) are:
    # (0, j0, 2*j0), (0, j1, 2*j1), (1, 2+j0, 2*j0+1), (1, 2+j1, 2*j1+1),
    # (2, j0, 2*j0+1), (2, j1, 2*j1+1), (3, 2+j0, 2*j0+1), (3, 2+j1, 2*j1+1)
    
    # Now, can we decompose this 8-term tensor into 7 rank-1 terms?
    # Each rank-1 term t contributes u_t[a]*v_t[b]*w_t[c] to T[a,b,c].
    
    # Let's set up the Strassen decomposition. In Strassen's algorithm for 2x2 matrices:
    # Given A = [[a,b],[c,d]] and B = [[e,f],[g,h]], compute C = A*B
    # 
    # P1 = a*(e+h)
    # P2 = (c+d)*e
    # P3 = (a+b)*h
    # P4 = d*(g-e)
    # P5 = (a+c)*(f-h)
    # P6 = (b-a+d)*g
    # P7 = (b-d)*(e+h)
    #
    # C = [[P1-P4+P6-P7, P3+P7], [P2+P4, P1-P3+P5+P6]]
    
    # In our tensor framework, each "product" P_t is a bilinear form in A and B entries.
    # For the sub-block, A entries are A[i,k] for i,k in {0,1} and B entries are B[k',j] for k' in {0,1}, j in {j0,j1}.
    
    # The key insight: in the tensor decomposition, each rank-1 term corresponds to one "multiplication" in the algorithm.
    # For Strassen, each P_t is a product of a linear combination of A entries and a linear combination of B entries.
    # 
    # P1 = A[0,0] * (B[0,j0] + B[1,j1])  -- but this doesn't directly map to a single (i,k,j) triple.
    
    # Actually, I think the correct way to view this is:
    # Each rank-1 term t has:
    # U[:,t] = vector of coefficients for A entries
    # V[:,t] = vector of coefficients for B entries  
    # W[:,t] = vector of coefficients for C entries
    # 
    # And the constraint is that sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t] = T
    
    # For a single (i,k,j) entry, the schoolbook term has U[i*m+k]=1, V[k*p+j]=1, W[j*n+i]=1.
    # 
    # For Strassen, we need to find 7 such vectors that sum to the tensor.
    
    # Let me just try to solve this numerically. Set up the linear system.
    
    # For the 2-column block with j0=0, j1=1 (first block), let's find the decomposition.
    # Local U indices: 0,1,2,3 (corresponding to (i,k) = (0,0),(0,1),(1,0),(1,1))
    # Local V indices: 0,1,2,3 (corresponding to (k',j) = (0,0),(0,1),(1,0),(1,1))
    # Local W indices: 0,1,2,3 (corresponding to (j',i') = (0,0),(0,1),(1,0),(1,1))
    
    # The local tensor S has S[a,b,c] = 1 for the 8 entries listed above (with j0=0,j1=1).
    # S[0,0,0]=1, S[0,1,2]=1, S[1,2,1]=1, S[1,3,3]=1,
    # S[2,0,1]=1, S[2,1,3]=1, S[3,2,1]=1, S[3,3,3]=1
    
    # Wait, let me recompute with j0=0, j1=1:
    # (0, j0=0, 2*j0=0) -> S[0,0,0]=1
    # (0, j1=1, 2*j1=2) -> S[0,1,2]=1
    # (1, 2+j0=2, 2*j0+1=1) -> S[1,2,1]=1
    # (1, 2+j1=3, 2*j1+1=3) -> S[1,3,3]=1
    # (2, j0=0, 2*j0+1=1) -> S[2,0,1]=1
    # (2, j1=1, 2*j1+1=3) -> S[2,1,3]=1
    # (3, 2+j0=2, 2*j0+1=1) -> S[3,2,1]=1
    # (3, 2+j1=3, 2*j1+1=3) -> S[3,3,3]=1
    
    # Now solve for 7 rank-1 terms.
    # This is equivalent to finding 7 triples of vectors (u_t, v_t, w_t) in R^4 such that
    # sum_t u_t ⊗ v_t ⊗ w_t = S
    
    # Let me use a numerical approach. Initialize with random vectors and use gradient descent.
    
    def find_rank_decomposition(tensor, rank, max_iter=5000, lr=0.01, seed=42):
        """Find a rank decomposition of a 4D tensor using gradient descent."""
        np.random.seed(seed)
        shape = tensor.shape
        # Initialize U, V, W randomly
        U = np.random.randn(shape[0], rank) * 0.1
        V = np.random.randn(shape[1], rank) * 0.1
        W = np.random.randn(shape[2], rank) * 0.1
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            # Compute current tensor
            T = np.einsum('ar,br,cr->abc', U, V, W)
            loss = np.sum((T - tensor)**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-10:
                break
            
            # Compute gradients
            # dLoss/dU = 2 * (T - tensor) contracted with V and W
            # dLoss/dU[a,r] = 2 * sum_{b,c} (T[a,b,c] - tensor[a,b,c]) * V[b,r] * W[c,r]
            diff = T - tensor
            dU = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
            dV = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
            dW = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Decay learning rate
            lr *= 0.9999
        
        return best_U, best_V, best_W, best_loss
    
    # Try to find rank-7 decomposition of the 2x2x2 sub-tensor
    S = np.zeros((4, 4, 4))
    S[0,0,0] = 1; S[0,1,2] = 1; S[1,2,1] = 1; S[1,3,3] = 1
    S[2,0,1] = 1; S[2,1,3] = 1; S[3,2,1] = 1; S[3,3,3] = 1
    
    # Try multiple random seeds
    best_rank7 = None
    best_loss7 = float('inf')
    
    if time.time() < deadline - 60:
        for seed in range(10):
            U7, V7, W7, loss7 = find_rank_decomposition(S, 7, max_iter=3000, lr=0.05, seed=seed)
            if loss7 < best_loss7:
                best_loss7 = loss7
                best_rank7 = (U7, V7, W7)
            if best_loss7 < 1e-8:
                break
    
    # Check if we found a good decomposition
    if best_rank7 is not None and best_loss7 < 1e-6:
        U7, V7, W7 = best_rank7
        # Build the full decomposition using this rank-7 block
        rank = 6 * 7 + 4  # 6 blocks of 2 columns (rank 7 each) + 1 block of 1 column (rank 4)
        rank = 46
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        # 6 blocks of 2 columns
        for block in range(6):
            j0 = block * 2
            j1 = block * 2 + 1
            
            # Map local indices to global
            # Local U: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1) -> global U[0..3]
            # Local V: 0->(0,j0), 1->(0,j1), 2->(1,j0), 3->(1,j1) -> global V[j0], V[j1], V[m+j0], V[m+j1]
            # Local W: 0->(j0,0), 1->(j0,1), 2->(j1,0), 3->(j1,1) -> global W[j0*n], W[j0*n+1], W[j1*n], W[j1*n+1]
            
            for t in range(7):
                U[0, col+t] = U7[0, t]
                U[1, col+t] = U7[1, t]
                U[2, col+t] = U7[2, t]
                U[3, col+t] = U7[3, t]
                
                V[j0, col+t] = V7[0, t]
                V[j1, col+t] = V7[1, t]
                V[m+j0, col+t] = V7[2, t]
                V[m+j1, col+t] = V7[3, t]
                
                W[j0*n, col+t] = W7[0, t]
                W[j0*n+1, col+t] = W7[1, t]
                W[j1*n, col+t] = W7[2, t]
                W[j1*n+1, col+t] = W7[3, t]
            
            col += 7
        
        # Last block: 1 column (j=12)
        # Schoolbook for 2x2 * 2x1: rank 4
        j = 12
        for i in range(n):
            for k in range(m):
                U[i*m+k, col] = 1
                V[k*p+j, col] = 1
                W[j*n+i, col] = 1
                col += 1
        
        # Verify
        T = np.einsum('ar,br,cr->abc', U, V, W)
        T_true = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T_true[i*m+k, k*p+j, j*n+i] = 1
        
        error = np.sum((T - T_true)**2)
        
        if error < 1e-6:
            # Convert to lists with rounded values
            U_list = [[round(U[i][j], 6) for j in range(rank)] for i in range(n*m)]
            V_list = [[round(V[i][j], 6) for j in range(rank)] for i in range(m*p)]
            W_list = [[round(W[i][j], 6) for j in range(rank)] for i in range(p*n)]
            
            # Check if all values are close to integers or half-integers
            # For the purpose of this task, we just need exact decomposition
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
    rank = n * m * p  # 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][column] = 1
                V[k * p + j][column] = 1
                W[j * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
