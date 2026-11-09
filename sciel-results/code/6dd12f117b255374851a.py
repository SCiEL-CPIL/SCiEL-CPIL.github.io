"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Try to find minimum rank decomposition
    # Lower bound is max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # Schoolbook rank is n*m*p = 60
    
    # Strategy: Use block decomposition
    # Partition p columns into blocks, use efficient sub-decompositions
    
    # For 2x3 x 3xk multiplication, we can use the following:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a bilinear form. The tensor T[i,j,k] = 1 for all valid indices.
    
    # Let's try to find a rank-r decomposition by solving the system
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # We'll use a greedy/iterative approach to find minimal rank
    
    # First, let's try the schoolbook as baseline
    rank_schoolbook = n * m * p  # 60
    
    # Try to find better decompositions using block structure
    # Partition the p=10 columns into groups
    
    # Key insight: For fixed (i,j), the entry depends on k through B[j,k]
    # The tensor is "separable" in the sense that T[i,j,k] = 1 for all i,j,k
    # This means it's the all-ones tensor of shape (2,3,10)
    
    # The all-ones tensor of shape (a,b,c) has rank = min(a*b, b*c, a*c) if we can use
    # the structure. Actually, for all-ones tensor, rank = min(a,b,c) * something...
    
    # Wait, the all-ones tensor T[i,j,k] = 1 can be written as:
    # T[i,j,k] = sum_{c=1}^{r} u[i,c] * v[j,c] * w[k,c]
    # If we set u[i,c] = 1/sqrt(r), v[j,c] = 1/sqrt(r), w[k,c] = 1/sqrt(r) for c=1..r,
    # then T[i,j,k] = r * (1/sqrt(r))^3 = 1/sqrt(r), which is not 1.
    
    # Actually, for the all-ones tensor, the rank is 1! Because:
    # T[i,j,k] = 1 = (1) * (1) * (1) for all i,j,k
    # So U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1 gives rank 1!
    
    # Wait, that can't be right. Let me re-examine.
    # The matrix multiplication tensor is NOT the all-ones tensor.
    # T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    # For standard matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k).
    # So yes, it IS the all-ones tensor!
    
    # But wait, that would mean rank 1, which is clearly wrong for matrix multiplication.
    # Let me reconsider the tensor definition.
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 where:
    # i indexes rows of A (and C), j indexes columns of A / rows of B, k indexes columns of B (and C)
    # The multiplication is C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is indeed T[i,j,k] = 1 for all valid (i,j,k).
    
    # But the rank of this tensor is NOT 1. The rank of the all-ones tensor of shape (a,b,c)
    # is actually min(a,b,c) in general? No...
    
    # Let me think again. The tensor T[i,j,k] = 1 can be decomposed as:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Note that W depends on (k,i), not just k.
    
    # If I set U[i,j,c] = 1, V[j,k,c] = 1, W[k,i,c] = 1 for c=0 only,
    # then T[i,j,k] = 1*1*1 = 1. So rank 1 works?
    
    # But that would mean matrix multiplication has rank 1, which is absurd.
    # I must be misunderstanding the tensor structure.
    
    # Let me re-read the problem. The tensor for matrix multiplication A(n x m) * B(m x p) = C(n x p)
    # is T[i,j,k] where the multiplication is C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T has T[i,j,k] = 1 for all valid i,j,k.
    # A rank-1 decomposition would be T[i,j,k] = U[i,j] * V[j,k] * W[k,i]
    # If U[i,j]=1, V[j,k]=1, W[k,i]=1, then T[i,j,k]=1. So rank 1?
    
    # This is clearly wrong. The issue is that the tensor decomposition for matrix multiplication
    # is not simply the all-ones tensor. Let me look at this more carefully.
    
    # Actually, I think the issue is that the "tensor" here refers to the trilinear form, and
    # the rank is the bilinear complexity. The schoolbook algorithm has rank n*m*p, and
    # Strassen shows that for 2x2x2, rank is 7 < 8.
    
    # For the all-ones tensor of shape (a,b,c), the tensor rank is indeed 1 if we allow
    # arbitrary real/complex entries, because T[i,j,k] = 1 = 1*1*1.
    
    # But that contradicts Strassen's result. So I must be misunderstanding the tensor.
    
    # OH WAIT. I think I see the issue. The matrix multiplication tensor is NOT the all-ones tensor.
    # Let me reconsider. The tensor T is defined such that the bilinear form is:
    # <T, A, B> = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_{i,k} C[i,k]
    # where C = A*B.
    # So T[i,j,k] should be such that sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} (A*B)[i,k]
    # = sum_{i,k} sum_j A[i,j] B[j,k]
    # = sum_{i,j,k} A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k). It IS the all-ones tensor.
    
    # But then the rank should be 1, which contradicts known results. I must be making an error.
    
    # Let me look at this from a different angle. The tensor rank of the matrix multiplication tensor
    # T_{n,m,p} is the bilinear complexity of multiplying an n x m matrix by an m x p matrix.
    # For n=m=p=2, this is 7 (Strassen), not 1.
    
    # The resolution: The tensor T[i,j,k] = 1 is correct, but the rank-1 decomposition
    # T[i,j,k] = U[i,j] V[j,k] W[k,i] requires that the SAME U, V, W work for all i,j,k.
    # If U[i,j]=1, V[j,k]=1, W[k,i]=1, then indeed T[i,j,k]=1 for all i,j,k.
    # So why is the rank not 1?
    
    # I think the issue is that in tensor decomposition for matrix multiplication, the factors
    # are constrained. Specifically, the rank-r decomposition corresponds to r bilinear forms
    # that, when combined with the inputs, give the output. The rank-1 case would mean
    # the entire multiplication can be done with one scalar multiplication, which is impossible
    # for n,m,p > 1.
    
    # Let me reconsider. A rank-1 term in the tensor decomposition is:
    # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
    # The contribution to C[i,k] from this term is:
    # sum_j u[i,j] * v[j,k] * w[k,i] * A[i,j] * B[j,k]
    # = w[k,i] * (sum_j u[i,j] A[i,j]) * (sum_j v[j,k] B[j,k])
    # = w[k,i] * (U A)[i] * (V B)[k]
    # where U is a 1 x m row vector and V is a 1 x p row vector... no, that's not right either.
    
    # I think I need to be more careful. Let me just implement the schoolbook and then try
    # to find a better decomposition using known results.
    
    # For 2x3 x 3x10, let's use the following approach:
    # Split the 10 columns into 5 blocks of 2.
    # For each block, we compute 2x3 x 3x2 multiplication.
    # The 2x3 x 3x2 tensor has rank... let's compute it.
    
    # Actually, let me just try to construct a decomposition and verify it.
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is correct."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # Check shapes
        if U.shape != (n*m, rank):
            return False
        if V.shape != (m*p, rank):
            return False
        if W.shape != (p*n, rank):
            return False
        # Reconstruct tensor
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Should equal 1 for all valid (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Schoolbook decomposition
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find better decompositions
    # For 2x3 x 3xk, we can use the following:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = A[i,0] B[0,k] + A[i,1] B[1,k] + A[i,2] B[2,k]
    
    # For k in a block of size 2, say k=2b, 2b+1:
    # C[i,2b] = A[i,0]B[0,2b] + A[i,1]B[1,2b] + A[i,2]B[2,2b]
    # C[i,2b+1] = A[i,0]B[0,2b+1] + A[i,1]B[1,2b+1] + A[i,2]B[2,2b+1]
    
    # This is a 2x3 x 3x2 problem. Let's find its minimal rank.
    # The tensor is T[i,j,k] for i in {0,1}, j in {0,1,2}, k in {0,1}, all = 1.
    
    # Let's try rank 6 (schoolbook for this subproblem)
    # And see if we can do better.
    
    # Actually, for the full problem, let me try a systematic approach.
    # I'll attempt ranks from the lower bound up and check if a valid decomposition exists.
    
    # Lower bound: max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # But this is a weak bound. Let's try from 30 up.
    
    # For efficiency, let's use a randomized approach with verification.
    
    import random
    random.seed(42)
    
    def try_random_decomposition(n, m, p, rank, num_trials=100):
        """Try to find a random rank-r decomposition."""
        for trial in range(num_trials):
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            # Check if this works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return U.tolist(), V.tolist(), W.tolist()
        return None
    
    # Try various ranks
    for rank in range(30, best_rank + 1):
        if time.time() > deadline - 5:
            break
        result = try_random_decomposition(n, m, p, rank, num_trials=20)
        if result:
            U, V, W = result
            if verify_decomposition(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
    
    return best_result
# EVOLVE-BLOCK-END
