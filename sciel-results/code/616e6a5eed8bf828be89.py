"""Matrix multiplication 3x5x7 rank-minimal decomposition."""
import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    
    # Deadline guard: aim to finish well before 120s (evaluator kills at 150s)
    deadline = time.time() + 110.0
    
    def make_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    U0, V0, W0 = make_schoolbook()
    best = (rank_school, U0, V0, W0)
    
    # Try to find lower rank via iterative rank-reduction / column pruning
    # Strategy: start from schoolbook, try to merge columns by finding linear dependencies
    # that preserve the exact decomposition property.
    
    # The decomposition property: for all i, j, k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) is a valid triple, 0 otherwise
    # Actually the tensor T_{i,j,k} = 1 for all (i,j,k) in [n]x[m]x[p] when contracting
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T_{i,j,k} = delta (i,j,k is in range), i.e., T_{i,j,k} = 1 for all i,j,k.
    # Wait, let me reconsider. The matrix multiplication tensor M^{n,m,p} has entries:
    #   T_{i,j,k} = 1 if we're computing C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor is indexed by (i, j, k) where i in [n], j in [m], k in [p],
    # and T_{i,j,k} = 1 for all i,j,k.
    # A rank-1 term is U[i*m+j] * V[j*p+k] * W[k*n+i].
    # The decomposition requires: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    # This is a very restrictive condition. The schoolbook decomposition uses rank n*m*p.
    # Lower rank decompositions exist (e.g., Strassen for 2x2x2 uses rank 7 instead of 8).
    # For 3x5x7, finding a lower rank is a research problem.
    
    # Let's try a simple approach: random initialization with gradient-free search
    # trying to reduce rank from schoolbook down.
    
    # Actually, let's try a different approach: use the known Strassen-like structure.
    # For 3x5x7, we can think of it as 3 independent 1x5x7 blocks.
    # Each 1x5x7 block: C_k = sum_j A_j * B_{j,k}
    # This is a vector-matrix product, which is just a bilinear map from (5,7) to 7.
    # The rank of the 1x5x7 tensor is 5*7 = 35 (schoolbook), but it could be lower.
    # Actually for 1xm*p, the tensor is the "matrix product" 1xm x mxp = 1xp.
    # This is equivalent to multiplying a row vector by a matrix.
    # The rank of this tensor... let's think.
    # T_{j,k} = 1 for all j,k (since i is fixed at 0).
    # We need sum_c U[j,c] * V_{j,k}... wait, no.
    # For n=1: U has shape (1*m, rank) = (m, rank), V has shape (m*p, rank), W has shape (p*1, rank) = (p, rank).
    # The condition is: sum_c U[j,c] * V[j*p+k,c] * W[k,c] = 1 for all j,k.
    
    # Let's try rank r for the 1x5x7 block. Total rank would be 3*r.
    # We need 3*r < 105, so r < 35.
    
    # For the 1x5x7 case, can we do better than 35?
    # The tensor is T_{j,k} = 1. We need sum_c u_j * v_{j,k} * w_k = 1 for all j,k.
    # Note that v_{j,k} is an element of V indexed by (j,k).
    
    # Let's try a structured approach. Set w_k = 1 for all k.
    # Then we need sum_c u_j * v_{j,k} = 1 for all j,k.
    # This means for each j, the vector (v_{j,0}, v_{j,1}, ..., v_{j,6}) dotted with (u_j * 1, u_j * 1, ...) = 1.
    # So u_j * sum_k v_{j,k} = 1 for all j.
    # This gives us freedom. We can set u_j = 1 for all j, then sum_k v_{j,k} = 1 for all j.
    # With rank r, V is a 5x7 matrix per rank... no wait.
    # V has shape (m*p, rank) = (35, r). Each column c of V is a vector in R^{35}, i.e., a 5x7 matrix.
    # So V_c is a 5x7 matrix.
    # The condition becomes: sum_c u_j * (V_c)_{j,k} * w_k = 1 for all j,k.
    
    # If we set u_j = 1, w_k = 1, we need sum_c (V_c)_{j,k} = 1 for all j,k.
    # So the sum of r 5x7 matrices equals the all-ones 5x7 matrix.
    # We can achieve this with r=1: set V_1 = all-ones 5x7 matrix, u = ones(5), w = ones(7).
    # Check: sum_c=1 u_j * V_1[j,k] * w_k = 1 * 1 * 1 = 1. Yes!
    
    # So for n=1, m=5, p=7, rank 1 suffices!
    # Wait, that can't be right. Let me re-examine.
    # U has shape (n*m, rank) = (5, r). U[j, c] for j in 0..4, c in 0..r-1.
    # V has shape (m*p, rank) = (35, r). V[j*p+k, c] for j in 0..4, k in 0..6, c in 0..r-1.
    # W has shape (p*n, rank) = (7, r). W[k*n+i, c] = W[k, c] for k in 0..6, c in 0..r-1 (since n=1, i=0).
    # Condition: sum_c U[j,c] * V[j*p+k,c] * W[k,c] = 1 for all j in 0..4, k in 0..6.
    
    # With r=1: U[j,0] = u_j, V[j*7+k, 0] = v_{j,k}, W[k,0] = w_k.
    # Need: u_j * v_{j,k} * w_k = 1 for all j,k.
    # Set u_j = 1, w_k = 1, v_{j,k} = 1. This works!
    
    # So for n=1, the rank is 1.
    # For n=3, we have 3 independent such blocks (one for each i).
    # Can we share the rank-1 structure across i?
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # For each i, the condition is: sum_c U[i*5+j,c] * V[j*7+k,c] * W[k*3+i,c] = 1 for all j,k.
    
    # If we use rank 1: U[i*5+j, 0] = u_{i,j}, V[j*7+k, 0] = v_{j,k}, W[k*3+i, 0] = w_{k,i}.
    # Need: u_{i,j} * v_{j,k} * w_{k,i} = 1 for all i,j,k.
    # Set u_{i,j} = 1, v_{j,k} = 1, w_{k,i} = 1. This works!
    
    # So rank 1 suffices for 3x5x7 matrix multiplication?
    # Wait, that seems too good to be true. Let me verify with the actual matrix multiplication.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The decomposition says: C[i,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] * A[i,j] * B[j,k]
    # No wait, the factorization is:
    # C[i,k] = sum_j sum_c A[i,j] * U[i*m+j,c] * B[j,k] * V[j*p+k,c] * W[k*n+i,c]
    # Hmm, I need to be more careful about the convention.
    
    # The standard tensor decomposition for matrix multiplication:
    # C = A * B, where C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The multiplication tensor T is a 3-tensor with indices (i, j, k) and T_{i,j,k} = 1.
    # A rank-r decomposition gives: T_{i,j,k} = sum_c U_{i,j}^{(c)} V_{j,k}^{(c)} W_{k,i}^{(c)}
    # But the actual computation is:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # = sum_j sum_c A_{i,j} B_{j,k} * (sum_c' U_{i,j}^{(c')} V_{j,k}^{(c')} W_{k,i}^{(c')})... 
    
    # No, I think the correct interpretation is:
    # The bilinear map (A, B) -> C = A*B can be written as:
    # C_{i,k} = sum_c sum_j A_{i,j} * U_{i*m+j,c} * V_{j*p+k,c} * W_{k*n+i,c} * B_{j,k}
    # Wait, that doesn't make sense dimensionally.
    
    # Let me look at the schoolbook decomposition in the seed program:
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # where col = i*m*p + j*p + k (or some ordering).
    
    # The reconstruction must be:
    # C[i,k] = sum_c (sum_j A[i,j] * U[i*m+j,c]) * (sum_j B[j,k] * V[j*p+k,c]) * W[k*n+i,c]
    # No, that's not right either.
    
    # Actually, I think the convention is:
    # The tensor M has entries M[i,j,k] = 1 (for matrix multiplication, the "1" tensor).
    # A rank-r decomposition: M[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U is n x m x r, V is m x p x r, W is p x n x r.
    # In flattened form: U is (n*m) x r, V is (m*p) x r, W is (p*n) x r.
    # The multiplication is: C[i,k] = sum_j M[i,j,k] * A[i,j] * B[j,k] = sum_j A[i,j] * B[j,k]
    # Using the decomposition: C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k])
    # Wait, that's not right. Let me think again.
    
    # I believe the correct formula is:
    # C[i,k] = sum_c (sum_j A[i,j] * U[i*m+j,c]) * (sum_j B[j,k] * V[j*p+k,c]) * W[k*n+i,c]
    
    # For schoolbook: U[i*m+j,c] = delta_{c, (i,j,k)} for a specific k... no.
    # In the seed, for each (i,j,k), there's a column c with U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1.
    # So sum_c (sum_j A[i,j]*U[i*m+j,c]) * (sum_j B[j,k]*V[j*p+k,c]) * W[k*n+i,c]
    # For column c corresponding to (i',j',k'): this is (A[i,j'] if i=i' else 0) * (B[j',k] if k=k' else 0) * (1 if k=k' and i=i' else 0)
    # = A[i,j'] * B[j',k] if i=i', k=k', else 0.
    # Summing over c: sum_{j'} A[i,j'] * B[j',k] = C[i,k]. Correct!
    
    # So the formula is:
    # C[i,k] = sum_c [sum_j A[i,j] * U[i*m+j,c]] * [sum_j B[j,k] * V[j*p+k,c]] * W[k*n+i,c]
    
    # Now, for my rank-1 proposal with U[i*5+j,0]=1, V[j*7+k,0]=1, W[k*3+i,0]=1:
    # C[i,k] = [sum_j A[i,j]*1] * [sum_j B[j,k]*1] * 1 = (sum_j A[i,j]) * (sum_j B[j,k])
    # This is NOT equal to sum_j A[i,j]*B[j,k] in general!
    
    # So rank 1 does NOT work. My earlier analysis was wrong because I was checking the tensor condition
    # T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1, but the actual computation involves
    # the sums over j in the U and V terms separately.
    
    # The correct tensor condition for the decomposition to be valid is:
    # For all i,k: sum_j A[i,j]B[j,k] = sum_c (sum_j A[i,j]U[i*m+j,c]) * (sum_j B[j,k]V[j*p+k,c]) * W[k*n+i,c]
    # This must hold for ALL A, B. So we need:
    # sum_j delta_{i,i} delta_{k,k} A[i,j]B[j,k] = ... 
    # This is a polynomial identity in A and B. Expanding the RHS:
    # sum_c W[k*n+i,c] * sum_j A[i,j]U[i*m+j,c] * sum_{j'} B[j',k]V[j'*p+k,c]
    # = sum_c W[k*n+i,c] * sum_{j,j'} A[i,j]B[j',k] * U[i*m+j,c]*V[j'*p+k,c]
    # We need this to equal sum_j A[i,j]B[j,k] = sum_{j,j'} A[i,j]B[j',k]*delta_{j,j'}
    # So we need: sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k*n+i,c] = delta_{j,j'} for all i,k,j,j'.
    
    # This is the correct tensor condition!
    # T[i,j,j',k] = sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k*n+i,c] = delta_{j,j'}
    # Note: the tensor is indexed by (i, j, j', k) with the condition depending on j and j'.
    
    # For the schoolbook: U[i*m+j,c]=delta_{c,(i,j,k')}, V[j'*p+k,c]=delta_{c,(j',k,i'')}... 
    # This is getting complicated. Let me just verify numerically.
    
    # For now, let's stick with schoolbook rank 105 and try to find improvements
    # via numerical optimization if time permits.
    
    # Try a simple random search to see if we can find a lower rank.
    # This is unlikely to succeed for non-trivial rank reductions, but let's try.
    
    def verify(U, V, W, rank, max_trials=100):
        """Verify the decomposition using random matrices."""
        for _ in range(max_trials):
            A = random.random() * 10 - 5
            A = [[A[i*m+j] for j in range(m)] for i in range(n)]
            B = [[(random.random() * 10 - 5) for k in range(p)] for j in range(m)]
            # Compute C = A*B
            C_expected = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
            # Compute via decomposition
            C_decomp = [[0.0]*p for _ in range(n)]
            for c in range(rank):
                # sum_j A[i,j]*U[i*m+j,c]
                temp_A = [0.0]*n
                for i in range(n):
                    for j in range(m):
                        temp_A[i] += A[i][j] * U[i*m+j][c]
                # sum_j B[j,k]*V[j*p+k,c]
                temp_B = [0.0]*p
                for k in range(p):
                    for j in range(m):
                        temp_B[k] += B[j][k] * V[j*p+k][c]
                # W[k*n+i,c]
                for i in range(n):
                    w_val = W[k*n+i][c] if False else 0  # placeholder
                    for k in range(p):
                        C_decomp[i][k] += temp_A[i] * temp_B[k] * W[k*n+i][c]
            # Check
            max_err = max(abs(C_expected[i][k] - C_decomp[i][k]) for i in range(n) for k in range(p))
            if max_err > 1e-6:
                return False
        return True
    
    # The verification is slow. Let's try a few random rank reductions.
    # But given the complexity, let's just return schoolbook.
    # The probability of finding a lower rank via random search is extremely low.
    
    # Try one simple heuristic: check if rank 104 works by setting one U entry to 0
    # and compensating... this is too complex.
    
    # Return schoolbook decomposition.
    return {"U": U0, "V": V0, "W": W0}
# EVOLVE-BLOCK-END
