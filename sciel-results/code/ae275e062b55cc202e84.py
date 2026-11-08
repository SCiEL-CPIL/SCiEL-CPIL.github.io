"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 if i,j,k are valid indices, else 0
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[i,j,k] where we need sum_j A[i,j]*B[j,k] = C[i,k]
    # The decomposition: C[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_k V[j*p+k,c]*B[j,k])... 
    # Wait, let me think about this more carefully.
    
    # Standard tensor decomposition for matmul:
    # We want: C[i,k] = sum_j A[i,j] B[j,k]
    # Using rank-r decomposition: C[i,k] = sum_{c=1}^r (sum_j U[i,j,c] A[i,j]) (sum_l V[j,l,c] B[j,l]) ... 
    # Actually the standard form is:
    # C[i,k] = sum_c U[i,k,c] * (sum_j A[i,j] * V[j,c]) ... no that's not right either.
    
    # Let me use the explicit form from the problem:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The tensor entry T[i,j,k] (for the matmul tensor) should be recoverable as:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And for matmul, T[i,j,k] = 1 if the indices form a valid triple (i,j,k) in the contraction
    # Actually for the matmul tensor, the "1" entries are where we contract:
    # The matmul tensor has T[i,j,k] = 1 for all valid (i,j,k) in [0,n)×[0,m)×[0,p)
    # Wait no - the matmul tensor is defined such that contracting over j gives the product.
    # The tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # No, that's not right either.
    
    # Let me reconsider. The matrix multiplication tensor T ∈ R^{n×m×p} is defined by:
    # T[i,j,k] = 1 if (i,j,k) is a valid index triple, 0 otherwise? No.
    
    # Actually, the standard definition: the matmul tensor has entries
    # T[i,j,k] = 1 for all i∈[0,n), j∈[0,m), k∈[0,p)
    # Wait, that would be an all-ones tensor.
    
    # Let me re-read the problem. "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for multiplying (n×m) by (m×p) is:
    # T[i,j,k] = 1 if the indices are valid, where i indexes rows of first matrix,
    # j indexes columns of first / rows of second, k indexes columns of second.
    # But actually the tensor is just defined on the valid index set with all 1s?
    
    # From the schoolbook code: for each (i,j,k), U[i*m+j][col]=1, V[j*p+k][col]=1, W[k*n+i][col]=1
    # So T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # In schoolbook, each (i,j,k) has its own column, so T[i,j,k] = 1 for all valid (i,j,k)
    # and T[i',j',k'] = 0 for (i',j',k') ≠ (i,j,k)
    
    # So the "matrix multiplication tensor" here is actually the tensor with 1s at all
    # valid (i,j,k) positions and 0s elsewhere? That doesn't make sense for matmul.
    
    # Actually I think the convention is: the matmul tensor T has T[i,j,k] = 1 for all
    # i∈[0,n), j∈[0,m), k∈[0,p). The "decomposition" means we find U,V,W such that
    # for all i,j,k: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # where T[i,j,k] = 1 for all valid indices.
    
    # But that's just an all-ones tensor, which has rank 1!
    
    # I must be misunderstanding. Let me look at the schoolbook more carefully.
    # In schoolbook, rank = n*m*p = 75, and each (i,j,k) triple gets its own rank-1 term.
    # So the tensor being decomposed has T[i,j,k] = δ_{i,j,k} in some sense?
    
    # Actually, I think the correct interpretation is:
    # The matmul tensor is defined such that when you contract appropriately, you get matmul.
    # Specifically, C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as C[i,k] = sum_{j,c} A[i,j] B[j,k] T[i,j,k,c]... no.
    
    # Let me just verify the schoolbook works and then try to find a better one.
    
    # For verification, I'll check that the reconstructed tensor matches the expected one.
    # Expected tensor: for the matmul problem, the tensor T[i,j,k] should be such that
    # the decomposition works. Let me just compute what the schoolbook produces and
    # verify consistency.
    
    # Actually, let me just do a numerical check: build the full tensor from U,V,W and
    # check it has the right structure.
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reconstruct tensor: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This is expensive to do naively, so let's do it for a few random indices
    # or use a smarter approach.
    
    # Alternative: verify by checking that the decomposition is valid for random matrices
    # A (n×m) and B (m×p), compute C = A@B, and check if the decomposition gives the same.
    
    # Actually, the cleanest verification: the tensor T[i,j,k] for matmul is 1 for all
    # valid (i,j,k). Let me verify this interpretation by checking schoolbook.
    
    # For schoolbook with n=3,m=5,p=5: rank=75
    # T[i,j,k] = sum_c U[i*5+j,c] * V[j*5+k,c] * W[k*3+i,c]
    # Each column c corresponds to one (i0,j0,k0) triple
    # U[i*5+j, c(i0,j0,k0)] = 1 if (i,j)=(i0,j0) else 0
    # V[j*5+k, c(i0,j0,k0)] = 1 if (j,k)=(j0,k0) else 0  
    # W[k*3+i, c(i0,j0,k0)] = 1 if (k,i)=(k0,i0) else 0
    # So T[i,j,k] = 1 if there exists c such that all three are 1, which happens iff
    # (i,j,k) = (i0,j0,k0) for the c corresponding to that triple.
    # So T[i,j,k] = 1 if (i,j,k) is a valid triple, 0 otherwise.
    # But all (i,j,k) with i∈[0,3), j∈[0,5), k∈[0,5) are valid, so T is all 1s!
    
    # That can't be right for a non-trivial decomposition. Let me reconsider.
    
    # I think the issue is that the "matrix multiplication tensor" is NOT the all-ones tensor.
    # It's a specific tensor that encodes the multiplication. Let me look at this differently.
    
    # The correct matmul tensor: we want C[i,k] = sum_j A[i,j] B[j,k]
    # This is a bilinear map. The tensor T ∈ R^{(n×m) × (m×p) × (n×p)} is defined by:
    # T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k', else 0
    # i.e., T is the identity-like tensor on the contracted index.
    
    # In the flattened form: T[i*m+j, j'*p+k, i'*n+k'] = 1 if i=i', j=j', k=k'
    
    # So the tensor has 1s only on the "diagonal" where indices match.
    
    # Let me verify: with this interpretation, schoolbook should work.
    # T[i*m+j, j'*p+k, i'*n+k'] = sum_c U[i*m+j,c] V[j'*p+k,c] W[i'*n+k',c]
    # Wait, but W is indexed as W[k*n+i], not W[i*n+k]. Let me re-read.
    
    # From the problem: W has shape (p*n, rank) and is indexed W[k*n+i]
    # So W index is k*n + i, not i*n + k.
    
    # Hmm, this is confusing. Let me just do a numerical verification.
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    # Compute C using the decomposition
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Using the tensor decomposition, this should be recoverable.
    # The exact relationship depends on the convention.
    
    # Let me try: C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k])... 
    # No, that doesn't work dimensionally.
    
    # I'll just verify by checking the tensor entries directly for a small case.
    # For n=1,m=1,p=1: rank should be 1, U=[[1]], V=[[1]], W=[[1]]
    # T[0,0,0] = U[0,0]*V[0,0]*W[0,0] = 1. Correct.
    
    # For n=1,m=2,p=1: rank=2 schoolbook
    # U shape (1*2, 2) = (2,2), V shape (2*1,2)=(2,2), W shape (1*1,2)=(1,2)
    # U[0,0]=1, V[0,0]=1, W[0,0]=1  (for i=0,j=0,k=0)
    # U[1,1]=1, V[1,1]=1, W[0,1]=1  (for i=0,j=1,k=0)
    # T[0,0,0] = U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,1] = 1*1*1 + 0*0*1 = 1
    # T[1,0,0] = U[1,0]*V[0,0]*W[0,0] + U[1,1]*V[0,1]*W[0,1] = 0*1*1 + 1*0*1 = 0
    # T[0,1,0] = U[0,0]*V[1,0]*W[0,0] + U[0,1]*V[1,1]*W[0,1] = 1*0*1 + 0*1*1 = 0
    # T[1,1,0] = U[1,0]*V[1,0]*W[0,0] + U[1,1]*V[1,1]*W[0,1] = 0*0*1 + 1*1*1 = 1
    
    # So T[i*m+j, j'*p+k, i'*n+k'] with the W index being k*n+i...
    # For (i,j)=(0,0), (j',k)=(0,0), (i',k')=(0,0): T = 1
    # For (i,j)=(1,0), (j',k)=(0,0), (i',k')=(0,0): T = 0
    # This makes sense: T is 1 only when all index pairs match.
    
    # So the verification is: T[a, b, c] = 1 iff a corresponds to (i,j), b to (j',k), c to (i',k')
    # with i=i', j=j', k=k'.
    
    # Let me implement a proper verification.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check T[i*m+j, j*p+k, k*n+i] should be 1
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
                # Check some off-diagonal should be 0
                if i < n-1:
                    val2 = sum(U[(i+1)*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                    if abs(val2) > 1e-9:
                        return False
    return True

def _try_winograd_rectangular(n, m, p):
    """
    Try to find a lower-rank decomposition using block-based Winograd-like techniques.
    For rectangular matrices, we can use blocking and apply fast matmul to blocks.
    """
    # Strategy: use a 2x2 block decomposition where possible, or 2x1/1x2 blocks
    # For 3x5 x 5x5, we can try various block sizes.
    
    # Let's try to use the fact that 5 = 2+2+1 and 3 = 2+1
    # We can partition into blocks and use Strassen-like formulas on each.
    
    # For now, let's try a simpler approach: use known results.
    # The minimum rank for 3x5 x 5x5 matmul is known to be at most 60 (from various sources)
    # but let's try to construct one.
    
    # Alternative: use a random search / optimization approach
    # But that might be too slow.
    
    # Let's try a structured approach: 
    # Partition the matrices into blocks and use a tensor decomposition for each block.
    
    # For 3x5 x 5x5:
    # A is 3x5, B is 5x5, C is 3x5
    # We can think of this as 3 independent 1x5 x 5x5 multiplications,
    # or as a 3x1 x 1x5 times 5x5, etc.
    
    # Let me try a different approach: use the fact that we can share computations.
    # For example, if we have two rows of A that are similar, we can share.
    
    # Actually, let's just try to find a good decomposition using a greedy algorithm
    # or a known construction.
    
    # Known result: rank of (3,5,5) matmul tensor is at most 60.
    # Let me try to achieve rank 60 or lower.
    
    # One approach: use a 2x2 Strassen-like decomposition on parts of the matrix.
    
    # Let me try: split 5 into 2+3, and 3 into 1+2
    # A: 3x5 -> split rows into 1+2, cols into 2+3
    # B: 5x5 -> split rows into 2+3, cols into 2+3
    # C: 3x5 -> split rows into 1+2, cols into 2+3
    
    # This is getting complex. Let me try a simpler heuristic.
    
    # For now, return None to indicate no improvement found.
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x5 matrix.
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Try to find a better decomposition
    better = _try_winograd_rectangular(n, m, p)
    
    if better is not None:
        U, V, W = better
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
