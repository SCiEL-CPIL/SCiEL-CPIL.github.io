"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Dimensions: A is n x m (2x3), B is m x p (3x7)
    # Output C is n x p (2x7)
    
    # The multiplication tensor T has dimensions (n*m) x (m*p) x (p*n) ? No.
    # Standard tensor decomposition for matrix mult:
    # C = A * B
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor T_{i,j,k} = A_{i,j} B_{j,k}
    # The task specifies:
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    # Trace convention: W[k*n + i, r]
    
    # The tensor being decomposed is the structure tensor of matrix multiplication.
    # Usually, we look for rank R such that:
    # T_{i,j,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)} ?
    # Let's check the interface carefully.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # The product sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal the tensor entry.
    # For matrix multiplication, the tensor entry T_{i,j,k} is 1 if the indices correspond to a valid multiplication term, 0 otherwise.
    # Specifically, T_{i,j,k} = 1 for all i, j, k? No.
    # The tensor for matrix multiplication M_{n,m,p} is defined on indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    # The value is 1 if the operation is defined, which it is for all i,j,k in the standard bilinear form.
    # Wait, the tensor is T_{i,j,k} = 1 for all i,j,k?
    # If T_{i,j,k} = 1, then we are decomposing the all-ones tensor of size n*m*p?
    # No, that's not matrix multiplication.
    # Matrix multiplication tensor is T_{i,j,k} = 1.
    # Let's re-read standard definitions.
    # The structure tensor of matrix multiplication M_{n,m,p} is an element of (F^n)^* \otimes (F^m)^* \otimes (F^p) ... no.
    # It is in (F^{n \times m})^* \otimes (F^{m \times p})^* \otimes (F^{n \times p})?
    # Usually, the tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k.
    # So T_{i,j,k} = 1 for all i,j,k.
    
    # If T_{i,j,k} = 1 for all i,j,k, then we are looking for the rank of the all-ones tensor of dimensions n x m x p.
    # The rank of the all-ones tensor of size a x b x c is min(a,b,c) * ...?
    # Actually, the rank of the all-ones tensor T_{i,j,k}=1 is 1? No.
    # T_{i,j,k} = 1 * 1 * 1. This is a rank 1 tensor!
    # U_{i,j} = 1, V_{j,k} = 1, W_{k,i} = 1.
    # Let's check: sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}.
    # If rank=1, U_{i,j}=1, V_{j,k}=1, W_{k,i}=1.
    # Sum is 1. This matches T_{i,j,k}=1.
    
    # BUT, is the task about the all-ones tensor?
    # "Matrix multiplication 2,3,7"
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix."
    # The tensor associated with matrix multiplication is indeed the one with 1s everywhere in the index space (i,j,k).
    # The bilinear form is C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor T is defined such that T(A,B) = A*B.
    # In coordinates, T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # Since C_{i,k} depends on A_{i,j} B_{j,k} with coefficient 1, T_{i,j,k} = 1 for all i,j,k.
    
    # So the problem is to find the rank of the tensor T_{i,j,k} = 1 for i in [0,2), j in [0,3), k in [0,7).
    # This tensor is simply 1 * 1 * 1.
    # Rank is 1?
    
    # Let's verify the interface indices again.
    # U: (n*m, rank) -> index i*m+j
    # V: (m*p, rank) -> index j*p+k
    # W: (p*n, rank) -> index k*n+i
    #
    # If I set rank=1:
    # U[i*m+j, 0] = 1
    # V[j*p+k, 0] = 1
    # W[k*n+i, 0] = 1
    #
    # Then sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1*1*1 = 1.
    # This exactly decomposes the tensor T_{i,j,k}=1.
    
    # Is it that simple?
    # Why would the seed program use rank n*m*p?
    # The seed program uses the "Schoolbook decomposition".
    # In schoolbook, each term A_{i,j} B_{j,k} is a separate rank-1 term.
    # There are n*m*p such terms.
    # But they are not independent in the tensor sense if we can combine them?
    # No, the tensor T_{i,j,k}=1 is rank 1.
    # 
    # Wait. Is the tensor T_{i,j,k}=1?
    # Let's check a small case. 1x1x1 matrix multiplication.
    # A (1x1), B (1x1), C (1x1).
    # C_00 = A_00 B_00.
    # Tensor T_{0,0,0} = 1.
    # Rank 1.
    #
    # 1x2x1 matrix multiplication.
    # A (1x2), B (2x1), C (1x1).
    # C_00 = A_00 B_00 + A_01 B_10.
    # Tensor indices: i=0, j in {0,1}, k=0.
    # T_{0,0,0} = 1 (coeff of A_00 B_00)
    # T_{0,1,0} = 1 (coeff of A_01 B_10)
    # T_{0,0,1} = 0 (B has no col 1)
    # T_{0,1,1} = 0
    #
    # Wait, the tensor dimensions are determined by the matrix shapes.
    # A is n x m. B is m x p. C is n x p.
    # The tensor lives in space of size (n*m) x (m*p) x (n*p)?
    # No, the standard tensor for matrix multiplication M_{n,m,p} is in (F^n \otimes F^m)^* \otimes (F^m \otimes F^p)^* \otimes (F^n \otimes F^p).
    # This is a 3-tensor with dimensions (nm) x (mp) x (np).
    # The entries are T_{(i,j), (j',k), (i',k')}.
    # T_{(i,j), (j',k), (i',k')} = 1 if i=i', j=j', k=k'. Otherwise 0.
    # 
    # Let's re-read the interface.
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # 
    # The index for W is k*n + i. This corresponds to (k, i) or (i, k)?
    # "W[k*n+i, r]". If n=2, k=0, i=1 -> index 1. k=1, i=0 -> index 2.
    # This maps (k, i) to a linear index.
    # Usually C is n x p. Index is i*p + k or k*n + i?
    # If C is stored row-major (i, k), index is i*p + k.
    # If C is stored column-major (k, i), index is k*n + i.
    # The prompt says "output factor follows the official trace convention W[k*n+i,r]".
    # This implies the third mode of the tensor is indexed by (k, i).
    #
    # So the tensor T has entries T_{u, v, w} where:
    # u = i*m + j
    # v = j'*p + k
    # w = k'*n + i'
    # 
    # T_{u, v, w} = 1 if (i,j) from u, (j',k) from v, (i',k') from w satisfy:
    # i = i', j = j', k = k'.
    # Otherwise 0.
    #
    # This is the standard structure tensor of matrix multiplication.
    # It is NOT the all-ones tensor. It is a sparse tensor with 1s on the "diagonal" defined by index matching.
    #
    # Number of non-zero entries:
    # For each i, j, k, there is exactly one non-zero entry:
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    # 
    # So the tensor has n*m*p non-zero entries, each being 1.
    #
    # The rank of this tensor is the minimum number of rank-1 tensors needed to sum to it.
    # This is the bilinear complexity of matrix multiplication, denoted M(n,m,p).
    #
    # For n=2, m=3, p=7.
    # We need to find the rank of this specific tensor.
    #
    # Known bounds for M(2,3,7)?
    # General bounds: M(n,m,p) <= n*m*p.
    # Strassen: M(2,2,2) = 7.
    # 
    # This is a hard problem. Exact rank is rarely known for arbitrary sizes.
    # However, the task asks for a "rank-minimal exact tensor decomposition".
    # If we don't know the minimal rank, we must find a valid decomposition with low rank.
    # The seed uses rank n*m*p = 42.
    #
    # Can we do better than 42?
    # For 2x2x2, rank is 7 (vs 8).
    # For 2x3x7, is there a known algorithm?
    # 
    # Strategy:
    # 1. Construct the tensor T as a 3D array or a set of non-zero coordinates.
    # 2. Use a rank-revealing algorithm or heuristic to find a low-rank decomposition.
    # 3. Since exact rational/integer solutions are required, we might need to solve a linear system.
    #
    # However, finding the exact minimal rank is NP-hard.
    # We should probably stick to a known good algorithm or a heuristic that finds a valid decomposition.
    #
    # Let's try to construct a decomposition using the "block" structure or known algorithms.
    # If no specific algorithm for 2x3x7 is known, we can try to use the Strassen-like approach or just improve on schoolbook.
    #
    # Actually, for small matrices, we can use a solver to find a decomposition of a specific rank R.
    # We can iterate R from lower bound to upper bound.
    # Lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21.
    # Upper bound: 42.
    #
    # We can try to find a decomposition of rank R by solving:
    # T = sum_{r=1}^R U^{(r)} \otimes V^{(r)} \otimes W^{(r)}
    #
    # This is equivalent to finding matrices U, V, W of sizes (nm x R), (mp x R), (pn x R) such that
    # T_{u,v,w} = sum_{r=1}^R U_{u,r} V_{v,r} W_{w,r}.
    #
    # This is a system of polynomial equations.
    # We can linearize it by fixing two of the matrices and solving for the third, or using alternating least squares (ALS) but with exact arithmetic?
    # ALS with floats might not yield exact integer/half-integer results.
    #
    # Given the constraints (120s limit, exactness), and the difficulty of finding minimal rank,
    # maybe the best strategy is to use a known efficient algorithm for small matrices or a heuristic search.
    #
    # However, the directive says: "Model the tensor decomposition as a linear code problem... applying sparse linear algebra... to find a minimal set of rank-1 components that span the tensor's support".
    # This suggests looking for a basis of the tensor space? No, the tensor is fixed.
    # Maybe it means: represent the tensor as a vector in a high-dimensional space and find a low-rank approximation?
    #
    # Let's reconsider the "linear code" hint.
    # Perhaps it implies using the fact that the tensor is sparse and using Gaussian elimination on the associated matricization?
    # Matricization of T along mode 1: T_(1) is (nm) x (mp * pn).
    # Rank of T_(1) is an upper bound on the tensor rank.
    # Rank of T_(1) is at most min(nm, mp*pn).
    # For n=2,m=3,p=7: nm=6, mp*pn = 21*14 = 294.
    # Rank of T_(1) is likely 6.
    # This doesn't help reduce rank below 21.
    #
    # Let's try a different approach.
    # We can use a randomized algorithm or a greedy approach to find a valid decomposition with rank < 42.
    # Or, we can simply return the schoolbook decomposition (rank 42) if we can't find better quickly.
    # But the seed already does that and scores 0.0000?
    # Wait, the seed scores 0.0000. Why?
    # "score is the rank". "evaluator measures the raw value (lower is better)".
    # If the seed returns rank 42, the score should be 42.
    # Why is it 0.0000?
    # "an invalid result, an exception or a timeout scores 0".
    # Maybe the seed program is invalid?
    # Let's check the seed code.
    # U, V, W are lists of lists.
    # The return is a dict.
    # Maybe the evaluator expects numpy arrays?
    # "factor matrices of shapes ... with integer/half-integer entries".
    # Lists of integers should be fine.
    #
    # Let's check the index mapping in the seed.
    # U[i*m+j, column] = 1
    # V[j*p+k, column] = 1
    # W[k*n+i, column] = 1
    #
    # For a specific term (i,j,k):
    # u = i*m+j
    # v = j*p+k
    # w = k*n+i
    #
    # This matches the tensor definition T_{u,v,w}=1 if indices match.
    # So the seed is a valid decomposition of rank 42.
    # Why score 0?
    # Maybe the evaluator crashes on lists?
    # Let's convert to numpy arrays to be safe.
    #
    # Also, maybe the rank can be reduced.
    # Let's try to find a decomposition with rank < 42.
    #
    # For 2x2x2, rank is 7.
    # For 2x3x7, we can try to embed the 2x2x2 Strassen algorithm?
    # Or use a block algorithm.
    #
    # Let's try to use a solver to find a decomposition of rank R.
    # We can start with R=21 (lower bound) and increase.
    # Solving for exact decomposition is hard.
    #
    # Alternative: Use the "linear code" hint.
    # Maybe the hint implies that we should treat the columns of U, V, W as codewords?
    #
    # Let's try a simple improvement:
    # Use numpy to check if a random decomposition works? No, must be exact.
    #
    # Let's stick to the schoolbook but ensure it's valid and fast.
    # If the seed is invalid due to type, converting to numpy might fix it.
    #
    # Let's also try to see if rank 41 is possible? Unlikely to be minimal.
    #
    # What if we use the Strassen algorithm for the 2x2 block and schoolbook for the rest?
    # 2x3x7 can be split?
    #
    # Let's just fix the potential type issue and return the schoolbook decomposition.
    # If that scores 0, then there is a deeper issue.
    # But I suspect the list vs numpy array issue or some other formatting.
    #
    # Let's write a robust construct() that returns numpy arrays.
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
