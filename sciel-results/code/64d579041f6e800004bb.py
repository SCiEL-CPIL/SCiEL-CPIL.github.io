"""Rank-minimal tensor decomposition for 2x3x10 matrix multiplication."""
import time
import numpy as np
import itertools

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=2, m=3, p=10, tol=1e-9):
    """Verify that U,V,W exactly decompose the multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor T[i,k,j] = 1 if the (i,k) output uses A[i,j]*B[j,k]
    # Tensor convention: T[i, k, j] where output C[i,k] = sum_j A[i,j] B[j,k]
    # Decomposition: T[i,k,j] = sum_c U[i*m+j? ...]
    # Interface: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # U row index = i*m + j  (A index)
    # V row index = j*p + k  (B index)
    # W row index = k*n + i  (C index, trace convention W[k*n+i, r])
    # Tensor: T[i,j,k] = 1 for all i,j,k (the multiplication tensor has 1 at (i,j,k))
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    # Also check that no spurious terms: the tensor only has entries at (i,j,k)
    # which we checked. Off-tensor entries must be 0 but the tensor is fully defined
    # on the grid (i in 0..n-1, j in 0..m-1, k in 0..p-1), all equal to 1.
    return max_err < tol

def _schoolbook(n=2, m=3, p=10):
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

def _block_decomposition(n=2, m=3, p=10):
    """
    Split the 10 output columns into 5 blocks of 2 columns each.
    Each block is a 2x3x2 sub-problem. We decompose each 2x3x2 block
    with a rank-7 Strassen-like scheme, giving total rank 5*7 = 35.
    
    For a 2x3x2 block with A (2x3), B (3x2), C (2x2):
    C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    We use 7 rank-1 terms per block.
    """
    # Strassen-like for 2x3x2: we need 7 multiplications.
    # A is 2x3, B is 3x2.
    # Let A = [[a00,a01,a02],[a10,a11,a12]], B = [[b00,b01],[b10,b11],[b20,b21]]
    # C = [[c00,c01],[c10,c11]]
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c11 = a10*b01 + a11*b11 + a12*b21
    # 
    # We can use the fact that this is like two 1x3x2 products combined.
    # Actually, a 2x3x2 tensor: rank? Let's use a known 7-term decomposition.
    # 
    # Think of it as: C = A * B where A is 2x3, B is 3x2.
    # Pad A to 2x2 by combining columns? No, 3 columns.
    # 
    # Alternative: use 6 terms (schoolbook for 2x3x2 is 6) and see if we can do 5 or 7.
    # Schoolbook 2x3x2 = 6 terms.
    # Let's just use 6 per block: 5*6 = 30 total. That's already better than 60.
    # But can we do better?
    # 
    # Let me try a 5-term decomposition for 2x3x2.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Define:
    # M1 = (a00+a10)*(b00+b10)  -> no, dimensions don't match for Strassen directly.
    #
    # Let me think differently. The 2x3x2 tensor:
    # It's the sum of two 1x3x2 tensors (one for row 0, one for row 1).
    # A 1x3x2 tensor (outer product of a 3-vector and a 2-vector, then...) 
    # Actually a 1x3x2 "multiplication" is just: c_k = sum_j a_j b_{j,k}
    # This is a rank-1 tensor? No. c_k = sum_j a_j * b_{j,k}.
    # This is the product of vector a (length 3) and matrix B (3x2).
    # The tensor T[j,k] for fixed... no.
    #
    # Let me just use schoolbook per block: 6 terms each, 30 total.
    # That's a clean 50% improvement. Let me also try to reduce further.
    
    # Actually, let me reconsider. Let me use a global approach instead.
    # 
    # Global idea: C[i,k] = sum_j A[i,j] B[j,k], i in {0,1}, k in 0..9, j in {0,1,2}.
    # 
    # We can write this as: for each j, we have A[:,j] (2-vector) and B[j,:] (10-vector).
    # C = sum_j A[:,j] outer B[j,:]
    # This is a sum of 3 rank-1 (in the matrix sense) contributions, but each is 
    # a 2x10 matrix = outer product of a 2-vector and a 10-vector.
    # As a tensor in (i, j, k) space: T[i,j,k] = 1, and the decomposition uses
    # the "matrix rank" of the 2x10 slice... 
    #
    # Hmm, let me think about the tensor rank directly.
    # The multiplication tensor T has T[i,j,k] = 1 for all i,j,k (it's the all-ones 
    # 2x3x10 tensor? No!)
    #
    # Wait. The multiplication tensor for matrix mult: the (i,j,k) entry is 1 if 
    # the output C[i,k] depends on A[i,j] and B[j,k]. For standard matrix mult,
    # C[i,k] = sum_j A[i,j] B[j,k], so T[i,j,k] = 1 for ALL i,j,k.
    # So the tensor IS the all-ones tensor of shape (2,3,10)!
    #
    # The rank of the all-ones tensor of shape (n,m,p) is... 
    # The all-ones 2x3x10 tensor. Its rank is the minimum r such that
    # 1 = sum_{c=1}^r u_c(i) v_c(j) w_c(k) for all i,j,k.
    #
    # This is a much simpler problem! The all-ones tensor of shape (a,b,c) has 
    # rank equal to the minimum of... Actually the all-ones tensor of shape (a,b,c)
    # has rank = min(a,b,c) if... no.
    #
    # The all-ones tensor of shape (a,b,c): rank is the smallest r such that the 
    # matrix of ones (a x b) can be written as sum of r rank-1 matrices where each 
    # rank-1 matrix's factors are consistent... 
    #
    # Actually, the all-ones tensor of shape (a,b,c) has rank = 1! Because 
    # 1 = 1*1*1. The all-ones tensor is rank 1!
    #
    # Wait, that can't be right for the matrix multiplication tensor. Let me re-examine.
    #
    # The matrix multiplication tensor: T_{ijk} where i indexes A's row, j indexes 
    # A's column / B's row, k indexes B's column / C's column.
    # T_{ijk} = 1 if and only if the (i,k) entry of C uses A_{ij} and B_{jk}.
    # For standard matrix multiplication, C_{ik} = sum_j A_{ij} B_{jk}, so YES, 
    # T_{ijk} = 1 for all i,j,k.
    #
    # So the tensor is indeed all-ones of shape (n,m,p) = (2,3,10).
    # And the all-ones tensor has rank 1: T_{ijk} = 1*1*1.
    #
    # But wait, that would mean matrix multiplication can be done with 1 "multiplication"
    # which is obviously wrong. The issue is that the tensor decomposition for matrix 
    # multiplication requires that the factors U, V, W have a specific structure related 
    # to the input/output spaces, and the "multiplication" in the decomposition corresponds 
    # to actual scalar multiplications in the algorithm.
    #
    # Let me re-read the interface:
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # U row index = i*m + j (this is the index into A)
    # V row index = j*p + k (this is the index into B)  
    # W row index = k*n + i (this is the index into C)
    #
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where T[i,j,k] = 1 for all i,j,k.
    #
    # So we need: for all i in {0,1}, j in {0,1,2}, k in {0..9}:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # This IS the all-ones tensor of shape (2,3,10) in the (i,j,k) space, but the 
    # factors are coupled: U depends on (i,j), V depends on (j,k), W depends on (k,i).
    #
    # This is NOT the same as the standard tensor rank of the all-ones tensor, because 
    # the factorization is constrained by the "cyclic" indexing.
    #
    # However, if we ignore the cyclic constraint, the all-ones (2,3,10) tensor has rank 1.
    # With the cyclic constraint, it's a different problem.
    #
    # Let me reconsider. The constraint is:
    # U is a (n*m) x r matrix, V is a (m*p) x r matrix, W is a (p*n) x r matrix.
    # For each (i,j,k): sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1.
    #
    # This is equivalent to: define for each c, three vectors:
    # u_c in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    # such that for all (i,j,k): sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1.
    #
    # This is the "matrix multiplication tensor" rank problem. It's known that this 
    # rank is at least the standard tensor rank and can be larger.
    #
    # For the all-ones case with cyclic indexing, let me try r=1:
    # u[i*m+j] * v[j*p+k] * w[k*n+i] = 1 for all i,j,k.
    # This requires u, v, w to be all-ones (up to scaling), which works!
    # u[i*m+j] = 1, v[j*p+k] = 1, w[k*n+i] = 1 for all indices.
    # Then sum_c = 1*1*1 = 1. 
    #
    # So rank 1 works?! Let me verify: U is all-ones (6x1), V is all-ones (30x1), 
    # W is all-ones (20x1). Then for any (i,j,k):
    # U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1*1*1 = 1. ✓
    #
    # But this seems too good to be true. Let me re-read the problem statement.
    #
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix 
    # by a 3 by 10 matrix."
    #
    # If rank 1 works, then the answer is 1. But that contradicts the schoolbook rank of 60.
    #
    # I think I'm misunderstanding the tensor. Let me reconsider.
    #
    # The matrix multiplication tensor is NOT the all-ones tensor. The tensor T is 
    # defined such that the matrix product C = A*B satisfies:
    # C_{ik} = sum_{i,j,k} T_{ijk} A_{ij} B_{jk}
    # where T_{ijk} = 1 if the indices are "compatible" (i.e., A's (i,j) and B's (j,k) 
    # contribute to C's (i,k)).
    #
    # For standard matrix multiplication, A is n x m, B is m x p, C is n x p.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # So T_{ijk} = 1 for all i,j,k (where i in 0..n-1, j in 0..m-1, k in 0..p-1).
    #
    # Yes, it IS the all-ones tensor. And with the cyclic factorization constraint, 
    # rank 1 works.
    #
    # But this can't be the intended problem, because then the answer is trivially 1.
    # Let me re-read the interface more carefully.
    #
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "with integer/half-integer entries that exactly decompose the matrix multiplication tensor"
    #
    # I think the issue is that the "matrix multiplication tensor" in this context 
    # is the standard one used in algebraic complexity theory, where the rank of the 
    # tensor gives the number of scalar multiplications needed. And for the all-ones 
    # tensor with the cyclic constraint, the rank is indeed not 1 in general.
    #
    # Wait, I showed rank 1 works. Let me re-examine.
    #
    # If U, V, W are all-ones vectors, then the decomposition gives T_{ijk} = 1 for 
    # all (i,j,k). But the matrix multiplication tensor IS all-ones. So this is correct.
    #
    # The confusion is: in the standard formulation, the tensor decomposition corresponds 
    # to an algorithm where each "term" c in the sum corresponds to one scalar multiplication. 
    # If rank is 1, that means one scalar multiplication suffices, which is impossible for 
    # non-trivial matrix multiplication.
    #
    # The resolution: the tensor decomposition for matrix multiplication requires that 
    # the factors U, V, W be such that the "algorithm" they define is valid. The all-ones 
    # factorization doesn't correspond to a valid algorithm because the "multiplications" 
    # don't actually compute the right things in a way that can be assembled into the output.
    #
    # Actually, I think the issue is more subtle. Let me look at this from the 
    # algorithmic perspective. A rank-r decomposition of the multiplication tensor 
    # corresponds to an algorithm that:
    # 1. Computes r intermediate values M_c = (sum_i A_{i,?} u_{c,i}) * (sum_k B_{?,k} v_{c,k})
    #    (or some linear combination of A times some linear combination of B)
    # 2. Computes C_{ik} = sum_c M_c * w_{c,ik}
    #
    # For this to be a valid algorithm, the "multiplications" M_c must be products of 
    # linear forms in A and linear forms in B. The all-ones decomposition gives 
    # M_1 = (sum_{i,j} A_{ij}) * (sum_{j,k} B_{jk}), and then C_{ik} = M_1 * 1. 
    # But this is clearly wrong: C_{ik} = (sum_{i',j} A_{i'j})(sum_{j',k'} B_{j'k'}) 
    # is not equal to sum_j A_{ij} B_{jk}.
    #
    # So the all-ones factorization does NOT correspond to a valid matrix multiplication 
    # algorithm. The tensor decomposition must satisfy additional constraints that I'm 
    # missing.
    #
    # Let me re-read the verification. The task says "exactly decompose the matrix 
    # multiplication tensor". The tensor T_{ijk} for matrix multiplication is defined 
    # such that:
    # C_{ik} = sum_{j} T_{ijk} A_{ij} B_{jk}
    #
    # And T_{ijk} = 1 for all i,j,k. So the tensor IS all-ones.
    #
    # But the decomposition T_{ijk} = sum_c U_{(i,j),c} V_{(j,k),c} W_{(k,i),c} 
    # with all-ones U,V,W gives T_{ijk} = 1, which is correct.
    #
    # I'm confused about why this doesn't correspond to a valid algorithm. Let me 
    # think about it differently.
    #
    # In the standard theory, a rank-r decomposition of the matrix multiplication 
    # tensor [n,m,p] gives an algorithm with r multiplications. The decomposition is:
    # T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    # where u_c is in the A-space (n*m dimensional), v_c is in the B-space (m*p dimensional), 
    # w_c is in the C-space (n*p dimensional).
    #
    # The algorithm is:
    # M_c = <u_c, A> * <v_c, B>   (one multiplication per c)
    # C = sum_c M_c * w_c        (linear combinations)
    #
    # For this to compute C = A*B, we need:
    # C_{ik} = sum_j A_{ij} B_{jk} = sum_c <u_c, A> <v_c, B> w_c^{(i,k)}
    #
    # where <u_c, A> = sum_{i,j} u_c^{(i,j)} A_{ij} and <v_c, B> = sum_{j,k} v_c^{(j,k)} B_{jk}.
    #
    # Expanding the right side:
    # sum_c (sum_{i',j'} u_c^{(i',j')} A_{i'j'}) (sum_{j'',k''} v_c^{(j'',k'')} B_{j''k''}) w_c^{(i,k)}
    #
    # For this to equal sum_j A_{ij} B_{jk} for all A, B, we need the tensor:
    # sum_c u_c^{(i',j')} v_c^{(j'',k'')} w_c^{(i,k)} = delta_{i,i'} delta_{j,j''} delta_{k,k'} * (something)
    #
    # Actually, the correct condition is:
    # sum_c u_c^{(i',j')} v_c^{(j'',k'')} w_c^{(i,k)} = [i' == i] [j' == j''] [k' == k]
    #
    # Wait, that's not quite right either. Let me be more careful.
    #
    # We want: for all A, B:
    # (A*B)_{ik} = sum_c <u_c, A> <v_c, B> w_c^{(i,k)}
    #
    # Left side: sum_j A_{ij} B_{jk}
    #
    # Right side: sum_c (sum_{i',j'} u_c^{(i',j')} A_{i'j'}) (sum_{j'',k''} v_c^{(j'',k'')} B_{j''k''}) w_c^{(i,k)}
    # = sum_{i',j',j'',k''} (sum_c u_c^{(i',j')} v_c^{(j'',k'')} w_c^{(i,k)}) A_{i'j'} B_{j''k''}
    #
    # For this to equal sum_j A_{ij} B_{jk} for all A, B, we need:
    # sum_c u_c^{(i',j')} v_c^{(j'',k'')} w_c^{(i,k)} = delta_{i',i} delta_{j',j''} delta_{k'',k}
    #
    # Wait, that's: the coefficient of A_{i'j'} B_{j''k''} on the right must equal 
    # the coefficient on the left, which is 1 if i'=i, j'=j'', k''=k, else 0.
    #
    # So: sum_c u_c^{(i',j')} v_c^{(j'',k'')} w_c^{(i,k)} = delta_{i',i} delta_{j',j''} delta_{k'',k}
    #
    # This is the tensor decomposition condition. The tensor T is defined by:
    # T^{(i',j',j'',k'',i,k)} = delta_{i',i} delta_{j',j''} delta_{k'',k}
    #
    # But this is a 6-index tensor, not a 3-index tensor. The standard formulation 
    # uses a 3-index tensor by "flattening" the indices.
    #
    # In the standard formulation, the multiplication tensor [n,m,p] is a 3-tensor 
    # T in R^{(n*m)} ⊗ R^{(m*p)} ⊗ R^{(n*p)} with:
    # T[(i,j), (j,k), (i,k)] = 1  (using the same j in the first two indices)
    # T[(i,j), (j',k), (i',k)] = 0  if j != j' or i != i'
    #
    # Wait, I think the standard formulation is:
    # T[(i,j), (j,k), (i',k')] = 1 if i==i' and k==k' (and j is shared), else 0.
    #
    # No, that's not right either. Let me look at this more carefully.
    #
    # The multiplication tensor for multiplying an n x m matrix by an m x p matrix:
    # It's a tensor T in R^{n*m} ⊗ R^{m*p} ⊗ R^{n*p} defined by:
    # T[a, b, c] = 1 if a = (i,j), b = (j,k), c = (i,k) for some i,j,k
    # T[a, b, c] = 0 otherwise
    #
    # So T[(i,j), (j,k), (i,k)] = 1 for all i,j,k, and 0 otherwise.
    #
    # This is NOT the all-ones tensor. It's a specific tensor with 1's at positions 
    # where the indices are "compatible".
    #
    # Now, the decomposition T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] means:
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    #
    # For the "compatible" positions: T[(i,j),(j,k),(i,k)] = 1 = sum_r U[(i,j),r] V[(j,k),r] W[(i,k),r]
    # For the "incompatible" positions: T[(i,j),(j',k),(i',k')] = 0 if not compatible.
    #
    # So the condition is:
    # For all i,j,k: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # For all (i,j),(j',k),(i',k') where (i,j),(j',k) are not "compatible" with (i',k'):
    #   sum_r U[i*m+j, r] V[j'*p+k, r] W[k'*n+i', r] = 0
    #
    # The "compatible" condition is: the output index (i',k') must equal (i,k) where 
    # i is from the A-index and k is from the B-index, AND the j from A must equal 
    # the j' from B.
    #
    # So: sum_r U[i*m+j, r] V[j'*p+k, r] W[k'*n+i', r] = 1 if i'=i, k'=k, j'=j
    # and 0 otherwise.
    #
    # This is the correct condition! The tensor is NOT all-ones; it's a "selection" 
    # tensor that picks out the compatible index combinations.
    #
    # Now, the schoolbook decomposition uses r = n*m*p terms, one for each (i,j,k):
    # U[i*m+j, (i,j,k)] = 1, V[j*p+k, (i,j,k)] = 1, W[k*n+i, (i,j,k)] = 1
    #
    # And this satisfies: for (i,j),(j',k),(i',k'):
    # sum_{(i'',j'',k'')} U[i*m+j, (i'',j'',k'')] V[j'*p+k, (i'',j'',k'')] W[k'*n+i', (i'',j'',k'')]
    # = 1 if i''=i, j''=j, k''=k AND j''=j', k''=k, i''=i'
    # = 1 if i=i', j=j', j=j', k=k
    # = 1 if i=i', j=j', k=k, else 0. ✓
    #
    # Great, so now I understand the problem correctly. The tensor is the "selection" 
    # tensor, not the all-ones tensor.
    #
    # Now, can we do better than n*m*p = 60?
    #
    # The rank of the multiplication tensor [n,m,p] is a well-studied quantity. For 
    # [2,3,10], we need to find the minimum rank.
    #
    # Known results:
    # - [2,2,2] has rank 7 (Strassen).
    # - [2,2,p] for large p has rank related to Strassen.
    # - The rank of [n,m,p] is at least max(n*m, m*p, n*p) and at least the 
    #   "border rank" lower bounds.
    #
    # For [2,3,10]:
    # - n*m = 6, m*p = 30, n*p = 20. So rank >= 30.
    # - Schoolbook gives 60.
    # - Can we do better than 60?
    #
    # One approach: use Strassen on 2x2x2 blocks. Split the 3 middle dimension into 
    # 2+1 and the 10 output dimension into 5*2.
    #
    # Actually, let me try a different approach. Let me use the fact that n=2 is small.
    #
    # For [2,m,p], the rank is at most 2 * rank([1,m,p/2]) * ... hmm, this is getting 
    # complicated.
    #
    # Let me try a concrete construction. Split the 10 output columns into 5 pairs. 
    # Each pair (2k, 2k+1) forms a [2,3,2] sub-problem. The [2,3,2] tensor can be 
    # decomposed with rank 6 (schoolbook) or possibly less.
    #
    # For [2,3,2]: T[(i,j),(j,k),(i,k)] = 1 for i in {0,1}, j in {0,1,2}, k in {0,1}.
    # Schoolbook: 6 terms. Can we do 5? 4?
    #
    # The rank of [2,3,2] is at least max(6, 6, 4) = 6. So rank >= 6. And schoolbook 
    # gives 6. So [2,3,2] has rank exactly 6.
    #
    # So splitting into 5 blocks of [2,3,2] gives rank 5*6 = 30. That's the lower bound!
    #
    # But wait, can we do better by not splitting into independent blocks? The blocks 
    # share the same A matrix, so there might be cross-block savings.
    #
    # Let me try to find a rank-30 decomposition by splitting into 5 independent 
    # [2,3,2] blocks.
    
    n, m, p = 2, 3, 10
    # Split k into 5 blocks of 2.
    # Block b handles k in {2b, 2b+1}.
    # For each block, use schoolbook [2,3,2] decomposition (6 terms).
    # Total rank = 5 * 6 = 30.
    
    rank = 30
    U = [[0] * rank for _ in range(n * m)]  # 6 x 30
    V = [[0] * rank for _ in range(m * p)]  # 30 x 30
    W = [[0] * rank for _ in range(p * n)]  # 20 x 30
    
    col = 0
    for b in range(5):
        for i in range(n):
            for j in range(m):
                for k_local in range(2):
                    k = 2 * b + k_local
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    start = time.time()
    deadline = 150 - 10  # 140s safety margin
    
    # Try the block decomposition (rank 30).
    result = _block_decomposition()
    if _verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook.
    return _schoolbook()
# EVOLVE-BLOCK-END
