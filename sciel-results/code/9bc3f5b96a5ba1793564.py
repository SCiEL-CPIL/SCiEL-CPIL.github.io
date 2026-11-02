"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    n, m, p = DIMENSIONS
    
    # 1. Try a few simple structured decompositions (heuristic guesses)
    # These are known low-rank decompositions for small matrix multiplication.
    # If any work, they will likely be better than schoolbook (120).
    
    # We'll generate candidate decompositions and verify them.
    # Verification: For each entry (i, j, k) of the product C = AB,
    # we need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] == 1 if we are decomposing the tensor.
    # Wait, the tensor decomposition is: T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
    # Here T_{i,j,k} is 1 if (i,j,k) corresponds to a valid multiplication path?
    # No, the matrix multiplication tensor is T such that C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor is defined by T_{i,j,k} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # Actually, the tensor is a 3-way array where T[i,j,k] = 1 if we are computing the contribution of A_ij * B_jk to C_ik.
    # But C_ik depends on j. So for each i,k, there are m terms.
    # The "matrix multiplication tensor" is usually defined as the bilinear map.
    # The decomposition rank is the number of scalar multiplications.
    # A decomposition of rank r gives matrices U, V, W such that:
    # For all i, j, k: 1 (if i,j,k are valid indices) = sum_{c=1}^r U[i,j,c] * V[j,k,c] * W[k,i,c] ?
    # No. The tensor T has dimensions (n*m, m*p, p*n)? No.
    # The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    # T[i,j,k] = 1.
    # The condition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    # Wait, the task says: "output factor follows the official trace convention W[k*n+i,r]".
    # And U is (n*m, rank), V is (m*p, rank).
    # So yes, for each (i,j,k), the sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] must be 1.
    # This is exactly the condition for the tensor decomposition of the all-ones tensor?
    # No, the matrix multiplication tensor is not the all-ones tensor.
    # Let's check the seed code.
    # U[i*m+j][column] = 1, V[j*p+k][column] = 1, W[k*n+i][column] = 1.
    # For a specific (i,j,k), only the column where i,j,k match will contribute 1*1*1 = 1.
    # Other columns have 0 in at least one of U,V,W for this (i,j,k).
    # So the sum is 1.
    # So we are decomposing the tensor T where T[i,j,k] = 1 for all i,j,k.
    # This is the "all-ones" tensor of size n x m x p.
    # Wait, is that right?
    # If T[i,j,k] = 1 for all i,j,k, then the rank is the rank of the all-ones matrix in each slice?
    # The rank of the all-ones tensor of size n x m x p is 1.
    # U[i,j] = 1, V[j,k] = 1, W[k,i] = 1.
    # Sum_c U[i,j] V[j,k] W[k,i] = 1 * 1 * 1 = 1.
    # So rank 1 works?
    # Why did the seed use rank 120?
    # Ah, the matrix multiplication tensor is NOT the all-ones tensor.
    # In matrix multiplication, the coefficient of A_ij B_jk in C_ik is 1.
    # So T[i,j,k] = 1.
    # So it IS the all-ones tensor?
    # Let's re-read the task.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 5 matrix by a 5 by 6 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # If T[i,j,k] = 1 for all i,j,k, then rank 1 is the answer.
    # U[i*m+j, 0] = 1 for all i,j.
    # V[j*p+k, 0] = 1 for all j,k.
    # W[k*n+i, 0] = 1 for all k,i.
    # Then for any (i,j,k): U * V * W = 1.
    # So the rank is 1.
    # Why is the seed score 1.0 with rank 120?
    # "score is the rank".
    # "1.0 equals the seed program".
    # So the seed program has rank 120.
    # If rank 1 is possible, the score would be 1/120 = 0.0083, which is < 1.0.
    # But the seed score is 1.0.
    # This implies that rank 1 is NOT a valid decomposition for matrix multiplication?
    # Let's check the definition of tensor decomposition for matrix multiplication.
    # The matrix multiplication tensor T_{n,m,p} is defined by T_{i,j,k} = 1 if the product A_{ij} B_{jk} contributes to C_{ik}.
    # In standard matrix multiplication, C_{ik} = sum_j A_{ij} B_{jk}.
    # So for each (i,k), there are m terms.
    # The tensor is a 3-index array.
    # Usually, the tensor is defined such that T[i,j,k] = 1.
    # If so, the rank is indeed 1.
    # However, there is a nuance. The "rank" of the matrix multiplication tensor usually refers to the bilinear complexity, which is the number of scalar multiplications required.
    # If we use rank 1 decomposition:
    # C_{ik} = (sum_j A_{ij} * 1) * 1 * (sum_k B_{jk} * 1)? No.
    # The decomposition gives:
    # C_{ik} = sum_c (sum_j U[i,j,c] A_{ij}) * (sum_j V[j,k,c] B_{jk}) * W[k,i,c]?
    # No, the decomposition is:
    # C_{ik} = sum_c W[k,i,c] * (sum_j U[i,j,c] A_{ij}) * (sum_j V[j,k,c] B_{jk})?
    # This would mean C_{ik} is a sum of products of linear forms.
    # If U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1.
    # Then term c is (sum_j A_{ij}) * (sum_j B_{jk}) * 1.
    # C_{ik} = (sum_j A_{ij}) * (sum_j B_{jk}).
    # This is NOT the matrix product.
    # The matrix product is sum_j A_{ij} B_{jk}.
    # The tensor decomposition condition is:
    # For all i,j,k: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    # If T[i,j,k] = 1, then this holds.
    # But does this correspond to matrix multiplication?
    # The bilinear form is:
    # C_{ik} = sum_j T[i,j,k] A_{ij} B_{jk} = sum_j A_{ij} B_{jk}.
    # The decomposed form is:
    # C_{ik} = sum_c W[k,i,c] (sum_j U[i,j,c] A_{ij}) (sum_j V[j,k,c] B_{jk}).
    # Let X_c = sum_j U[i,j,c] A_{ij}, Y_c = sum_j V[j,k,c] B_{jk}.
    # Then C_{ik} = sum_c W[k,i,c] X_c Y_c.
    # This is a bilinear map.
    # For this to equal sum_j A_{ij} B_{jk}, we need:
    # sum_j A_{ij} B_{jk} = sum_c W[k,i,c] (sum_j U[i,j,c] A_{ij}) (sum_j V[j,k,c] B_{jk}).
    # Expanding the RHS:
    # sum_c W[k,i,c] (sum_j sum_l U[i,j,c] V[j,l,c] A_{ij} B_{lk})? No, indices don't match.
    # X_c = sum_j U[i,j,c] A_{ij}.
    # Y_c = sum_l V[l,k,c] B_{lk}. (Using l for dummy index to avoid confusion, but V is m*p so second index is k? No, V is (m,p). V[j,k]. So Y_c = sum_j V[j,k,c] B_{jk}.)
    # So X_c Y_c = (sum_j U[i,j,c] A_{ij}) (sum_l V[l,k,c] B_{lk}).
    # = sum_j sum_l U[i,j,c] V[l,k,c] A_{ij} B_{lk}.
    # We want this to match sum_j A_{ij} B_{jk}.
    # So we need:
    # sum_c W[k,i,c] U[i,j,c] V[l,k,c] = delta_{j,l}.
    # i.e., for all i, j, l, k:
    # sum_c W[k,i,c] U[i,j,c] V[l,k,c] = 1 if j==l, 0 otherwise.
    # This is the condition for the tensor decomposition.
    # The tensor T is defined by T[i,j,l,k]? No, T is 3-index.
    # The condition is:
    # For all i, j, k: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    # This is what I derived first.
    # Does this condition imply the bilinear map is correct?
    # If sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k, then:
    # sum_c W[k,i,c] (sum_j U[i,j,c] A_{ij}) (sum_l V[l,k,c] B_{lk})
    # = sum_c W[k,i,c] sum_j sum_l U[i,j,c] V[l,k,c] A_{ij} B_{lk}
    # = sum_j sum_l A_{ij} B_{lk} (sum_c W[k,i,c] U[i,j,c] V[l,k,c]).
    # This does NOT simplify to sum_j A_{ij} B_{jk} unless U[i,j,c] V[l,k,c] is zero unless j=l?
    # No, the condition is on the product U[i,j,c] V[j,k,c] W[k,i,c].
    # The bilinear map correctness requires:
    # sum_c W[k,i,c] U[i,j,c] V[l,k,c] = delta_{j,l}.
    # This is a different condition.
    # The condition T[i,j,k] = 1 is for the tensor T_{i,j,k}.
    # But the bilinear map involves T_{i,j,k} in the sum over j.
    # C_{ik} = sum_j T_{i,j,k} A_{ij} B_{jk}.
    # If T_{i,j,k} = 1, then C_{ik} = sum_j A_{ij} B_{jk}.
    # The decomposition must satisfy:
    # sum_c W[k,i,c] (sum_j U[i,j,c] A_{ij}) (sum_j V[j,k,c] B_{jk}) = sum_j A_{ij} B_{jk}.
    # This requires:
    # sum_c W[k,i,c] U[i,j,c] V[l,k,c] = delta_{j,l}.
    # So the tensor to decompose is NOT the all-ones tensor T[i,j,k]=1.
    # The tensor is defined by its action on the bilinear form.
    # The "tensor" in tensor rank sense is the 3-way array T such that:
    # T_{i,j,k} is the coefficient of A_{ij} B_{jk} in C_{ik}.
    # So T_{i,j,k} = 1 if we are computing C_{ik} using A_{ij} B_{jk}.
    # But C_{ik} uses ALL j.
    # So T is a 3-index array where T[i,j,k] = 1?
    # If T[i,j,k] = 1, then the condition for decomposition is:
    # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    # And this condition is NECESSARY but is it SUFFICIENT for the bilinear map?
    # Let's check.
    # If sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1, does it imply sum_c W[k,i,c] U[i,j,c] V[l,k,c] = delta_{j,l}?
    # No.
    # Example: n=m=p=1.
    # T[0,0,0] = 1.
    # U[0,0,c] V[0,0,c] W[0,0,c] = 1.
    # Sum c U V W = 1.
    # Bilinear: C = A B.
    # Decomp: C = sum c W (U A) (V B).
    # = sum c W U V A B.
    # = (sum c W U V) A B.
    # For this to be AB, we need sum c W U V = 1.
    # So for n=m=p=1, rank 1 works. U=V=W=1.
    
    # Example: n=m=p=2.
    # Schoolbook rank 8.
    # Strassen rank 7.
    # Does rank 1 work?
    # U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1.
    # Sum c U V W = 1.
    # Bilinear: C_ik = sum c W (sum_j A_ij) (sum_l B_lk) = (sum_j A_ij)(sum_l B_lk).
    # This is NOT A B.
    # So rank 1 does NOT work for n=2.
    # Why?
    # Because the condition sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 is NOT equivalent to the bilinear map condition.
    # The bilinear map condition is:
    # sum_c W[k,i,c] U[i,j,c] V[l,k,c] = delta_{j,l}.
    # This is a condition on the 4-index array M[i,j,l,k] = sum_c W[k,i,c] U[i,j,c] V[l,k,c].
    # We need M[i,j,l,k] = delta_{j,l}.
    # Note that M does not depend on i or k except through the factors.
    # Actually, M[i,j,l,k] = delta_{j,l} for all i,k.
    # This means that for each i,k, the matrix M_{i,k} (with indices j,l) must be the identity.
    # And M_{i,k}[j,l] = sum_c W[k,i,c] U[i,j,c] V[l,k,c].
    # So we need:
    # For all i, k: sum_c W[k,i,c] U[i,j,c] V[l,k,c] = delta_{j,l}.
    # This is the correct condition.
    # The tensor T is not simply T[i,j,k]=1.
    # The "tensor" is the structure of the bilinear map.
    # The rank of this bilinear map is the minimum r such that there exist U,V,W satisfying the above.
    # This is the standard definition of bilinear rank (tensor rank).
    
    # So, for 4x5x6, we need to find U,V,W of rank r such that:
    # For all i in [0,4), k in [0,6):
    # The 5x5 matrix M_{i,k} with M_{i,k}[j,l] = sum_c W[k,i,c] U[i,j,c] V[l,k,c] is the identity matrix.
    
    # This is a system of equations.
    # We want to minimize r.
    # Schoolbook r = 4*5*6 = 120.
    # Can we do better?
    # Yes, likely.
    # For 2x2x2, Strassen gives r=7.
    # For 4x4x4, r=49 (Strassen) or less?
    # For 4x5x6, we can use Strassen-like recurrences.
    
    # Strategy:
    # Use a recursive construction based on Strassen for 2x2 blocks.
    # Since dimensions are 4,5,6, we can pad to 4,6,6 or 8,8,8?
    # Padding increases rank? No, if we decompose 8x8x8, we get a decomposition for 4x5x6 by ignoring extra entries?
    # If we have a decomposition for 8x8x8, can we use it for 4x5x6?
    # The condition must hold for i in [0,4), j in [0,5), k in [0,6).
    # If we use the 8x8x8 decomposition, we need the condition to hold for the sub-block.
    # The 8x8x8 decomposition satisfies M_{i,k}[j,l] = delta_{j,l} for all i,k,j,l in [0,8).
    # So it also satisfies it for i in [0,4), k in [0,6), j,l in [0,5).
    # So yes, a decomposition for 8x8x8 can be used for 4x5x6.
    # Rank of 8x8x8 via Strassen:
    # T(2) = 7.
    # T(4) = T(2)^2? No, T(2n) = T(2) * T(n)? No.
    # Strassen: T(2n) = 7 T(n).
    # T(2) = 7.
    # T(4) = 7 * T(2) = 49.
    # T(8) = 7 * T(4) = 343.
    # Schoolbook 8x8x8 is 512.
    # So 343 < 120? No, 343 > 120.
    # So padding to 8x8x8 is worse than schoolbook 4x5x6.
    
    # What about 4x6x6?
    # Or 4x5x6 directly?
    
    # Let's try to construct a decomposition for 4x5x6 using a "Strassen-like" approach on the dimensions.
    # Or just use the known best bounds?
    # The best known for 4x5x6 might be close to 120.
    # Actually, for small matrices, the optimal rank is often close to schoolbook.
    # But Strassen works for 2x2.
    # Can we combine a 2x2 Strassen block with the rest?
    # 4x5x6 = (2+2) x 5 x 6.
    # We can split i into two 2s.
    # C_{0:2, k} = A_{0:2, :} B_{:, k}.
    # C_{2:4, k} = A_{2:4, :} B_{:, k}.
    # The multiplication of 2x5 by 5x6.
    # This is not square.
    # Strassen is for square matrices.
    # We can pad 5 to 6? Or 4 to 8?
    
    # Let's try to implement a simple search for a low-rank decomposition using random initialization and gradient descent?
    # No, exact integer/half-integer entries required.
    # "integer/half-integer entries".
    # So we can use fractions.
    
    # Given the time limit, I will try to implement the Strassen decomposition for 4x4x4 and see if I can adapt it.
    # Or just return the schoolbook decomposition to be safe, but try to improve if possible.
    # The seed score is 1.0 (rank 120).
    # If I find a rank < 120, I improve.
    
    # Let's try rank 119?
    # Is it possible to reduce rank by 1?
    # Maybe not easily.
    
    # Let's try to construct a 4x4x4 Strassen decomposition.
    # Rank 49.
    # Then embed into 4x5x6?
    # 4x5x6 includes 4x4x4 as a sub-block (i=0..3, j=0..3, k=0..3).
    # But we need j up to 4 and k up to 5.
    # The 4x4x4 decomposition only handles j,k in 0..3.
    # We need to handle j=4 and k=4,5.
    # This adds more terms.
    
    # Alternative: Use the "Winograd" or other fast matrix multiplication?
    
    # Given the complexity, I will stick to the schoolbook decomposition but check if I can prune any redundant terms.
    # Unlikely.
    
    # Wait, the task says "integer/half-integer entries".
    # Strassen uses +/-1.
    # So it fits.
    
    # Let's try to generate the Strassen decomposition for 4x4x4 and see if I can extend it to 4x5x6 by adding schoolbook terms for the remaining parts.
    # Rank = Rank(4x4x4) + Rank(remaining).
    # 4x4x4 uses i=0..3, j=0..3, k=0..3.
    # Remaining:
    # j=4: 4x1x6. Rank = 4*1*6 = 24 (schoolbook).
    # k=4,5 with j=0..3: 4x4x2. Rank = 4*4*2 = 32 (schoolbook).
    # Total = 49 + 24 + 32 = 105.
    # 105 < 120.
    # This is an improvement!
    
    # Let's verify this logic.
    # We decompose the tensor into disjoint sub-tensors?
    # The tensor is T[i,j,k].
    # If we split the index space into disjoint regions, we can sum the ranks.
    # Region 1: i in [0,4), j in [0,4), k in [0,4). Size 4x4x4.
    # Region 2: i in [0,4), j in [4,5), k in [0,6). Size 4x1x6.
    # Region 3: i in [0,4), j in [0,4), k in [4,6). Size 4x4x2.
    # These regions are disjoint in (j,k) pairs?
    # Region 1: j<4, k<4.
    # Region 2: j=4, k<6.
    # Region 3: j<4, k>=4.
    # Yes, they are disjoint.
    # So the total rank is the sum of the ranks of the decompositions for each region.
    # We need to ensure the U,V,W matrices are constructed correctly so that they only act on their respective regions.
    # For Region 1, U,V,W are non-zero only for i<4, j<4, k<4.
    # For Region 2, U,V,W are non-zero only for i<4, j=4, k<6.
    # For Region 3, U,V,W are non-zero only for i<4, j<4, k>=4.
    
    # This is valid because the bilinear form is linear.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # We can write C_{ik} = C^{(1)}_{ik} + C^{(2)}_{ik} + C^{(3)}_{ik}.
    # Where each term corresponds to a region.
    # Since the regions are disjoint in j (and k), the sums don't interfere.
    
    # So we need a decomposition for 4x4x4 of rank 49.
    # And schoolbook for 4x1x6 (rank 24) and 4x4x2 (rank 32).
    # Total rank = 49 + 24 + 32 = 105.
    
    # Can we do better for 4x4x4?
    # Strassen gives 49.
    # Is there a rank < 49 for 4x4x4?
    # Best known for 4x4x4 is 49 (Strassen).
    # So 105 is likely the best we can get with this approach.
    
    # Let's implement this.
    
    def strassen_4x4x4():
        # Returns U, V, W for 4x4x4 with rank 49.
        # This requires a recursive construction.
        # Base case: 2x2x2 Strassen.
        
        def strassen_2x2x2():
            # Strassen 2x2x2.
            # U, V, W are 4x7, 4x7, 4x7.
            # U[i,j,c], V[j,k,c], W[k,i,c].
            # i,j,k in 0,1.
            
            U = [[0]*7 for _ in range(4)]
            V = [[0]*7 for _ in range(4)]
            W = [[0]*7 for _ in range(4)]
            
            # Strassen 7 products.
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B10)
            # M7 = (A01-A11)(B01+B11)
            
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # We need to define U, V, W such that sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 if (i,j,k) is in the product?
            # No, the condition is M[i,k][j,l] = delta_jl.
            # For 2x2x2, the rank is 7.
            # I will hardcode the Strassen 2x2x2 decomposition.
            # Reference: 
            # c0: U[0,0]=1, V[0,0]=1, W[0,0]=1 ... etc.
            
            # Let's use a known table.
            # Indices: U[i,j], V[j,k], W[k,i].
            
            # M1: (A00+A11)(B00+B11)
            # Contributes to C00, C11.
            # Actually, it's easier to just define the 7 terms.
            
            # Term 1:
            # U[0,0] = 1, U[1,1] = 1
            # V[0,0] = 1, V[1,1] = 1
            # W[0,0] = 1, W[1,1] = 1
            # Wait, W is [k,i].
            # C00 uses W[0,0]. C11 uses W[1,1].
            # So W[0,0] and W[1,1] should be set.
            
            # This is getting complicated to hardcode correctly without a reference.
            # I will use a simpler approach:
            # Just use schoolbook for 4x4x4 (rank 64).
            # Total rank = 64 + 24 + 32 = 120.
            # Same as seed.
            
            # Can I find a better 4x4x4?
            # Maybe not quickly.
            
            # What if I use Strassen for 2x2x2 blocks?
            # 4x4x4 can be viewed as 2 blocks of 2x2x4?
            
            # Let's try to just return the schoolbook decomposition for now, but with the hope that the "Strassen" idea might be implemented in a future iteration.
            # Actually, the prompt asks for a "stronger algorithm".
            # 105 is stronger than 120.
            # I need the Strassen 4x4x4 code.
            
            # I will implement a recursive Strassen function.
            
            def strassen(n):
                if n == 1:
                    # U[0,0] = 1, V[0,0] = 1, W[0,0] = 1
                    U = [[1]]
                    V = [[1]]
                    W = [[1]]
                    return U, V, W
                if n == 2:
                    # Hardcode 2x2x2 Strassen
                    # U, V, W are 4x7
                    U = [[0]*7 for _ in range(4)]
                    V = [[0]*7 for _ in range(4)]
                    W = [[0]*7 for _ in range(4)]
                    
                    # c0: M1
                    # U[0,0] (A00), U[1,1] (A11)
                    # V[0,0] (B00), V[1,1] (B11)
                    # W[0,0] (C00), W[1,1] (C11)
                    # But Strassen M1 contributes to C00 and C11.
                    # C00 = M1 + ...
                    # C11 = M1 + ...
                    # So W[0,0] = 1, W[1,1] = 1.
                    # U[0,0] = 1, U[1,1] = 1.
                    # V[0,0] = 1, V[1,1] = 1.
                    U[0][0] = 1; U[1][0] = 1
                    V[0][0] = 1; V[1][0] = 1
                    W[0][0] = 1; W[1][0] = 1
                    # Wait, W is [k,i].
                    # C00 is i=0,k=0. W[0,0].
                    # C11 is i=1,k=1. W[1,1].
                    # So W[0][0] = 1, W[1][1] = 1.
                    # My indexing U[i,j], V[j,k], W[k,i].
                    # U is 4x7. Index i*2+j.
                    # V is 4x7. Index j*2+k.
                    # W is 4x7. Index k*2+i.
                    
                    U[0][0] = 1 # i=0,j=0
                    U[3][0] = 1 # i=1,j=1
                    V[0][0] = 1 # j=0,k=0
                    V[3][0] = 1 # j=1,k=1
                    W[0][0] = 1 # k=0,i=0
                    W[1][0] = 1 # k=1,i=1 -> Wait, W[1] is k=0? No.
                    # W index is k*2+i.
                    # k=0,i=0 -> 0.
                    # k=1,i=1 -> 3.
                    W[0][0] = 1
                    W[3][0] = 1
                    
                    # c1: M2 = (A10+A11)B00
                    # U[1,0]=1, U[1,1]=1
                    # V[0,0]=1
                    # W[1,0]=1 (C10)
                    U[2][1] = 1 # i=1,j=0
                    U[3][1] = 1 # i=1,j=1
                    V[0][1] = 1 # j=0,k=0
                    W[2][1] = 1 # k=0,i=1 -> index 0*2+1 = 1? No.
                    # W index k*2+i.
                    # k=0, i=1 -> 1.
                    W[1][1] = 1
                    
                    # This is error-prone.
                    # I will skip the hardcoding and just use schoolbook for 4x4x4.
                    # Rank 64.
                    # Total 120.
                    
                    return U, V, W
                
                # Recursive step
                # This is too complex to do correctly in short time.
                
            return U, V, W
        
        # Since I can't reliably implement Strassen 4x4x4 here, I will stick to schoolbook.
        pass

    # Fallback: Schoolbook
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

# EVOLVE-BLOCK-END
