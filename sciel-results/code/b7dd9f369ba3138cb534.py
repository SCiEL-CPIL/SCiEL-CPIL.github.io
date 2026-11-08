"""Program for tensor rank of 2x5 by 5x6 matrix multiplication."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find the rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    """
    n, m, p = 2, 5, 6
    
    # Known best ranks for small matrix multiplication:
    # mu(1,1,1)=1, mu(1,2,2)=4, mu(2,2,2)=7, mu(1,2,3)=6, mu(2,2,3)=10, mu(1,3,3)=9
    # mu(1,2,4)=8, mu(2,2,4)=12, mu(1,3,4)=12, mu(2,3,3)=14, mu(1,2,5)=10, mu(2,2,5)=14? 
    # Actually mu(2,2,n) is known to be 2n-2 for n>=2? No, mu(2,2,2)=7, mu(2,2,3)=10, mu(2,2,4)=12?
    # Let's use known exact ranks or good upper bounds.
    
    # We can decompose M_{2,5,6} by splitting dimensions.
    # Option 1: Split 5 into 2+2+1. 
    #   M_{2,5,6} = M_{2,2,6} + M_{2,2,6} + M_{2,1,6}
    #   mu(2,1,6) = 6
    #   mu(2,2,6): split 6 into 3+3 -> mu(2,2,3)+mu(2,2,3) = 10+10=20? Or mu(2,2,4)+mu(2,2,2)=12+7=19?
    #   So mu(2,5,6) <= 19+19+6 = 44.
    
    # Option 2: Split 6 into 3+3.
    #   M_{2,5,6} = M_{2,5,3} + M_{2,5,3}
    #   mu(2,5,3): split 5 into 2+2+1 -> mu(2,2,3)+mu(2,2,3)+mu(2,1,3) = 10+10+3 = 23.
    #   Total = 46. Worse.
    
    # Option 3: Split 5 into 3+2.
    #   M_{2,5,6} = M_{2,3,6} + M_{2,2,6}
    #   mu(2,3,6): split 6 into 3+3 -> mu(2,3,3)+mu(2,3,3) = 14+14=28.
    #   mu(2,2,6) = 19 (from above).
    #   Total = 47. Worse.
    
    # Option 4: Split 5 into 2+3.
    #   Same as above.
    
    # Option 5: Split 2 into 1+1.
    #   M_{2,5,6} = M_{1,5,6} + M_{1,5,6}
    #   mu(1,5,6) = 5+6-1 = 10.
    #   Total = 20. This is much better!
    
    # Let's verify mu(1,m,p) = m+p-1.
    # This is a standard result. The tensor for 1xm * mxp is just the sum of outer products of basis vectors.
    # Rank is m+p-1.
    # So mu(2,5,6) <= mu(1,5,6) + mu(1,5,6) = 10 + 10 = 20.
    
    # Can we do better?
    # Known lower bound for mu(2,5,6) is at least 2*5+2*6 - 2*2 = 10+12-4 = 18? 
    # Actually, the lower bound is n+p+m - ... no, it's more complex.
    # But 20 is a very strong upper bound.
    
    # Let's construct the decomposition for rank 20.
    # We split the first dimension (n=2) into two 1s.
    # For each i in {0, 1}, we have a M_{1,5,6} tensor.
    # The M_{1,5,6} tensor has rank 5+6-1 = 10.
    # We need to construct U, V, W for each part and then concatenate.
    
    # M_{1, m, p} construction:
    # The tensor T[i,j,k] = delta_{i,0} (since i is always 0 for 1xm).
    # We want U, V, W such that sum_c U[i,j,c] V[j,k,c] W[k,i,c] = T[i,j,k].
    # For 1xm * mxp, the output is a 1xp matrix.
    # Let's use the standard construction for mu(1,m,p) = m+p-1.
    # We can index the rank by (a,b) where a=1..m, b=1..p, with a+b <= m+p? No.
    # Standard construction:
    # Let the rank be m+p-1.
    # We can use the fact that the product of a 1xm vector and mxp matrix is a 1xp vector.
    # Let u be the 1xm vector, v be the mxp matrix.
    # (u v)[k] = sum_j u_j v_{j,k}.
    # This is a bilinear form.
    # The rank is m+p-1.
    # One construction:
    # For c = 0 .. m+p-2:
    #   Let a = min(c, m-1), b = c - a.
    #   This doesn't seem right.
    
    # Let's use a known explicit construction for mu(1,m,p).
    # Let the rank be R = m+p-1.
    # We can write the product as:
    # C[k] = sum_j A[j] B[j,k]
    # We can introduce an intermediate variable z.
    # Actually, the simplest way is to note that the tensor is the sum of m+p-1 rank-1 tensors.
    # Let's just generate it numerically or use a known pattern.
    
    # Pattern for mu(1,m,p):
    # Let R = m+p-1.
    # For c in 0..R-1:
    #   Let a = min(c, m-1)
    #   Let b = c - a
    #   If b >= p: continue? No.
    
    # Let's use the fact that we can compute the 1xm * mxp product by:
    # 1. Compute partial sums of A.
    # 2. Use prefix sums to compute the product in m+p-1 multiplications.
    
    # Explicitly:
    # Let A = [a_1, ..., a_m]
    # Let B = [b_{j,k}]
    # C_k = sum_{j=1}^m a_j b_{j,k}
    
    # We can define:
    # S_0 = 0
    # S_i = a_1 + ... + a_i
    # Then a_j = S_j - S_{j-1}
    # C_k = sum_j (S_j - S_{j-1}) b_{j,k}
    #     = sum_j S_j b_{j,k} - sum_j S_{j-1} b_{j,k}
    #     = sum_j S_j b_{j,k} - sum_i S_i b_{i+1,k}
    #     = S_0 b_{0,k} + sum_{j=1}^{m-1} S_j (b_{j,k} - b_{j+1,k}) + S_m b_{m,k}
    #     = sum_{j=1}^{m-1} S_j (b_{j,k} - b_{j+1,k}) + S_m b_{m,k} (since S_0=0)
    
    # This requires m subtractions (to compute differences) and m multiplications (S_j * diff) and m additions.
    # But we count multiplications only.
    # The terms are S_j * (b_{j,k} - b_{j+1,k}) and S_m * b_{m,k}.
    # For each k, we have m terms.
    # Total terms = m * p. This is not m+p-1.
    
    # Wait, the rank of the 1xm * mxp tensor is indeed m+p-1.
    # Let's look up the explicit construction.
    # It is related to the Pascal triangle.
    
    # Let's just use a numerical approach to find the rank 10 decomposition for 1x5*5x6.
    # Or, we can use the fact that we can split 5x6 into 5x1 + ... + 5x1? No.
    
    # Let's try to find the rank of 1x5*5x6 by solving the linear system.
    # We have 30 equations (5*6) and 10 unknowns (rank 10).
    # Each unknown corresponds to a triplet (u_j, v_{j,k}, w_k).
    # This is a system of bilinear equations.
    
    # Let's use a simpler approach:
    # We know mu(1,m,p) = m+p-1.
    # We can construct it by:
    # Let R = m+p-1.
    # For c in 0..R-1:
    #   Let a = min(c, m-1)
    #   Let b = c - a
    #   If b < p:
    #     U[a, c] = 1
    #     V[a, b, c] = 1
    #     W[b, c] = 1
    #   Else:
    #     ...
    
    # Let's verify for m=2, p=2. R=3.
    # c=0: a=0, b=0. U[0,0]=1, V[0,0,0]=1, W[0,0]=1.
    # c=1: a=1, b=0. U[1,1]=1, V[1,0,1]=1, W[0,1]=1.
    # c=2: a=1, b=1. U[1,2]=1, V[1,1,2]=1, W[1,2]=1.
    
    # Product C[k] = sum_j A[j] B[j,k]
    # C[0] = A[0]B[0,0] + A[1]B[1,0]
    # Using the tensors:
    # T[0,0,0] = U[0,0]V[0,0,0]W[0,0] = 1
    # T[1,0,1] = U[1,1]V[1,0,1]W[0,1] = 1
    # T[1,1,2] = U[1,2]V[1,1,2]W[1,2] = 1
    # T[0,1,?] ...
    # This doesn't seem to match.
    
    # Let's use a different known construction.
    # The rank-1 tensors correspond to the paths in a grid from (0,0) to (m,p).
    # There are m+p-1 such paths? No, there are binomial(m+p, m) paths.
    
    # Actually, the tensor is the "outer product" of a vector and a matrix.
    # It is equivalent to the matrix product of a 1xm and mxp matrix.
    # This is just the dot product of a vector with each column of the matrix.
    # The rank is the rank of the flattening of the tensor.
    
    # Let's just use the fact that we can compute it with m+p-1 multiplications.
    # For 1x5 * 5x6, R=10.
    # We will construct the 10 rank-1 tensors.
    
    # Let's use the following construction for mu(1,m,p):
    # Let R = m+p-1.
    # For c in 0..R-1:
    #   Let a = min(c, m-1)
    #   Let b = c - a
    #   If b < p:
    #     U[a, c] = 1
    #     V[a, b, c] = 1
    #     W[b, c] = 1
    #   Else:
    #     # This case shouldn't happen if R=m+p-1
    #     pass
    
    # Let's re-evaluate the construction for m=2, p=2.
    # R=3.
    # c=0: a=0, b=0. U[0,0]=1, V[0,0,0]=1, W[0,0]=1.
    # c=1: a=1, b=0. U[1,1]=1, V[1,0,1]=1, W[0,1]=1.
    # c=2: a=1, b=1. U[1,2]=1, V[1,1,2]=1, W[1,2]=1.
    
    # Let's compute the product for A=[a0, a1], B=[[b00, b01], [b10, b11]].
    # C[0] = a0*b00 + a1*b10
    # C[1] = a0*b01 + a1*b11
    
    # Using the tensors:
    # T[i,j,k] = sum_c U[i,c] V[j,k,c] W[k,c] (wait, the indices are U[i,c], V[j,k,c], W[k,c]?)
    # The tensor is T[i,j,k] where i is the row of A, j is the column of A/row of B, k is the column of B.
    # For 1xm, i is always 0.
    # So T[0,j,k] = sum_c U[j,c] V[j,k,c] W[k,c]? No.
    # The standard decomposition is T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    # For 1xm, i=0. So T[0,j,k] = sum_c U[0,j,c] V[j,k,c] W[k,0,c].
    # In the construction above, I used U[a,c], V[a,b,c], W[b,c]. This implies i=a, j=a, k=b?
    # No, that's not right.
    
    # Let's use a different indexing.
    # Let the rank be R.
    # For each c, we have a triplet (u_c, v_c, w_c).
    # u_c is a 1xm vector, v_c is a mxp matrix, w_c is a 1xp vector? No.
    # The tensor is T[i,j,k].
    # Rank 1: T[i,j,k] = A[i,j] B[j,k] C[k,i].
    # For 1xm, i=0.
    # T[0,j,k] = A[j] B[j,k] C[k].
    # We want sum_c A_c[j] B_c[j,k] C_c[k] = delta_{i,0} (which is 1 for all j,k since i=0).
    # Wait, the target tensor is T[0,j,k] = 1 for all j,k? No.
    # The target tensor for matrix multiplication is T[i,j,k] = delta_{i,0} (since the first matrix is 1xm).
    # Actually, the tensor represents the bilinear map (A, B) -> C.
    # C[k] = sum_j A[j] B[j,k].
    # So the tensor is T[j,k] = 1 for all j,k? No, the tensor is a 3-tensor T[i,j,k].
    # T[i,j,k] is the coefficient of A[i] B[j,k] in C[l].
    # C[l] = sum_i,j A[i] B[j,k] T[i,j,k]? No.
    # C[l] = sum_i,j A[i] B[i,j] (for matrix product).
    # Here, A is 1xm, B is mxp.
    # C[k] = sum_j A[j] B[j,k].
    # So the tensor is T[j,k] with no i index?
    # No, the task says "multiplying a 2 by 5 matrix by a 5 by 6 matrix".
    # So n=2, m=5, p=6.
    # The tensor is T[i,j,k] where i=0..1, j=0..4, k=0..5.
    # T[i,j,k] = 1 if i=0 and j=k? No.
    # The product C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor is T[i,j,k] = 1 for all i,j,k? No.
    # The tensor is defined such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k].
    # So T[i,j,k] = 1 for all i,j,k.
    
    # Ah, so the tensor is simply the all-ones tensor? No.
    # If T[i,j,k] = 1, then C[i,k] = sum_j A[i,j] B[j,k], which is correct.
    # So the tensor is the all-ones tensor of size 2x5x6.
    # The rank of the all-ones tensor is 1? No, because it's not a rank-1 tensor in the sense of outer product.
    # Wait, T[i,j,k] = 1 = 1*1*1. So it is a rank-1 tensor?
    # If T[i,j,k] = U[i] V[j] W[k], then T[i,j,k] = 1*1*1 = 1.
    # So the rank is 1?
    # But that's not right. Matrix multiplication is not a rank-1 tensor operation.
    # The tensor is not T[i,j,k] = 1.
    # The tensor is T[i,j,k] = delta_{j, something}?
    
    # Let's re-read the tensor definition.
    # The tensor for matrix multiplication M_{n,m,p} is defined by:
    # T[i,j,k] = 1 if i,j,k are the indices, and the multiplication is C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor is T[i,j,k] = 1 for all i,j,k?
    # If so, then it is a rank-1 tensor.
    # But we know that the rank of M_{2,2,2} is 7.
    # So my understanding of the tensor is wrong.
    
    # The correct definition:
    # The tensor T is a 3-tensor such that the bilinear form is:
    # <T, A \otimes B> = C
    # where C is the matrix product.
    # So T[i,j,k] is the coefficient of A[i,j] B[j,k] in C[i,k].
    # C[i,k] = sum_j A[i,j] B[j,k].
    # So the coefficient of A[i,j] B[j,k] in C[i,k] is 1 if the indices match, and 0 otherwise.
    # But the tensor T is indexed by (i,j,k).
    # The output is indexed by (i,k).
    # So the tensor is T[i,j,k] where i is the row of C, j is the inner index, k is the column of C.
    # And T[i,j,k] = 1 for all i,j,k.
    # But then the bilinear form is sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k].
    # This is not C[i,k]. This is the sum of all elements of C.
    
    # Ah, the tensor is not a single scalar output.
    # The tensor is a family of tensors, one for each output element C[i,k].
    # Or, the tensor is T[i,j,k] where the output is a vector/matrix.
    # In the context of tensor rank, the tensor is T in U \otimes V \otimes W.
    # The multiplication map is a bilinear map U x V -> W.
    # This corresponds to a tensor T in U \otimes V \otimes W^*.
    # T(i, v, w^*) = w^*(B(i,v)).
    # So T[i,j,k] = w_k^*(C[i,k]).
    # If we use the standard basis, then T[i,j,k] = 1 if the output index is (i,k) and the input indices are (i,j) and (j,k).
    # So T[i,j,k] = 1 for all i,j,k.
    # But this gives the sum, not the individual elements.
    
    # I think the tensor is T[i,j,k] where i, j, k are the indices, and the tensor represents the entire multiplication.
    # The rank of this tensor is the tensor rank of the multiplication.
    # For M_{2,2,2}, the rank is 7.
    # So the tensor is not the all-ones tensor.
    
    # Let's look at the seed program.
    # It uses rank n*m*p.
    # And U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # This suggests that the tensor is T[i,j,k] = 1 for all i,j,k, but the way it's decomposed is different.
    # Wait, if T[i,j,k] = 1, then the rank is 1.
    # So the tensor must be something else.
    
    # Let's re-examine the seed program.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # This is the standard format for the matrix multiplication tensor.
    # The tensor T is in U^* \otimes V^* \otimes W^*.
    # T(U, V, W) = sum_{i,j,k} T[i,j,k] U[i,j] V[j,k] W[k,i].
    # For matrix multiplication, we want T[i,j,k] = 1 if the indices are consistent with the multiplication.
    # The multiplication is C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor should be such that T(A, B, C) = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i] = 0 if C is not the product.
    # Actually, the tensor is defined by T[i,j,k] = 1 for all i,j,k.
    # But then the rank is 1.
    
    # I am confused. Let's trust the seed program's structure.
    # The seed program uses rank n*m*p and sets U, V, W to 1 for specific indices.
    # This is the schoolbook algorithm.
    # So the rank is n*m*p.
    # My goal is to find a lower rank.
    
    # Let's go back to the block decomposition.
    # We can split n=2 into 1+1.
    # M_{2,5,6} = M_{1,5,6} + M_{1,5,6}.
    # The rank of M_{1,5,6} is 5+6-1 = 10.
    # So the total rank is 20.
    
    # Let's construct the M_{1,5,6} decomposition.
    # For 1xm * mxp, the rank is m+p-1.
    # We can use the following construction:
    # Let R = m+p-1.
    # For c in 0..R-1:
    #   Let a = min(c, m-1)
    #   Let b = c - a
    #   If b < p:
    #     U[a, c] = 1  # U is indexed by (i*m+j) = j, since i=0.
    #     V[a*p+b, c] = 1  # V is indexed by (j*p+k) = b.
    #     W[b*n+a, c] = 1  # W is indexed by (k*n+i) = b.
    
    # Let's verify for m=2, p=2, n=1.
    # R=3.
    # c=0: a=0, b=0. U[0,0]=1, V[0,0]=1, W[0,0]=1.
    # c=1: a=1, b=0. U[1,1]=1, V[1,0]=1, W[0,1]=1.
    # c=2: a=1, b=1. U[1,2]=1, V[1,1]=1, W[1,2]=1.
    
    # Let's check the product.
    # C[k] = sum_j A[j] B[j,k].
    # Using the tensors:
    # T[j,k] = sum_c U[j,c] V[j,k,c] W[k,c].
    # c=0: U[0,0]=1, V[0,0,0]=1, W[0,0]=1. So T[0,0] += 1.
    # c=1: U[1,1]=1, V[1,0,1]=1, W[0,1]=1. So T[1,0] += 1.
    # c=2: U[1,2]=1, V[1,1,2]=1, W[1,2]=1. So T[1,1] += 1.
    
    # So T[0,0]=1, T[1,0]=1, T[1,1]=1. T[0,1]=0.
    # But we want T[j,k]=1 for all j,k.
    # So this construction is wrong.
    
    # Let's try a different construction.
    # The rank of M_{1,m,p} is m+p-1.
    # We can use the fact that the product can be computed with m+p-1 multiplications.
    # Let's use the following:
    # Let R = m+p-1.
    # For c in 0..R-1:
    #   Let a = min(c, m-1)
    #   Let b = c - a
    #   If b < p:
    #     U[a, c] = 1
    #     V[a*p+b, c] = 1
    #     W[b, c] = 1  # Note: W is indexed by k, since n=1.
    
    # This is the same as before.
    
    # Let's try to find the correct construction online or in a book.
    # The tensor for 1xm * mxp is the sum of m+p-1 rank-1 tensors.
    # Each rank-1 tensor corresponds to a path in a grid.
    
    # Let's just use the numerical approach.
    # We can set up the linear system and solve it.
    # But that might be slow.
    
    # Let's try to use the fact that we can split the 5x6 matrix into 5x1 + 5x1 + ... + 5x1.
    # No, that doesn't help.
    
    # Let's use the known result that mu(1,m,p) = m+p-1.
    # And construct the decomposition using a simple pattern.
    
    # Pattern:
    # Let R = m+p-1.
    # For c in 0..R-1:
    #   Let a = c
    #   Let b = 0
    #   If a < m:
    #     U[a, c] = 1
    #     V[a*p, c] = 1
    #     W[0, c] = 1
    #   Else:
    #     a = m-1
    #     b = c - m + 1
    #     U[m-1, c] = 1
    #     V[(m-1)*p+b, c] = 1
    #     W[b, c] = 1
    
    # Let's verify for m=2, p=2. R=3.
    # c=0: a=0, b=0. U[0,0]=1, V[0,0]=1, W[0,0]=1.
    # c=1: a=1, b=0. U[1,1]=1, V[2,1]=1, W[0,1]=1.
    # c=2: a=2->1, b=1. U[1,2]=1, V[3,2]=1, W[1,2]=1.
    
    # Product:
    # T[0,0] = U[0,0]V[0,0]W[0,0] = 1.
    # T[0,1] = 0.
    # T[1,0] = U[1,1]V[2,1]W[0,1] = 1.
    # T[1,1] = U[1,2]V[3,2]W[1,2] = 1.
    
    # So T[0,0]=1, T[1,0]=1, T[1,1]=1, T[0,1]=0.
    # Still not all ones.
    
    # I think the tensor is not all ones.
    # The tensor is T[i,j,k] = 1 if i,j,k are the indices, but the output is C[i,k].
    # So the tensor is a 3-tensor with indices (i,j,k).
    # And the value is 1.
    # But then the rank is 1.
    
    # I must be missing something fundamental about the tensor definition.
    # Let's assume the seed program is correct and the rank is n*m*p.
    # And my block decomposition is correct and the rank is 20.
    # I will implement the block decomposition with rank 20.
    
    # For the 1x5*5x6 part, I will use a numerical solver to find the rank 10 decomposition.
    # Or I can use a known explicit formula.
    
    # Let's try the following:
    # For 1xm * mxp, the rank-1 tensors are:
    # For a=0..m-1, b=0..p-1, with a+b < m+p-1?
    
    # Let's just use the fact that we can compute it with m+p-1 multiplications.
    # I will use a simple heuristic to construct the decomposition.
    
    # Let's use the following construction for mu(1,m,p):
    # Let R = m+p-1.
    # For c in 0..R-1:
    #   Let a = min(c, m-1)
    #   Let b = c - a
    #   If b < p:
    #     U[a, c] = 1
    #     V[a*p+b, c] = 1
    #     W[b, c] = 1
    
    # Even if it doesn't work for m=2,p=2, it might work for larger m,p.
    # Or I can adjust it.
    
    # Let's try to fix the construction.
    # We want T[j,k] = 1 for all j,k.
    # So we need sum_c U[j,c] V[j,k,c] W[k,c] = 1.
    
    # Let's try:
    # For c in 0..m+p-2:
    #   Let a = min(c, m-1)
    #   Let b = c - a
    #   If b < p:
    #     U[a, c] = 1
    #     V[a*p+b, c] = 1
    #     W[b, c] = 1
    #   Else:
    #     U[m-1, c] = 1
    #     V[(m-1)*p+b-p, c] = 1
    #     W[b-p, c] = 1
    
    # Let's verify for m=2, p=2. R=3.
    # c=0: a=0, b=0. U[0,0]=1, V[0,0]=1, W[0,0]=1.
    # c=1: a=1, b=0. U[1,1]=1, V[1,0]=1, W[0,1]=1.
    # c=2: a=1, b=1. U[1,2]=1, V[1,1]=1, W[1,2]=1.
    
    # T[0,0] = 1.
    # T[0,1] = 0.
    # T[1,0] = 1.
    # T[1,1] = 1.
    
    # We need T[0,1]=1.
    # So we need another term.
    # But R=3, so we have only 3 terms.
    
    # This means the construction is wrong.
    
    # Let's try a different construction.
    # Let R = m+p-1.
    # For c in 0..R-1:
    #   Let a = c
    #   Let b = 0
    #   If a < m:
    #     U[a, c] = 1
    #     V[a*p, c] = 1
    #     W[0, c] = 1
    #   Else:
    #     a = m-1
    #     b = c - m + 1
    #     U[m-1, c] = 1
    #     V[(m-1)*p+b, c] = 1
    #     W[b, c] = 1
    
    # This is the same as before.
    
    # I think the correct construction is:
    # For c in 0..m+p-2:
    #   Let a = min(c, m-1)
    #   Let b = c - a
    #   U[a, c] = 1
    #   V[a*p+b, c] = 1
    #   W[b, c] = 1
    
    # And it works. My verification was wrong.
    # Let's re-verify.
    # m=2, p=2. R=3.
    # c=0: a=0, b=0. U[0,0]=1, V[0,0]=1, W[0,0]=1.
    # c=1: a=1, b=0. U[1,1]=1, V[1,0]=1, W[0,1]=1.
    # c=2: a=1, b=1. U[1,2]=1, V[1,1]=1, W[1,2]=1.
    
    # T[j,k] = sum_c U[j,c] V[j,k,c] W[k,c].
    # T[0,0] = U[0,0]V[0,0,0]W[0,0] = 1*1*1 = 1.
    # T[0,1] = U[0,0]V[0,1,0]W[1,0] + U[0,1]V[0,1,1]W[1,1] + ...
    # V[0,1,0] is V[0*p+1, 0] = V[1,0]. But V[1,0] is not set for c=0.
    # V is indexed by (j*p+k). So V[0,0] is j=0,k=0. V[1,0] is j=0,k=1.
    # In the construction, V[a*p+b, c] = 1.
    # For c=0, a=0, b=0. So V[0,0]=1.
    # For c=1, a=1, b=0. So V[1,1]=1? No, V[a*p+b, c] = V[1*2+0, 1] = V[2,1]=1.
    # So V[2,1]=1. This is j=1, k=0.
    # For c=2, a=1, b=1. So V[1*2+1, 2] = V[3,2]=1. This is j=1, k=1.
    
    # So V is:
    # V[0,0]=1 (j=0,k=0)
    # V[2,1]=1 (j=1,k=0)
    # V[3,2]=1 (j=1,k=1)
    
    # U is:
    # U[0,0]=1 (j=0)
    # U[1,1]=1 (j=1)
    # U[1,2]=1 (j=1)
    
    # W is:
    # W[0,0]=1 (k=0)
    # W[0,1]=1 (k=0)
    # W[1,2]=1 (k=1)
    
    # Now T[j,k] = sum_c U[j,c] V[j,k,c] W[k,c].
    # T[0,0] = U[0,0]V[0,0,0]W[0,0] = 1*1*1 = 1.
    # T[0,1] = U[0,0]V[0,1,0]W[1,0] + ... = 0.
    # T[1,0] = U[1,1]V[1,0,1]W[0,1] = 1*1*1 = 1. (V[1,0,1] is V[1*2+0,1]=V[2,1]=1)
    # T[1,1] = U[1,2]V[1,1,2]W[1,2] = 1*1*1 = 1. (V[1,1,2] is V[1*2+1,2]=V[3,2]=1)
    
    # So T[0,0]=1, T[1,0]=1, T[1,1]=1, T[0,1]=0.
    # Still not all ones.
    
    # I give up on finding the explicit construction.
    # I will use the numerical solver.
    
    # But for now, I will just return the schoolbook decomposition with rank 60.
    # Or I can try the block decomposition with rank 44.
    
    # Let's use the block decomposition with rank 44.
    # Split 5 into 2+2+1.
    # M_{2,5,6} = M_{2,2,6} + M_{2,2,6} + M_{2,1,6}.
    # mu(2,1,6) = 6.
    # mu(2,2,6) = 19 (from mu(2,2,4)+mu(2,2,2)=12+7=19).
    # Total = 19+19+6 = 44.
    
    # I will implement this.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper function to construct M_{n,m,p} decomposition.
    def construct_block(n, m, p, offset):
        # For now, use schoolbook for the blocks.
        # This will give rank n*m*p.
        # For M_{2,2,6}, rank = 24.
        # For M_{2,1,6}, rank = 12.
        # Total = 24+24+12 = 60. Same as schoolbook.
        # So this is not an improvement.
        
        # I need to use the lower rank for the blocks.
        # But I don't have the explicit construction.
        
        # Let's just use the schoolbook for now.
        # And return rank 60.
        pass
    
    # Let's just return the schoolbook decomposition.
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
