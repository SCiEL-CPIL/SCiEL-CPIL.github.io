"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: return best known result if we run out of time.
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 10.0
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Strategy: Use Strassen-like decomposition for 2x2x2 blocks combined with standard multiplication for the rest.
    # The matrix multiplication tensor is T_{i,j,k} = A_{i,j} B_{j,k} C_{k,i} (or similar indexing)
    # We want to decompose the multiplication of (n x m) by (m x p) into rank-r terms.
    
    # Let's try to exploit the structure.
    # n=2, m=3, p=10.
    # Total size 2*3*10 = 60.
    
    # We can view the 2x3 matrix A and 3x10 matrix B.
    # The product C is 2x10.
    # C_{i,k} = sum_{j=1}^3 A_{i,j} B_{j,k}.
    
    # We can try to use Strassen's algorithm on the 2x2 blocks of the result?
    # Or perhaps decompose the m=3 dimension?
    
    # Let's look at the known result: rank 55 was achieved by "Block decomposition using Strassen's algorithm for 2x2x2 sub-problems".
    # This suggests splitting the problem into blocks where Strassen applies.
    
    # Since n=2, we can't split n into 2x2 easily (it is already 2).
    # But we can split the output or the intermediate dimensions.
    
    # Actually, the standard Strassen algorithm reduces 2x2 matrix multiplication from 8 to 7 multiplications.
    # Here we have 2x3 * 3x10.
    
    # Let's try to construct a decomposition of rank 55 or lower.
    # If we treat the 2x3 matrix as two 2x2 blocks (overlapping?) or something similar.
    
    # Alternative: Use the fact that rank(2x3x10) <= rank(2x2x2) * something?
    # No, tensor rank is not multiplicative like that simply.
    
    # Let's try a direct construction based on the "55" hint.
    # 55 = 60 - 5. We saved 5 multiplications.
    
    # How to save 5?
    # If we split the 10 columns into two 5-column blocks?
    # Or split the 3 rows into 2+1?
    
    # Let's try splitting the m=3 dimension into 2+1.
    # A = [A_1 (2x2) | A_2 (2x1)]
    # B = [B_1 (2x10) ; B_2 (1x10)]
    # C = A_1 B_1 + A_2 B_2
    
    # Multiplying 2x2 by 2x10:
    # This can be seen as 10 independent 2x2 by 2x1 multiplications?
    # Or we can use Strassen for the 2x2 part of A and B?
    # Strassen applies to 2x2 * 2x2.
    # Here we have 2x2 * 2x10.
    # We can split the 2x10 into ten 2x1 matrices.
    # Each 2x2 * 2x1 multiplication requires 4 multiplications (standard) or can it be done in fewer?
    # The rank of 2x2x1 tensor?
    # T_{i,j,k} = A_{i,j} B_{j,k}.
    # This is essentially two vector multiplications.
    # Rank of 2x2x1 is 4?
    # Let's check:
    # c1 = a11*b1 + a12*b2
    # c2 = a21*b1 + a22*b2
    # This is bilinear.
    # Can we do it in 3?
    # (a11+a21)(b1) + (a12+a22)(b2) ... no, we need to separate them.
    # Actually, the rank of 2x2x1 is 4.
    # So 2x2 * 2x10 using standard is 10 * 4 = 40 multiplications.
    # Plus 2x1 * 1x10 is 1 * 10 = 10 multiplications.
    # Total 50?
    # Wait, if rank(2x2x1)=4, then 2x2 * 2x10 is 40.
    # 2x1 * 1x10 is 10.
    # Total 50.
    # This is better than 60.
    # Can we do better?
    
    # Let's verify the rank of 2x2x1.
    # The tensor is T = e1⊗e1⊗e1 + e1⊗e2⊗e2 + e2⊗e1⊗e1 + e2⊗e2⊗e2? No.
    # Indices: i in {1,2} (row of A), j in {1,2} (col of A / row of B), k in {1} (col of B).
    # T_{i,j,1} = A_{i,j} B_{j,1}.
    # The tensor is sum_{i,j} E_{i,j} ⊗ F_{j,1} ⊗ G_{1,i}?
    # Actually, the multiplication tensor M_{n,m,p} has rank.
    # For 2x2x1, the rank is 4.
    
    # So 50 is a valid upper bound.
    # Can we use Strassen on the 2x2 part?
    # Strassen reduces 2x2x2 from 8 to 7.
    # Does it reduce 2x2x1?
    # If we pad the 2x1 to 2x2, we get 2x2x2.
    # Rank 7 for 2x2x2.
    # But we only need half of the output?
    # This doesn't directly give a rank 3 for 2x2x1.
    
    # Let's check if 2x2x1 rank is 3.
    # If rank is 3, then 2x2 * 2x10 would be 30.
    # Plus 10 = 40.
    # Is rank(2x2x1) = 3?
    # The 2x2x1 tensor is the same as the "outer product" of two vectors? No.
    # It's the multiplication of a 2x2 matrix by a 2x1 vector.
    # This is linear in B for fixed A, and linear in A for fixed B.
    # It's a bilinear map.
    # The rank of the 2x2x1 tensor is 4.
    # Proof: The 2x2x1 tensor has 4 non-zero entries in its 2x2x1 array.
    # It cannot be decomposed into 3 rank-1 tensors because the 2x2 matrix of coefficients for any fixed slice is rank 2, and we need to cover 2 slices?
    # Actually, it is known that rank(2x2x1) = 4.
    
    # So 50 is a good target.
    # Let's try to get even lower.
    
    # What if we split n=2 into 1+1?
    # A = [A1 (1x3); A2 (1x3)]
    # B = 3x10
    # C1 = A1 B (1x10) -> 3 multiplications
    # C2 = A2 B (1x10) -> 3 multiplications
    # Total 6? No, 3+3=6?
    # Wait, 1x3 * 3x10 is 3 multiplications.
    # So 2x3 * 3x10 split into two 1x3 * 3x10 is 3+3=6?
    # That's impossible. The rank must be at least 20 (since 2x10 output, each entry needs at least 1 mult? No).
    # The standard rank is 60.
    # If I split n=2 into 1+1, I am doing:
    # C1 = A1 B (1x3 * 3x10) -> rank 3
    # C2 = A2 B (1x3 * 3x10) -> rank 3
    # Total rank 6?
    # This is wrong. The rank of 1x3x10 is 3.
    # But the two problems share the B matrix?
    # No, in tensor decomposition, we are decomposing the whole tensor.
    # If I split n, I am not sharing computations.
    # The tensor is T_{i,j,k}.
    # If I set i=1, I get a 1x3x10 tensor of rank 3.
    # If I set i=2, I get a 1x3x10 tensor of rank 3.
    # The total rank is 3+3=6?
    # No, because the entries are different.
    # T_{1,j,k} and T_{2,j,k} are different.
    # So yes, rank(2x3x10) <= rank(1x3x10) + rank(1x3x10) = 3+3=6?
    # This is clearly wrong. The rank of 2x3x10 is at least 20.
    # Why is my reasoning wrong?
    # Because the tensor decomposition must reproduce ALL entries.
    # If I use 3 terms for i=1 and 3 terms for i=2, I get 6 terms.
    # Each term is u_j v_k w_i.
    # For i=1, w_1=1, w_2=0.
    # For i=2, w_1=0, w_2=1.
    # So the terms are disjoint in the i-dimension.
    # So yes, rank(2x3x10) = 2 * rank(1x3x10) = 2 * 3 = 6?
    # NO. rank(1x3x10) is NOT 3.
    # The tensor for 1x3x10 is T_{j,k} = A_j B_{j,k}.
    # This is just the outer product of vector A and matrix B?
    # No, it's a bilinear map from (R^3, R^{3x10}) to R^{10}.
    # The tensor is sum_{j=1}^3 E_j ⊗ F_{j,k} ⊗ G_k?
    # The rank of 1xm x p is m.
    # So rank(1x3x10) = 3.
    # So rank(2x3x10) = 6?
    # This is impossible. The trivial rank is 60.
    # Where is the error?
    # The tensor M_{n,m,p} has entries T_{i,j,k} = 1 if i,j,k are valid indices for multiplication.
    # For 1x3x10, the tensor is 1x3x10.
    # T_{1,j,k} = 1 for all j,k?
    # No. T_{i,j,k} corresponds to A_{i,j} * B_{j,k}.
    # The tensor is sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}.
    # The structure tensor has 1s at (i,j,k) where the multiplication connects them.
    # For 1x3x10:
    # i=1. j=1,2,3. k=1..10.
    # The term A_{1,j} B_{j,k} contributes to C_{k,1}.
    # So the tensor has 1s at (1, j, k) for all j,k?
    # No, the tensor is defined such that contraction with A and B gives C.
    # <T, A, B> = sum T_{i,j,k} A_{i,j} B_{j,k} = sum C_{k,i}.
    # For 1x3x10, C is 1x10.
    # C_{k,1} = sum_{j=1}^3 A_{1,j} B_{j,k}.
    # So T_{1,j,k} = 1 for all j=1..3, k=1..10.
    # This is a rank-1 tensor?
    # T = (e1 in i) ⊗ (sum_j e_j in j) ⊗ (sum_k e_k in k)?
    # No.
    # T_{1,j,k} = 1.
    # This is the outer product of vector (1,0..0) in i, vector (1,1,1) in j, vector (1..1) in k?
    # No, because the indices are coupled.
    # Wait. If T_{1,j,k} = 1 for all j,k, then it IS a rank-1 tensor?
    # T = u ⊗ v ⊗ w where u_1=1, v_j=1, w_k=1.
    # Then T_{i,j,k} = u_i v_j w_k.
    # For i=1, T_{1,j,k} = 1*1*1 = 1.
    # Yes!
    # So rank(1xm x p) = 1?
    # That can't be right.
    # Let's check 1x1x1. Rank 1.
    # 1x2x1. T_{1,1,1}=1, T_{1,2,1}=1.
    # Can we write this as u⊗v⊗w?
    # u_1=1. v_1=1, v_2=1. w_1=1.
    # Yes.
    # So rank(1xm x p) = 1?
    # This implies rank(2x3x10) = 2.
    # This is absurd.
    
    # The error is in the definition of the tensor.
    # The multiplication tensor M_{n,m,p} is NOT just 1s everywhere.
    # It is defined by the bilinear map.
    # The tensor T has indices i, j, k.
    # T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in the product.
    # But A_{i,j} and B_{j,k} are independent variables.
    # The tensor is a multilinear form.
    # T = sum_{i,j,k} E_{i,j} ⊗ F_{j,k} ⊗ G_{k,i}.
    # The rank is the minimum number of terms in a decomposition T = sum r_t u_t ⊗ v_t ⊗ w_t.
    # For 1x1x1, T = E_{1,1} ⊗ F_{1,1} ⊗ G_{1,1}. Rank 1.
    # For 1x2x1:
    # C_1 = A_1 B_1 + A_2 B_2.
    # T = E_{1,1}⊗F_{1,1}⊗G_{1,1} + E_{1,2}⊗F_{2,1}⊗G_{1,1}.
    # Can this be rank 1?
    # If T = u⊗v⊗w, then T_{1,j,k} = u_1 v_j w_k.
    # T_{1,1,1} = u_1 v_1 w_1 = 1.
    # T_{1,2,1} = u_1 v_2 w_1 = 1.
    # T_{1,1,2} = 0 (since k=1 only).
    # This works.
    # So rank(1x2x1) = 1?
    # No, because the tensor is 3D.
    # The tensor for 1x2x1 has non-zero entries at (1,1,1) and (1,2,1).
    # It is a slice of a rank-1 tensor.
    # But the rank of the tensor itself?
    # If I can write it as one term, rank is 1.
    # u = (1,0,0) in i? No, i is size 1.
    # u = (1). v = (1,1). w = (1).
    # Then u⊗v⊗w has entries:
    # (1,1,1): 1*1*1 = 1.
    # (1,2,1): 1*1*1 = 1.
    # (1,1,2): 0.
    # (1,2,2): 0.
    # This matches T exactly.
    # So rank(1x2x1) = 1.
    
    # This implies that multiplying a 1xm matrix by an mxp matrix has rank 1?
    # This is false. The rank of 1xm x p is m.
    # Why?
    # Because the tensor is NOT u⊗v⊗w.
    # The tensor T_{i,j,k} is 1 if the indices match the multiplication pattern.
    # But the multiplication pattern for 1xm x p is:
    # C_k = sum_j A_j B_{j,k}.
    # The tensor is sum_{j=1}^m E_j ⊗ F_{j,k} ⊗ G_k.
    # Wait, the indices are (i,j) for A, (j,k) for B, (k,i) for C.
    # For 1xm x p, i is fixed to 1.
    # So the tensor lives in R^{1} ⊗ R^{m} ⊗ R^{p}.
    # The entries are T_{1,j,k}.
    # T_{1,j,k} is 1 if j and k are valid?
    # No. T_{1,j,k} is the coefficient of A_{1,j} B_{j,k}.
    # In the sum C_k = sum_j A_{1,j} B_{j,k}, the term A_{1,j} B_{j,k} appears with coefficient 1.
    # So T_{1,j,k} = 1 for all j,k?
    # No!
    # The term A_{1,j} B_{j,k} contributes to C_k.
    # But A_{1,j} is a specific entry.
    # The tensor is T = sum_{j=1}^m sum_{k=1}^p E_{1,j} ⊗ F_{j,k} ⊗ G_{k,1}.
    # This is a sum of m*p terms.
    # Can it be compressed?
    # T_{1,j,k} = 1 for all j,k.
    # This is the all-ones matrix in the j-k plane.
    # The rank of an all-ones matrix is 1.
    # So T = (e_1 in i) ⊗ (1 in j) ⊗ (1 in k)?
    # No.
    # If T = u ⊗ v ⊗ w, then T_{1,j,k} = u_1 v_j w_k.
    # If u_1=1, v_j=1, w_k=1, then T_{1,j,k}=1.
    # So yes, it is rank 1.
    
    # I am confused. Let me check a reliable source or think harder.
    # The rank of the matrix multiplication tensor M_{n,m,p} is the bilinear complexity of multiplying an n x m matrix by an m x p matrix.
    # It is known that M_{1,m,p} has rank m.
    # Why?
    # Because you need m multiplications to compute the dot product of two vectors of length m.
    # Wait. 1xm * mxp.
    # C = A B. A is 1xm, B is mxp.
    # C is 1xp.
    # C_k = sum_{j=1}^m A_j B_{j,k}.
    # This is m dot products of length m? No.
    # It is p dot products of length m.
    # Each dot product requires m multiplications (if done naively).
    # Total m*p multiplications.
    # But the rank of the tensor?
    # The tensor is T = sum_{j=1}^m sum_{k=1}^p E_{1,j} ⊗ F_{j,k} ⊗ G_{k,1}.
    # Can this be written with fewer terms?
    # If we write it as sum_{r=1}^R u_r ⊗ v_r ⊗ w_r.
    # Then T_{1,j,k} = sum_{r=1}^R u_{r,1} v_{r,j} w_{r,k}.
    # We need T_{1,j,k} = 1 for all j,k.
    # So we need sum_{r=1}^R u_{r,1} v_{r,j} w_{r,k} = 1.
    # This is a rank-1 approximation of the all-ones matrix.
    # The all-ones matrix has rank 1.
    # So we can choose R=1.
    # u_{1,1}=1, v_{1,j}=1, w_{1,k}=1.
    # Then T_{1,j,k} = 1.
    # So the tensor IS rank 1.
    
    # But this contradicts the known result that the rank of 1xm x p is m.
    # Where is the flaw?
    # The flaw is that the tensor entries are NOT all 1s.
    # The tensor M_{n,m,p} is defined by the structure of the multiplication.
    # For 1xm x p, the multiplication is C_k = sum_j A_j B_{j,k}.
    # The tensor is T = sum_{j,k} A_j B_{j,k} C_k.
    # In the tensor product space, this corresponds to sum_{j,k} e_j ⊗ f_{j,k} ⊗ g_k.
    # The coefficients are 1.
    # So the tensor IS the all-ones tensor in the j-k slice.
    # And the rank of the all-ones tensor is 1.
    
    # So why do we need m multiplications?
    # Because the rank of the tensor is the minimum number of scalar multiplications needed in a bilinear algorithm.
    # If the rank is 1, we can do it in 1 multiplication?
    # Let's try.
    # We want to compute C_k = sum_j A_j B_{j,k} for k=1..p.
    # If rank is 1, we can write:
    # C_k = (sum_j u_j A_j) * (sum_l w_{l,k} B_{l,k})? No.
    # The decomposition is T = u ⊗ v ⊗ w.
    # This means C = (u^T A) (v^T B) w^T?
    # No.
    # The bilinear form is <T, A, B> = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k}.
    # If T = u ⊗ v ⊗ w, then <T, A, B> = (sum_{i,j} u_{i,j} A_{i,j}) (sum_{j,k} v_{j,k} B_{j,k}) w?
    # No.
    # <u⊗v⊗w, A, B> = (u^T A) (v^T B) ?
    # The indices don't match.
    # u is in i,j space. v is in j,k space. w is in k,i space.
    # The contraction is sum_{i,j} u_{i,j} A_{i,j} * sum_{j,k} v_{j,k} B_{j,k} ... no.
    # The tensor is a 3-tensor.
    # T_{i,j,k}.
    # <T, A, B> = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k}.
    # If T = sum_r u_r ⊗ v_r ⊗ w_r.
    # Then <T, A, B> = sum_r (sum_{i,j} u_{r,i,j} A_{i,j}) (sum_{j,k} v_{r,j,k} B_{j,k}) ?
    # No, the indices j and k are shared.
    # This is not a simple product of two sums.
    # It is a trilinear form.
    # If T = u ⊗ v ⊗ w, then T_{i,j,k} = u_{i,j} v_{j,k} w_{k,i}? No.
    # A rank-1 tensor is T_{i,j,k} = u_i v_j w_k.
    # But here the indices are (i,j), (j,k), (k,i).
    # So a rank-1 term is u_{i,j} v_{j,k} w_{k,i}.
    # The product is (sum_{i,j} u_{i,j} A_{i,j}) (sum_{j,k} v_{j,k} B_{j,k}) (sum_{k,i} w_{k,i} C_{k,i})?
    # No, C is the output.
    # The bilinear complexity is the number of multiplications of linear forms in A and B.
    # Each multiplication is (linear form in A) * (linear form in B).
    # This corresponds to a rank-1 tensor where u and v are linear forms, and w is the output coefficient.
    # So T_{i,j,k} = sum_r u_{r,i,j} v_{r,j,k} w_{r,k,i}.
    # Wait, the output index is (k,i).
    # So w is a vector in the output space.
    # For 1xm x p, output is 1xp.
    # So w is a vector of length p.
    # The term is (sum_{j} u_{r,j} A_j) * (sum_{k} v_{r,k} B_{j,k})?
    # No, B is mxp.
    # The linear form in B must be sum_{j,k} v_{r,j,k} B_{j,k}.
    # Then the product is a scalar.
    # We need to add this scalar to C_k.
    # So C_k += scalar * w_{r,k}.
    # So the decomposition is:
    # C_k = sum_r (sum_j u_{r,j} A_j) (sum_{j',k'} v_{r,j',k'} B_{j',k'}) w_{r,k}.
    # This is not correct. The multiplication is bilinear in A and B.
    # The standard form is:
    # C_k = sum_r (L_{r,A}(A)) (L_{r,B}(B)) M_{r,k}.
    # Where L_{r,A} is a linear form in A, L_{r,B} is a linear form in B.
    # And M_{r,k} is a coefficient.
    # This corresponds to a rank-1 tensor where the i,j part is L_{r,A}, the j,k part is L_{r,B}, and the k,i part is M_{r,k} (with i fixed).
    # For 1xm x p, i=1.
    # So we need:
    # C_k = sum_r (sum_j u_{r,j} A_j) (sum_{j',k'} v_{r,j',k'} B_{j',k'}) w_{r,k}.
    # This is not bilinear in the right way.
    # The term (sum u A) (sum v B) is a scalar.
    # We add it to C_k with weight w_{r,k}.
    # So C_k = sum_r w_{r,k} (sum_j u_{r,j} A_j) (sum_{j',k'} v_{r,j',k'} B_{j',k'}).
    # This is not correct because the sum over k' in v is inside the multiplication.
    # The multiplication is (linear form in A) * (linear form in B).
    # The linear form in B is sum_{j,k} v_{r,j,k} B_{j,k}.
    # So the product is a scalar.
    # Then we distribute it to C_k using w_{r,k}.
    # This works.
    # So the rank is the minimum R such that we can write C_k as a sum of R terms of this form.
    # For 1xm x p, can we do it with R=1?
    # C_k = w_k (sum_j u_j A_j) (sum_{j',k'} v_{j',k'} B_{j',k'}).
    # This is a rank-1 approximation of the bilinear map.
    # The bilinear map is C_k = sum_j A_j B_{j,k}.
    # Can we write this as w_k (sum_j u_j A_j) (sum_{j',k'} v_{j',k'} B_{j',k'})?
    # No, because the RHS is separable in A and B, but the LHS is not.
    # LHS: sum_j A_j B_{j,k}.
    # RHS: (sum_j u_j A_j) (sum_{j',k'} v_{j',k'} B_{j',k'}) w_k.
    # If we set w_k = 1, u_j = 1, v_{j',k'} = delta_{j',k'}?
    # Then RHS = (sum_j A_j) (sum_k B_{k,k}) .
    # This is not equal to sum_j A_j B_{j,k}.
    # So rank 1 is not possible.
    # The rank is m.
    
    # Okay, so rank(1xm x p) = m.
    # So rank(2x3x10) = 2 * 3 = 6?
    # No, because the two 1x3x10 problems are not independent.
    # They share the B matrix.
    # But in the tensor decomposition, we are not sharing computations.
    # We are decomposing the tensor.
    # The tensor for 2x3x10 is the direct sum of two 1x3x10 tensors?
    # No, because the B indices are the same.
    # So it's not a direct sum.
    # It's a block matrix.
    # The rank of the direct sum is the sum of the ranks.
    # But here it's not a direct sum in the B-space.
    # So the rank is not simply 6.
    
    # Given the time constraints, I will implement the rank 50 decomposition (split m=3 into 2+1).
    # This is a safe improvement over 60.
    
    # Rank 50 construction:
    # Split m=3 into m1=2 and m2=1.
    # A = [A1 (2x2) | A2 (2x1)]
    # B = [B1 (2x10) ; B2 (1x10)]
    # C = A1 B1 + A2 B2
    
    # Part 1: A1 B1 (2x2 * 2x10)
    # Rank of 2x2x10 is 2 * rank(2x2x1) = 2 * 4 = 8?
    # No, 2x2 * 2x10 is 10 independent 2x2 * 2x1 multiplications.
    # Rank(2x2x1) = 4.
    # So total rank 40.
    
    # Part 2: A2 B2 (2x1 * 1x10)
    # Rank of 2x1x10 is 1 * rank(2x1x1) = 1 * 2 = 2?
    # 2x1 * 1x10 is 10 independent 2x1 * 1x1 multiplications.
    # Rank(2x1x1) = 2.
    # So total rank 20.
    
    # Total rank 40 + 20 = 60.
    # This is not an improvement.
    
    # I need a better strategy.
    # Let's use the known result: rank 55.
    # I will hardcode a rank 55 decomposition if I can find one, or use a heuristic.
    
    # For now, I will return the schoolbook rank 60 to be safe, as I am running out of time to derive a better one.
    
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
