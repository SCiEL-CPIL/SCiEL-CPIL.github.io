import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The tensor for matrix multiplication C = A * B
    # where A is n x m, B is m x p, C is n x p.
    # Indexing convention:
    # U index: u = i * m + j  (i in [0, n), j in [0, m))
    # V index: v = j * p + k  (j in [0, m), k in [0, p))
    # W index: w = k * n + i  (k in [0, p), i in [0, n))
    # 
    # The matrix multiplication tensor T is defined by:
    # T_{u, v, w} = 1 if the indices correspond to a valid multiplication step A_{i,j} * B_{j,k} -> C_{i,k}
    # Specifically, for each (i, j, k), the term contributes to:
    #   A_{i,j} (index u = i*m+j)
    #   B_{j,k} (index v = j*p+k)
    #   C_{i,k} (index w = k*n+i)
    #
    # This is a direct sum of p independent n x m x 1 tensors (since for each k, we multiply column k of B with A).
    # Actually, it's a direct sum of n*m*p scalar multiplications? No.
    # Let's look at the structure.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This is a bilinear form.
    # The rank of the matrix multiplication tensor is the minimum number of multiplications.
    # For n=2, m=2, p=12.
    # We can view this as 12 independent 2x2x1 multiplications (each column of C is A * column of B).
    # The rank of 2x2x1 matrix multiplication is known to be 7 (Strassen).
    # So the total rank should be 12 * 7 = 84?
    # Wait, is it 12 independent tensors?
    # The tensor is T = sum_{i,j,k} e_{i*m+j}^A e_{j*p+k}^B e_{k*n+i}^C.
    # This is a direct sum over k of tensors T_k, where T_k = sum_{i,j} e_{i*m+j}^A e_{j*p+k}^B e_{k*n+i}^C.
    # Since the A and B indices overlap?
    # A index: i*m+j. For different k, the A index is the same for the same i,j.
    # B index: j*p+k. For different k, the B index is different.
    # C index: k*n+i. For different k, the C index is different.
    # So the tensors T_k share the A part, but have distinct B and C parts.
    # This is NOT a direct sum in the sense of independent tensor spaces because the A indices are shared.
    # However, we can factor out the A part? No, tensor rank is not additive in that way if they share indices.
    #
    # Let's reconsider.
    # The multiplication is C = A * B.
    # A is 2x2, B is 2x12.
    # We can write B as [B_0, B_1] where B_0 and B_1 are 2x6.
    # Then C = [A*B_0, A*B_1].
    # This is a block diagonal structure?
    # No, it's just that the input B is partitioned.
    #
    # Let's use the known result for rectangular matrix multiplication.
    # The rank of M_{n,m,p} is at least max(n*m, m*p, p*n). Here max(4, 24, 24) = 24.
    # The trivial rank is n*m*p = 48.
    # Strassen's algorithm gives rank 7 for 2x2x2.
    # For 2x2x12, we can think of it as 6 independent 2x2x2 multiplications?
    # If we split B into 6 blocks of size 2x2, say B = [B_0, ..., B_5].
    # Then C = [C_0, ..., C_5] where C_k = A * B_k.
    # Each C_k = A * B_k is a 2x2x2 multiplication.
    # The tensors for these 6 multiplications are:
    # T_k = sum_{i,j} e_{i*m+j}^A e_{j*p + k*2 + l}^B e_{(k*2+l)*n + i}^C
    # Note that the A index i*m+j is the SAME for all k.
    # The B index is different for each k.
    # The C index is different for each k.
    # Since the A indices are shared, we cannot simply add the ranks.
    # However, we can view the whole thing as a single tensor.
    #
    # Actually, there is a known construction.
    # The rank of M_{2,2,p} is 7 * ceil(p/2) if p is even? No.
    # Let's check small cases.
    # M_{2,2,1}: rank 7.
    # M_{2,2,2}: Is it 14? Or can we do better?
    # If we use Strassen for 2x2x2, we need 7 multiplications.
    # For 2x2x4, we have two independent 2x2x2 multiplications.
    # T = T_0 + T_1.
    # T_0 uses A indices {0,1,2,3}, B indices {0,1,2,3}, C indices {0,1,2,3}.
    # T_1 uses A indices {0,1,2,3}, B indices {4,5,6,7}, C indices {4,5,6,7}.
    # Since B and C indices are disjoint, T_0 and T_1 are in orthogonal subspaces.
    # But A indices are shared.
    # The rank of a direct sum of tensors where only one subspace is shared is not simply the sum of ranks.
    #
    # Let's try a different perspective.
    # The tensor is T = sum_{i,j,k} e_{i*m+j} e_{j*p+k} e_{k*n+i}.
    # We can group by j.
    # T = sum_j ( sum_i e_{i*m+j} ) \otimes ( sum_k e_{j*p+k} ) \otimes ( sum_k e_{k*n+i} )? No, the indices are coupled.
    #
    # Let's use the fact that M_{2,2,p} is isomorphic to M_{2,2,1} \otimes I_p? No.
    #
    # Actually, let's look at the rank of M_{2,2,12}.
    # It is known that the rank of M_{2,2,n} is 7n for n >= 2? No.
    # For M_{2,2,2}, the rank is 14?
    # Let's check if we can do better than 48.
    # The trivial decomposition has rank 48.
    # Can we reduce it?
    #
    # Consider the case p=1. Rank is 7.
    # For p=2, we have two 2x2 matrices being multiplied by the same 2x2 matrix A.
    # C_1 = A * B_1, C_2 = A * B_2.
    # This is equivalent to multiplying a 2x2 matrix by a 2x4 matrix.
    # This is M_{2,2,4} if we group? No, M_{2,2,2} with p=2? No, p is the number of columns of B.
    # So M_{2,2,2} means A is 2x2, B is 2x2, C is 2x2.
    # Rank is 14?
    # Actually, the rank of M_{2,2,2} is 14. (2 * 7)
    # Why? Because the two columns of B are independent.
    # If we have A * [B_1, B_2], the tensor is T_1 + T_2 where T_k is the tensor for A * B_k.
    # Since the B and C indices for T_1 and T_2 are disjoint, but A indices are shared.
    # The rank of T_1 + T_2 is rank(T_1) + rank(T_2) = 14.
    # So for M_{2,2,p}, the rank is 7 * p?
    # If so, for p=12, rank = 84.
    # But the trivial rank is 48.
    # 84 > 48, so the trivial decomposition is better.
    #
    # So the minimal rank is at most 48.
    # Can we do better than 48?
    # The lower bound is max(2*2, 2*12, 12*2) = 24.
    # Is there a decomposition with rank between 24 and 48?
    #
    # Let's check M_{2,2,1}. Rank 7 < 4. Wait, 7 > 4.
    # M_{2,2,1}: A is 2x2, B is 2x1, C is 2x1.
    # Trivial rank: 2*2*1 = 4.
    # But Strassen says 7.
    # So trivial is better for p=1?
    # No, Strassen's algorithm is for 2x2x2 matrix multiplication (C = A * B where A, B, C are all 2x2).
    # The tensor M_{2,2,2} has rank 7.
    # My previous indexing was for M_{n,m,p} where A is n x m, B is m x p, C is n x p.
    # For M_{2,2,2}, n=2, m=2, p=2.
    # Trivial rank = 8.
    # Strassen rank = 7.
    #
    # For M_{2,2,1}, n=2, m=2, p=1.
    # A is 2x2, B is 2x1, C is 2x1.
    # Trivial rank = 4.
    # Is there a rank 3 decomposition?
    # The tensor is T = e_{0,0} e_{0,0} e_{0,0} + e_{0,1} e_{0,0} e_{0,1} + e_{1,0} e_{0,0} e_{1,0} + e_{1,1} e_{0,0} e_{1,1}?
    # No.
    # T_{u,v,w} = 1 if u=i*m+j, v=j*p+k, w=k*n+i.
    # For n=2, m=2, p=1:
    # u in 0..3, v in 0..1, w in 0..3.
    # (i,j,k) -> (u, v, w)
    # (0,0,0) -> (0, 0, 0)
    # (0,1,0) -> (1, 0, 2)
    # (1,0,0) -> (2, 1, 1)
    # (1,1,0) -> (3, 1, 3)
    # This is a sum of 4 rank-1 tensors.
    # Are they linearly independent? Yes.
    # So rank is 4.
    #
    # So for p=1, rank is 4.
    # For p=2, n=2, m=2:
    # Trivial rank = 8.
    # Can we do better?
    # The tensor is a direct sum of two M_{2,2,1} tensors?
    # T = T_{k=0} + T_{k=1}.
    # T_{k=0}: A indices 0..3, B indices 0..1, C indices 0..3 (specifically 0,2 for k=0? No, w=k*n+i. For k=0, w=i. For k=1, w=2+i).
    # B indices for k=0: v=j*2+0 = 0, 2.
    # B indices for k=1: v=j*2+1 = 1, 3.
    # C indices for k=0: w=0*i+0 = 0, 2? No, w=k*n+i.
    # k=0: w=i. i=0->0, i=1->1.
    # k=1: w=2+i. i=0->2, i=1->3.
    # A indices are shared.
    # B and C indices are disjoint.
    # The rank of T is rank(T_{k=0}) + rank(T_{k=1}) = 4 + 4 = 8?
    # Not necessarily, because the A part is shared.
    # But since B and C are in orthogonal subspaces, the tensors are in orthogonal subspaces with respect to B and C.
    # The rank is indeed 8.
    #
    # So for M_{2,2,p}, the rank is 4p?
    # If so, for p=12, rank = 48.
    #
    # Is it possible to do better than 4p?
    # For M_{2,2,2}, rank is 8.
    # But wait, M_{2,2,2} is NOT the same as the tensor for 2x2 * 2x2 -> 2x2?
    # Yes it is.
    # And its rank is 8.
    # But Strassen's algorithm has rank 7 for 2x2 * 2x2 -> 2x2.
    # Why the discrepancy?
    #
    # Ah! The indexing convention.
    # Strassen's algorithm multiplies two 2x2 matrices.
    # The tensor is T = sum_{i,j,k} e_{i,j}^A e_{j,k}^B e_{i,k}^C.
    # In my indexing:
    # u = i*m+j. For m=2, u = 2i+j.
    # v = j*p+k. For p=2, v = 2j+k.
    # w = k*n+i. For n=2, w = 2k+i.
    #
    # Let's check the terms for Strassen.
    # The 7 terms are combinations of sums of A and B.
    # The trivial decomposition has 8 terms.
    # Strassen reduces it to 7.
    # So the rank of M_{2,2,2} is 7.
    #
    # So my previous analysis that rank(M_{2,2,p}) = 4p is wrong for p=2.
    # For p=2, rank is 7.
    #
    # For p=4, is it 14?
    # If we have two independent 2x2 * 2x2 multiplications, the rank is 14.
    #
    # For p=12, we have 6 independent 2x2 * 2x2 multiplications.
    # So the rank is 6 * 7 = 42.
    #
    # Let's verify this.
    # Split B into 6 blocks of 2 columns each.
    # B = [B_0, B_1, ..., B_5].
    # C = [C_0, ..., C_5] where C_k = A * B_k.
    # The tensor is T = sum_{k=0}^5 T_k.
    # T_k = sum_{i,j} e_{i*m+j}^A e_{j*p + k*2 + l}^B e_{(k*2+l)*n + i}^C.
    # The A indices are shared.
    # The B indices for different k are disjoint.
    # The C indices for different k are disjoint.
    #
    # The rank of a sum of tensors T_k where the B and C indices are disjoint is the sum of the ranks.
    # Proof: If T = T_1 + T_2, and T_1, T_2 have disjoint B,C supports, then any rank-1 decomposition of T must decompose T_1 and T_2 separately?
    # Not necessarily, but the rank is at least the rank of each part.
    # Actually, if the supports are disjoint, the rank is additive.
    #
    # So rank(M_{2,2,12}) = 6 * rank(M_{2,2,2}) = 6 * 7 = 42.
    #
    # We need to construct a rank-42 decomposition.
    # We need the 7 Strassen terms for a 2x2 * 2x2 multiplication.
    #
    # Strassen's algorithm:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    #
    # We need to express this as a sum of 7 rank-1 tensors.
    #
    # Let's construct the U, V, W matrices for one 2x2 * 2x2 block.
    # Then we replicate this for each of the 6 blocks, adjusting the indices.
    
    def strassen_terms():
        # Returns U, V, W for M_{2,2,2} with rank 7.
        # Indices:
        # A: 0=a, 1=b, 2=c, 3=d
        # B: 0=e, 1=f, 2=g, 3=h
        # C: 0=C11, 1=C12, 2=C21, 3=C22
        #
        # Note: My global indexing for M_{2,2,2} is:
        # u = i*2+j
        # v = j*2+k
        # w = k*2+i
        #
        # Let's map the Strassen variables to these indices.
        # A: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        # B: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
        # C: C11=C[0,0], C12=C[0,1], C21=C[1,0], C22=C[1,1]
        
        terms = [
            # m1 = (a+d)(e+h)
            # A part: a+d -> [1, 0, 0, 1]
            # B part: e+h -> [1, 0, 0, 1]
            # C part: C11 += 1, C22 += 1 -> [1, 0, 0, 1]
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            # m2 = (c+d)e
            # A part: c+d -> [0, 0, 1, 1]
            # B part: e -> [1, 0, 0, 0]
            # C part: C21 += 1 -> [0, 0, 1, 0]
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
            # m3 = a(f-h)
            # A part: a -> [1, 0, 0, 0]
            # B part: f-h -> [0, 1, 0, -1]
            # C part: C12 += 1, C22 += 1 -> [0, 1, 0, 1]
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            # m4 = d(g-e)
            # A part: d -> [0, 0, 0, 1]
            # B part: g-e -> [-1, 0, 1, 0]
            # C part: C11 += 1, C21 += 1 -> [1, 0, 1, 0]
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            # m5 = (a+b)h
            # A part: a+b -> [1, 1, 0, 0]
            # B part: h -> [0, 0, 0, 1]
            # C part: C11 -= 1, C12 += 1 -> [-1, 1, 0, 0]
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            # m6 = (c-a)(e+f)
            # A part: c-a -> [-1, 0, 1, 0]
            # B part: e+f -> [1, 1, 0, 0]
            # C part: C22 += 1 -> [0, 0, 0, 1]
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            # m7 = (b-d)(g+h)
            # A part: b-d -> [0, 1, 0, -1]
            # B part: g+h -> [0, 0, 1, 1]
            # C part: C11 += 1, C12 -= 1 -> [1, -1, 0, 0]
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, -1, 0, 0])
        ]
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        for c, (u_vec, v_vec, w_vec) in enumerate(terms):
            U[:, c] = u_vec
            V[:, c] = v_vec
            W[:, c] = w_vec
            
        return U, V, W

    # Now, construct the full U, V, W for M_{2,2,12}.
    # n=2, m=2, p=12.
    # U size: 4 x 42
    # V size: 24 x 42
    # W size: 24 x 42
    
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    base_U, base_V, base_W = strassen_terms()
    
    for block in range(6):
        # Each block handles 2 columns of B and 2 rows of C?
        # No, 2 columns of B, so 2 rows of C.
        # B columns: 2*block, 2*block+1
        # C rows: 2*block, 2*block+1
        
        for c in range(7):
            col = block * 7 + c
            
            # A part is the same for all blocks
            U[:, col] = base_U[:, c]
            
            # B part:
            # base_V indices: 0=e, 1=f, 2=g, 3=h
            # Global B index: v = j*p + k
            # For block `block`, the B submatrix is B[:, 2*block : 2*block+2]
            # e = B[0, 2*block], f = B[0, 2*block+1]
            # g = B[1, 2*block], h = B[1, 2*block+1]
            # Global indices:
            # e: j=0, k=2*block -> v = 0*12 + 2*block = 2*block
            # f: j=0, k=2*block+1 -> v = 0*12 + 2*block+1 = 2*block+1
            # g: j=1, k=2*block -> v = 1*12 + 2*block = 12 + 2*block
            # h: j=1, k=2*block+1 -> v = 1*12 + 2*block+1 = 13 + 2*block
            
            V[2*block, col] += base_V[0, c]
            V[2*block+1, col] += base_V[1, c]
            V[12 + 2*block, col] += base_V[2, c]
            V[13 + 2*block, col] += base_V[3, c]
            
            # C part:
            # base_W indices: 0=C11, 1=C12, 2=C21, 3=C22
            # Global C index: w = k*n + i
            # For block `block`, the C submatrix is C[:, 2*block : 2*block+2]
            # C11 = C[0, 2*block], C12 = C[0, 2*block+1]
            # C21 = C[1, 2*block], C22 = C[1, 2*block+1]
            # Global indices:
            # C11: k=2*block, i=0 -> w = 2*block*2 + 0 = 4*block
            # C12: k=2*block+1, i=0 -> w = (2*block+1)*2 + 0 = 4*block+2
            # C21: k=2*block, i=1 -> w = 2*block*2 + 1 = 4*block+1
            # C22: k=2*block+1, i=1 -> w = (2*block+1)*2 + 1 = 4*block+3
            
            W[4*block, col] += base_W[0, c]
            W[4*block+2, col] += base_W[1, c]
            W[4*block+1, col] += base_W[2, c]
            W[4*block+3, col] += base_W[3, c]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
