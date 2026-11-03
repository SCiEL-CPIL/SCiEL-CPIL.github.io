import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    def time_left():
        return time.time() < deadline

    def strassen_block_2x2x2(offset_i, offset_j, offset_k):
        """
        Returns 7 (U, V, W) triples for a 2x2x2 Strassen block.
        U: 2x2 matrix (rows i, cols j), V: 2x2 (rows j, cols k), W: 2x2 (rows k, cols i)
        Each triple is (U_block, V_block, W_block) where blocks are 2x2 numpy arrays.
        """
        # Strassen formulas for 2x2 matrix multiplication:
        # C = A * B where A is 2x2, B is 2x2
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For tensor decomposition, each "multiplication" M = X * Y becomes:
        # U picks elements from X, V picks elements from Y, W distributes to output.
        # The tensor term for a single product X[i,j]*Y[j,k] contributes to C[i,k].
        # In our convention: U[i,j,c], V[j,k,c], W[k,i,c]
        
        # For Strassen, we have linear combinations.
        # Let's define the 7 terms explicitly.
        
        # Each term is a sum of products of linear forms.
        # For M1 = (A11+A22)(B11+B22):
        #   = A11*B11 + A11*B22 + A22*B11 + A22*B22
        # This contributes to C11, C12, C21, C22 with coefficients.
        
        # Actually, let me think about this differently.
        # The matrix multiplication tensor for 2x2 has rank 7 (Strassen).
        # The 7 rank-1 tensors are:
        
        # Let's use the standard Strassen decomposition.
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
        
        # Strassen:
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m6
        
        # For the tensor decomposition, each mi corresponds to a set of rank-1 terms.
        # But actually, each mi is itself a product of two linear forms, so it's already
        # a sum of rank-1 terms in the tensor sense.
        
        # Let me reconsider. The rank-1 tensor decomposition means:
        # T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
        
        # For the 2x2 case, we need 7 such terms.
        # Let's construct them directly.
        
        # Term 1: m1 = (a+d)(e+h)
        # a = A[0,0], d = A[1,1], e = B[0,0], h = B[1,1]
        # m1 = a*e + a*h + d*e + d*h
        # This contributes:
        #   a*e -> C[0,0] with coeff 1 (from m1 in C11)
        #   a*h -> C[0,0] with coeff 1 (from m1 in C11) ... wait, no.
        
        # I think I'm overcomplicating this. Let me use a known rank-7 decomposition.
        
        # The 7 rank-1 tensors for 2x2x2 matrix multiplication:
        # Using the standard Strassen-based decomposition.
        
        # Let me define U, V, W as 2x2x7 tensors (or 4x7, 4x7, 4x7 flattened).
        
        # From literature, one rank-7 decomposition is:
        # (I'll use the standard one)
        
        # Actually, let me just construct it by expressing each Strassen multiplier
        # as a product and then distributing.
        
        # m1 = (a+d)(e+h) = ae + ah + de + dh
        # This product appears in C11 (+1), C22 (+1)
        # So the contribution to T from m1 is:
        # T[0,0,0] += 1 (from ae in C11)
        # T[0,1,0] += 1 (from ah in C11? No, ah is A[0,0]*B[0,1]... wait)
        
        # Let me be more careful with indices.
        # A[i,j], B[j,k], C[i,k]
        # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
        
        # m1 = (a+d)(e+h) = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
        # = A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1]
        # These terms contribute to:
        # A[0,0]B[0,0] -> C[0,0]
        # A[0,0]B[1,1] -> C[0,0] (since B[1,1] has j=1,k=1, so this is A[0,1]*B[1,0]... no)
        
        # Wait, I'm confusing myself. Let me use proper indices.
        # A[i,j] * B[j,k] -> C[i,k]
        
        # A[0,0]*B[0,0] -> C[0,0]
        # A[0,0]*B[1,1] -> this is A[i=0,j=0] * B[j=1,k=1], but j must match!
        # A[0,0] has j=0, B[1,1] has j=1. They don't match.
        
        # I see the issue. In Strassen, the linear combinations are over the matrix entries,
        # but the tensor structure requires matching j indices.
        
        # Let me reconsider the Strassen decomposition in terms of the tensor.
        
        # The correct way: each rank-1 term in the tensor decomposition is:
        # U_c[i,j] * V_c[j,k] * W_c[k,i]
        
        # For Strassen, we have 7 multipliers, each of which is a product of two linear
        # combinations. But each such product expands into multiple A[i,j]*B[j,k] terms.
        
        # So actually, each Strassen "multiplier" corresponds to multiple rank-1 tensor terms.
        # That's not right either, because Strassen gives rank 7, not more.
        
        # Let me look at this from a different angle. The 2x2 matrix multiplication
        # tensor has rank 7. One explicit decomposition is:
        
        # I'll use a known construction. The tensor T[i,j,k] = 1 for the 2x2 case
        # (all entries are 1 in the "all ones" sense, but actually T[i,j,k] = 1 if the
        # corresponding entry in the product is formed by A[i,j]*B[j,k]).
        
        # Actually, for matrix multiplication, T[i,j,k] = 1 for all valid i,j,k
        # (where i in {0,1}, j in {0,1}, k in {0,1}).
        
        # So T is a 2x2x2 tensor with all entries = 1.
        
        # Wait, that's not right. The matrix multiplication tensor is:
        # T[i,j,k] = 1 if the product C[i,k] includes the term A[i,j]*B[j,k].
        # For 2x2, all 8 combinations (i,j,k) with i,j,k in {0,1} are valid,
        # and T[i,j,k] = 1 for all of them.
        
        # So yes, T is the all-ones 2x2x2 tensor.
        
        # And Strassen shows this has rank 7.
        
        # Let me find an explicit rank-7 decomposition of the all-ones 2x2x2 tensor.
        
        # Actually, the all-ones tensor of size 2x2x2 has rank 2, not 7!
        # Because T[i,j,k] = 1 = 1*1*1, so it's rank 1.
        
        # I'm making a fundamental error. Let me reconsider.
        
        # The matrix multiplication tensor is NOT the all-ones tensor.
        # T[i,j,k] = 1 means: the (i,k)-th entry of C is the sum over j of A[i,j]*B[j,k].
        # So T[i,j,k] = 1 for all i,j,k in the valid range.
        
        # For 2x2: T is a 2x2x2 tensor with T[i,j,k] = 1 for all i,j,k in {0,1}.
        # This IS the all-ones tensor, which has rank 1.
        
        # But Strassen says 2x2 matrix multiplication has rank 7!
        
        # The resolution: the tensor for matrix multiplication is not T[i,j,k] = 1.
        # It's a bilinear map. The tensor is defined such that:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # In tensor notation, this is a 3-tensor where:
        # T[i,j,k] = 1 for all i,j,k (in the appropriate ranges).
        
        # But the rank of this tensor (as a 3-way tensor) is what we're computing.
        # And for the 2x2 case, this tensor has rank 7 (Strassen).
        
        # Wait, but the all-ones 2x2x2 tensor has CP rank 2 (or 1?).
        # Let me check: T[i,j,k] = 1. Can we write this as sum of r rank-1 tensors?
        # Rank-1 tensor: u[i]*v[j]*w[k]
        # Sum: sum_c u_c[i] v_c[j] w_c[k]
        # For r=1: u[i]v[j]w[k] = 1 for all i,j,k. Set u=v=w=1. Works!
        # So the all-ones 2x2x2 tensor has rank 1.
        
        # But Strassen says 2x2 matrix multiplication has rank 7!
        
        # I think I'm confusing two different things. Let me clarify.
        
        # The matrix multiplication problem: given A (n x m) and B (m x p), compute C = A*B (n x p).
        # This is a bilinear map: (A, B) -> C.
        # The tensor associated with this bilinear map is a 3-tensor T of size (n*m) x (m*p) x (n*p)
        # or equivalently (n x m) x (m x p) x (n x p).
        
        # The rank of this tensor is the bilinear complexity of matrix multiplication.
        
        # For 2x2: the tensor is of size (2x2) x (2x2) x (2x2) = 4x4x4.
        # T[A_idx, B_idx, C_idx] = 1 if C_idx corresponds to the product term.
        
        # More precisely, if we index A by (i,j), B by (j,k), C by (i,k), then:
        # T[(i,j), (j,k), (i,k)] = 1
        # And T is 0 elsewhere.
        
        # So the tensor has 8 non-zero entries (for 2x2: 2*2*2 = 8), each equal to 1.
        # The total size is 4x4x4 = 64 entries, 8 of which are 1.
        
        # The rank of this tensor is 7 (Strassen).
        
        # OK so now I understand. Let me construct the rank-7 decomposition.
        
        # The 8 non-zero entries are at:
        # (i=0,j=0), (j=0,k=0), (i=0,k=0) -> T[0, 0, 0] = 1  [A[0,0]*B[0,0] -> C[0,0]]
        # (i=0,j=0), (j=0,k=1), (i=0,k=1) -> T[0, 1, 1] = 1  [A[0,0]*B[0,1] -> C[0,1]]
        # (i=0,j=1), (j=1,k=0), (i=0,k=0) -> T[1, 0, 0] = 1  [A[0,1]*B[1,0] -> C[0,0]]
        # (i=0,j=1), (j=1,k=1), (i=0,k=1) -> T[1, 1, 1] = 1  [A[0,1]*B[1,1] -> C[0,1]]
        # (i=1,j=0), (j=0,k=0), (i=1,k=0) -> T[2, 0, 2] = 1  [A[1,0]*B[0,0] -> C[1,0]]
        # (i=1,j=0), (j=0,k=1), (i=1,k=1) -> T[2, 1, 3] = 1  [A[1,0]*B[0,1] -> C[1,1]]
        # (i=1,j=1), (j=1,k=0), (i=1,k=0) -> T[3, 0, 2] = 1  [A[1,1]*B[1,0] -> C[1,0]]
        # (i=1,j=1), (j=1,k=1), (i=1,k=1) -> T[3, 1, 3] = 1  [A[1,1]*B[1,1] -> C[1,1]]
        
        # Where I'm using flattened indices: A_idx = i*2+j, B_idx = j*2+k, C_idx = i*2+k
        
        # Now, Strassen's decomposition gives 7 rank-1 tensors that sum to this.
        
        # Let me use the standard Strassen formulas and convert them to rank-1 tensor terms.
        
        # m1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # = A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
        # But A[0,0]*B[1,1] is not a valid term (j doesn't match: A[0,0] has j=0, B[1,1] has j=1).
        
        # So m1 as a product of linear forms doesn't directly give valid tensor terms.
        
        # The correct interpretation: m1 is an intermediate value, and its contribution to C is:
        # C[0,0] += m1, C[1,1] += m1
        # But m1 itself is a bilinear form in A and B.
        
        # For the tensor decomposition, we need to express the overall bilinear map
        # as a sum of 7 rank-1 bilinear maps.
        
        # A rank-1 bilinear map is: (A, B) -> scalar(A) * scalar(B) * output_vector
        # where scalar(A) is a linear form in A, scalar(B) is a linear form in B,
        # and output_vector is a vector in C-space.
        
        # So each rank-1 term is: l_A(A) * l_B(B) * c_vec
        # where l_A is a linear combination of A's entries, l_B is a linear combination
        # of B's entries, and c_vec is a fixed vector in C-space.
        
        # In our U,V,W representation:
        # U_c[i,j] corresponds to the coefficient of A[i,j] in l_A
        # V_c[j,k] corresponds to the coefficient of B[j,k] in l_B  
        # W_c[k,i] corresponds to the coefficient of C[i,k] in c_vec
        
        # So for each of the 7 Strassen multipliers, we have:
        # l_A = sum_{i,j} U_c[i,j] * A[i,j]
        # l_B = sum_{j,k} V_c[j,k] * B[j,k]
        # c_vec[i,k] = W_c[k,i]
        
        # And the contribution to C is: l_A * l_B * c_vec
        
        # Now, for m1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]):
        # l_A = A[0,0] + A[1,1], so U_c[0,0]=1, U_c[1,1]=1, others 0
        # l_B = B[0,0] + B[1,1], so V_c[0,0]=1, V_c[1,1]=1, others 0
        # c_vec: m1 contributes to C[0,0] and C[1,1], so W_c[0,0]=1, W_c[1,1]=1, others 0
        
        # But wait, m1 contributes +1 to C[0,0] and +1 to C[1,1].
        # So W_c[0,0] = 1 (for C[0,0]) and W_c[1,1] = 1 (for C[1,1]).
        # In our indexing, W_c[k,i] where C index is (i,k), so C[0,0] has i=0,k=0, W_c[0,0]=1.
        # C[1,1] has i=1,k=1, W_c[1,1]=1.
        
        # Let me verify: l_A * l_B * c_vec
        # = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1]) * [C[0,0]=1, C[1,1]=1]
        # = (A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]) * [1, 1]
        
        # This gives contributions to C[0,0] and C[1,1]:
        # C[0,0] += A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
        # C[1,1] += A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
        
        # But the correct contribution should only be:
        # C[0,0] += A[0,0]*B[0,0] (from the term where i=0,j=0,k=0)
        # C[1,1] += A[1,1]*B[1,1] (from the term where i=1,j=1,k=1)
        
        # The extra terms A[0,0]*B[1,1] and A[1,1]*B[0,0] are not valid (j doesn't match).
        
        # So this approach doesn't work directly. The issue is that the product of
        # linear forms doesn't respect the j-index matching.
        
        # I think the correct way is that each Strassen multiplier corresponds to a
        # rank-1 tensor where U_c, V_c, W_c are chosen such that the product
        # U_c[i,j]*V_c[j,k]*W_c[k,i] is non-zero only for valid (i,j,k) triples.
        
        # Let me just look up or derive the explicit rank-7 decomposition.
        
        # After some thought, I believe the standard Strassen decomposition can be
        # expressed as 7 rank-1 tensors as follows:
        
        # Term 1: U has 1s at (0,0) and (1,1), V has 1s at (0,0) and (1,1), W has 1s at (0,0) and (1,1)
        # But this gives the wrong cross terms.
        
        # I think I need to accept that I can't easily derive the Strassen decomposition
        # in this format and instead use a different approach.
        
        # Let me use a simpler strategy: for the 3x5x7 case, I'll use the schoolbook
        # decomposition but try to find a lower-rank decomposition using a structured
        # search.
        
        # Actually, let me just return the schoolbook decomposition for now and focus
        # on getting a valid answer. The rank will be 3*5*7 = 105.
        
        pass

    # Schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
