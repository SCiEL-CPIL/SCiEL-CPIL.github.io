import time
import itertools
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    def check_decomposition(U, V, W):
        """Check if the decomposition is exact."""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for rr in range(r):
                        s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Strassen coefficients for 2x2x2
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # p1 = (a00+a11)*(b00+b11)
    # p2 = (a10+a11)*b00
    # p3 = a00*(b10-b11)
    # p4 = a11*(b00-b01)
    # p5 = (a00+a01)*b11
    # p6 = (a10-a00)*(b01+b11)
    # p7 = (a01-a11)*(b00+b10)
    
    # c00 = p1 + p4 - p5 - p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6

    # U_coeffs: shape (7, 4) where 4 is flattened A index (i*2+j)
    # V_coeffs: shape (7, 4) where 4 is flattened B index (j*2+k)
    # W_coeffs: shape (7, 4) where 4 is flattened C index (k*2+i) -> Note: W index is k*n+i
    
    # Local indices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> but global W is k*n+i
    
    # Let's define the local tensors for the 2x2x2 case
    # U_local[r, a_idx]
    # V_local[r, b_idx]
    # W_local[r, c_idx]
    
    U_local = np.zeros((7, 4))
    V_local = np.zeros((7, 4))
    W_local = np.zeros((7, 4))
    
    # r=0 (p1)
    # A: a00+a11 -> [1, 0, 0, 1]
    # B: b00+b11 -> [1, 0, 0, 1]
    # C: c00 += 1, c11 += 1 -> [1, 0, 0, 1]
    U_local[0] = [1, 0, 0, 1]
    V_local[0] = [1, 0, 0, 1]
    W_local[0] = [1, 0, 0, 1]
    
    # r=1 (p2)
    # A: a10+a11 -> [0, 0, 1, 1]
    # B: b00 -> [1, 0, 0, 0]
    # C: c10 += 1 -> [0, 0, 1, 0]
    U_local[1] = [0, 0, 1, 1]
    V_local[1] = [1, 0, 0, 0]
    W_local[1] = [0, 0, 1, 0]
    
    # r=2 (p3)
    # A: a00 -> [1, 0, 0, 0]
    # B: b10-b11 -> [0, 0, 1, -1]
    # C: c01 += 1 -> [0, 1, 0, 0]
    U_local[2] = [1, 0, 0, 0]
    V_local[2] = [0, 0, 1, -1]
    W_local[2] = [0, 1, 0, 0]
    
    # r=3 (p4)
    # A: a11 -> [0, 0, 0, 1]
    # B: b00-b01 -> [1, -1, 0, 0]
    # C: c00 += 1, c10 += 1 -> [1, 0, 1, 0]
    U_local[3] = [0, 0, 0, 1]
    V_local[3] = [1, -1, 0, 0]
    W_local[3] = [1, 0, 1, 0]
    
    # r=4 (p5)
    # A: a00+a01 -> [1, 1, 0, 0]
    # B: b11 -> [0, 0, 0, 1]
    # C: c00 -= 1, c01 += 1 -> [-1, 1, 0, 0]
    U_local[4] = [1, 1, 0, 0]
    V_local[4] = [0, 0, 0, 1]
    W_local[4] = [-1, 1, 0, 0]
    
    # r=5 (p6)
    # A: a10-a00 -> [-1, 0, 1, 0]
    # B: b01+b11 -> [0, 1, 0, 1]
    # C: c11 += 1 -> [0, 0, 0, 1]
    U_local[5] = [-1, 0, 1, 0]
    V_local[5] = [0, 1, 0, 1]
    W_local[5] = [0, 0, 0, 1]
    
    # r=6 (p7)
    # A: a01-a11 -> [0, 1, 0, -1]
    # B: b00+b10 -> [1, 0, 1, 0]
    # C: c00 -= 1 -> [-1, 0, 0, 0]
    U_local[6] = [0, 1, 0, -1]
    V_local[6] = [1, 0, 1, 0]
    W_local[6] = [-1, 0, 0, 0]
    
    def get_partitions(dim):
        """Get all ways to partition a dimension into blocks of size 1 or 2."""
        if dim == 0:
            return [()]
        res = []
        if dim >= 1:
            for rest in get_partitions(dim - 1):
                res.append((1,) + rest)
        if dim >= 2:
            for rest in get_partitions(dim - 2):
                res.append((2,) + rest)
        return res

    # We want to partition m (4) and p (6) into blocks of size 2 to use the 2x2x2 Strassen core.
    # n (2) is already size 2, so it's one block.
    
    # Partitions of m=4 into 2s: only (2, 2)
    # Partitions of p=6 into 2s: only (2, 2, 2)
    # So there is only 1 valid partition structure for a pure Kronecker product of 2x2x2 blocks.
    # However, we can also mix 1x1 blocks (schoolbook) and 2x2 blocks (Strassen).
    # But to minimize rank, we want as many 2x2 blocks as possible.
    # Rank of 2x2x2 Strassen is 7. Rank of 1x1x1 is 1.
    # Cost per element: Strassen 7/4 = 1.75. Schoolbook 1/1 = 1.0.
    # Wait, schoolbook is better per element?
    # No, the tensor rank of 2x2x2 multiplication is 7.
    # The tensor rank of 1x1x1 multiplication is 1.
    # If we use a 2x2 block, we replace 4 scalar multiplications (rank 4) with 7 rank-1 terms.
    # 7 > 4. So why is Strassen better?
    # Because for large matrices, $7(n/2)^3 = 7/8 n^3 < n^3$.
    # For small matrices, it depends.
    # Here, we have fixed dimensions.
    # Let's check the rank of the whole tensor.
    # If we use 2x2x2 blocks:
    # Number of blocks = (n/2) * (m/2) * (p/2) ? No.
    # The Kronecker product structure:
    # A is n x m. B is m x p. C is n x p.
    # If we partition A into blocks A_ij (2x2), B into blocks B_jk (2x2), C into blocks C_ik (2x2).
    # Then C_ik = sum_j A_ij B_jk.
    # This requires that the partition of m matches for A and B.
    # And the partition of p matches for B and C.
    # And the partition of n matches for A and C.
    
    # n=2: must be one block of size 2.
    # m=4: can be (2,2) or (1,1,1,1) or (2,1,1) etc.
    # p=6: can be (2,2,2) or ...
    
    # To use the 2x2x2 Strassen core, we need blocks of size 2 in all dimensions.
    # If we have a block of size 1, we use the 1x1x1 core (rank 1).
    # If we have a block of size 2, we use the 2x2x2 core (rank 7).
    # Can we mix them?
    # If A has a 2x1 block, B has a 1x2 block? No, the inner dimension m must match.
    # So if A is partitioned into sizes [s1, s2, ...] along m, B must be partitioned into the same sizes along m.
    # Similarly for p and n.
    
    # Let's enumerate partitions of n, m, p into 1s and 2s.
    # n=2: [(2,), (1,1)]
    # m=4: [(2,2), (2,1,1), (1,2,1), (1,1,2), (1,1,1,1)]
    # p=6: [(2,2,2), (2,2,1,1), ...]
    
    # For a given set of partitions (Pn, Pm, Pp), the rank is:
    # Rank = sum over blocks (i,j,k) of Rank(Core_{size_i, size_j, size_k})
    # Wait, the Kronecker product is not just a sum of independent cores for each block.
    # It is a single tensor constructed by Kronecker products of the local cores.
    # If we use a mixed partition, the tensor is a sum of terms where each term corresponds to a specific combination of block sizes?
    # Actually, if we partition the indices, the tensor decomposes into a direct sum of tensors corresponding to each block combination.
    # T_{ijk} = sum_{r} U_i_r V_j_r W_k_r
    # If we partition i into blocks I_a, j into J_b, k into K_c.
    # Then the tensor is the direct sum over a,b,c of T_{I_a, J_b, K_c}.
    # The rank of the direct sum is the sum of the ranks.
    # So we just need to find the partition that minimizes the sum of ranks of the constituent blocks.
    
    # Rank of 1x1x1 block: 1
    # Rank of 2x1x1 block: 1 (it's just multiplication)
    # Rank of 1x2x1 block: 1
    # Rank of 1x1x2 block: 1
    # Rank of 2x2x1 block: 2 (two 1x1x1 blocks? No, 2x2x1 is two independent 1x1x1s? No.)
    # A (2x2), B (2x1), C (2x1).
    # C = A B.
    # This is two 2x1 multiplications? No.
    # A is 2x2, B is 2x1. C is 2x1.
    # c0 = a00 b0 + a01 b1
    # c1 = a10 b0 + a11 b1
    # This is two independent scalar multiplications? No, it's two dot products.
    # Can we do it in rank 2?
    # p1 = a00 b0, p2 = a01 b1, p3 = a10 b0, p4 = a11 b1. Rank 4.
    # Can we do better?
    # This is equivalent to multiplying a 2x2 matrix by a 2x1 vector.
    # Tensor rank of 2x2x1 is 2?
    # Let's check.
    # U: 4xR, V: 2xR, W: 2xR.
    # If R=2:
    # u1 = [1,0,0,0], v1=[1,0], w1=[1,0] -> c0 += a00 b0
    # u2 = [0,1,0,0], v2=[0,1], w2=[1,0] -> c0 += a01 b1
    # u3 = [0,0,1,0], v3=[1,0], w3=[0,1] -> c1 += a10 b0
    # u4 = [0,0,0,1], v4=[0,1], w4=[0,1] -> c1 += a11 b1
    # This requires R=4.
    # Is there a trick?
    # No, 2x2x1 is just two 1x1x1 multiplications sharing the same B? No, B is shared.
    # Wait, B is 2x1.
    # c0 = a00 b0 + a01 b1
    # c1 = a10 b0 + a11 b1
    # This is rank 4.
    
    # So, any block involving a 2 in any dimension but not all 3 dimensions might have rank equal to the product of the sizes?
    # Let's check 2x2x2. Rank 7 < 8.
    # 2x2x1. Rank 4 = 4.
    # 2x1x1. Rank 1 = 2? No, 2x1x1 is c0=a00b0, c1=a10b0.
    # p1 = a00 b0, p2 = a10 b0.
    # w1 = [1,0], w2=[0,1].
    # u1=[1,0], u2=[0,1]. v1=[1], v2=[1].
    # Rank 2. Product 2*1*1=2.
    
    # So, it seems only 2x2x2 has a "win" (7 < 8).
    # Any other combination of 1s and 2s has rank = product of sizes.
    # Therefore, to minimize rank, we should use as many 2x2x2 blocks as possible.
    
    # n=2: Must be one block of size 2? Or two of size 1?
    # If we use (1,1) for n, we have two blocks of height 1.
    # If we use (2,) for n, we have one block of height 2.
    
    # Let's evaluate the cost of different partitions.
    # Cost of a block of size (ni, mj, pk) is ni*mj*pk.
    # Except if ni=mj=pk=2, cost is 7.
    
    # We need to partition n=2, m=4, p=6 into 1s and 2s.
    # Let Pn be the partition of n.
    # Let Pm be the partition of m.
    # Let Pp be the partition of p.
    
    # The total rank is sum_{i in Pn, j in Pm, k in Pp} Cost(i, j, k).
    
    # We want to minimize this sum.
    
    def cost(ni, mj, pk):
        if ni == 2 and mj == 2 and pk == 2:
            return 7
        return ni * mj * pk

    # Enumerate all partitions
    parts_n = get_partitions(n)
    parts_m = get_partitions(m)
    parts_p = get_partitions(p)
    
    best_rank = n * m * p
    best_partition = None
    
    for pn in parts_n:
        for pm in parts_m:
            for pp in parts_p:
                # Check if lengths match? No, the blocks are independent.
                # The tensor is the direct sum of blocks.
                # So we just sum the costs.
                rank = 0
                for ni in pn:
                    for mj in pm:
                        for pk in pp:
                            rank += cost(ni, mj, pk)
                if rank < best_rank:
                    best_rank = rank
                    best_partition = (pn, pm, pp)
                    
    if best_partition is None:
        result, rank = schoolbook()
        return result

    pn, pm, pp = best_partition
    rank = best_rank
    
    # Construct the decomposition
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Iterate through blocks
    # i_start, i_end for A and C
    # j_start, j_end for A and B
    # k_start, k_end for B and C
    
    i_pos = 0
    for ni in pn:
        j_pos = 0
        for mj in pm:
            k_pos = 0
            for pk in pp:
                # This block covers A[i_pos:i_pos+ni, j_pos:j_pos+mj]
                # B[j_pos:j_pos+mj, k_pos:k_pos+pk]
                # C[i_pos:i_pos+ni, k_pos:k_pos+pk]
                
                # Determine the local core
                if ni == 2 and mj == 2 and pk == 2:
                    # Use Strassen core
                    # Map local indices to global
                    # Local A index: i_loc * mj + j_loc ? No, local A is ni x mj.
                    # Local index for A: i_loc * mj + j_loc
                    # Local index for B: j_loc * pk + k_loc
                    # Local index for C: k_loc * ni + i_loc ? No, C is ni x pk.
                    # Global W index is k*n + i.
                    # Local C index (for W) should be k_loc * ni + i_loc.
                    
                    # U_local is (7, 4). 4 = ni*mj = 2*2.
                    # V_local is (7, 4). 4 = mj*pk = 2*2.
                    # W_local is (7, 4). 4 = pk*ni = 2*2.
                    
                    for r in range(7):
                        # U
                        for a_idx in range(4):
                            coeff = U_local[r, a_idx]
                            if coeff != 0:
                                i_loc = a_idx // mj
                                j_loc = a_idx % mj
                                global_i = i_pos + i_loc
                                global_j = j_pos + j_loc
                                U[global_i * m + global_j][col] += coeff
                                
                        # V
                        for b_idx in range(4):
                            coeff = V_local[r, b_idx]
                            if coeff != 0:
                                j_loc = b_idx // pk
                                k_loc = b_idx % pk
                                global_j = j_pos + j_loc
                                global_k = k_pos + k_loc
                                V[global_j * p + global_k][col] += coeff
                                
                        # W
                        for c_idx in range(4):
                            coeff = W_local[r, c_idx]
                            if coeff != 0:
                                # Local C index c_idx corresponds to k_loc * ni + i_loc
                                k_loc = c_idx // ni
                                i_loc = c_idx % ni
                                global_k = k_pos + k_loc
                                global_i = i_pos + i_loc
                                W[global_k * n + global_i][col] += coeff
                                
                        col += 1
                else:
                    # Schoolbook core
                    # Rank = ni * mj * pk
                    for i_loc in range(ni):
                        for j_loc in range(mj):
                            for k_loc in range(pk):
                                global_i = i_pos + i_loc
                                global_j = j_pos + j_loc
                                global_k = k_pos + k_loc
                                
                                U[global_i * m + global_j][col] = 1
                                V[global_j * p + global_k][col] = 1
                                W[global_k * n + global_i][col] = 1
                                col += 1
                                
                k_pos += pk
            j_pos += mj
        i_pos += ni
        
    if check_decomposition(U, V, W):
        return {"U": U, "V": V, "W": W}, rank
    
    # Fallback
    result, rank = schoolbook()
    return result

# EVOLVE-BLOCK-END
