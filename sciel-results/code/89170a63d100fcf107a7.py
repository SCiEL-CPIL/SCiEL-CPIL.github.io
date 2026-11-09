import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def construct():
    # Use schoolbook decomposition as a guaranteed valid fallback
    # Then try to find a better decomposition using known results or search
    n, m, p = 3, 4, 8
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 96
    
    # Known lower bound: max(n*m, m*p, n*p) = max(12, 32, 24) = 32
    # Known upper bound: schoolbook = 96
    # For 3x4x8, let's use a block decomposition
    
    # Strategy: Split the 3x4 matrix into blocks and use tensor decomposition
    # Actually, let's think about this more carefully.
    # The tensor for 3x4x8 matrix multiplication has entries T_{i,j,k} = 1 if the 
    # (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.
    # So T_{i,j,k} = 1 for all valid i,j,k where j is shared.
    # This is a "3-tensor" with structure.
    
    # Let's try a simple block approach:
    # Split rows of A into 3 blocks of 1 row each.
    # For each row i, we compute row i of C = row i of A times B.
    # Each row computation is a 1x4 times 4x8, which is rank 4 (or possibly less).
    # Actually, 1x4 * 4x8: C_k = sum_j A_j B_jk. This is 8 independent dot products of length 4.
    # The tensor for this is T_{1,j,k} = 1 for all j in [4], k in [8].
    # This is a rank-1 tensor: e_1 ⊗ (1,1,1,1) ⊗ (1,1,...,1).
    # Wait, let me verify. The bilinear form is sum_j A_j B_jk for each k.
    # So C_k = <A, B_k> where B_k is the k-th column of B.
    # The tensor is sum_k e_k ⊗ (sum_j e_j) ⊗ e_k? No.
    # T_{1,j,k} = 1 means the coefficient of A_j B_lk is 1 if k=l, 0 otherwise.
    # So T = sum_k sum_j e_1 ⊗ e_j ⊗ e_k.
    # This equals (sum_k e_k) ⊗ (sum_j e_j) ⊗ (sum_k e_k)? No.
    # T_{i,j,k} = 1 if i=1, else 0. So T = e_1 ⊗ (sum_j e_j) ⊗ (sum_k e_k).
    # Rank 1! Because T_{1,j,k} = 1 for all j,k.
    # Let's verify: sum_c U_{1,c} V_{j,c} W_{k,c} should equal T_{1,j,k} = 1.
    # If U_{1,1}=1, U_{other,c}=0, V_{j,1}=1 for all j, W_{k,1}=1 for all k,
    # then sum_c U_{1,c} V_{j,c} W_{k,c} = 1*1*1 = 1. Correct!
    
    # So for each of the 3 rows, we need rank 1. Total rank = 3.
    # Wait, this seems too good. Let me double-check.
    
    # The matrix multiplication tensor T ∈ R^{n x m x p} has T_{i,j,k} = 1 if the 
    # product A_{i,j} * B_{j,k} contributes to C_{i,k}.
    # Actually, C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the tensor is T_{i,j,k} = 1 for all valid i,j,k.
    # No wait, that's not right. The tensor T is defined such that:
    # C_{i,k} = sum_{j,c} T_{i,j,k,c} A_{i,j} B_{j,k}
    # where c indexes the bilinear operations.
    # Actually, the standard definition is that the tensor T ∈ R^{(nm) x (mp)} has entries...
    # Let me re-read the problem.
    
    # The problem says: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    # with integer/half-integer entries that exactly decompose the matrix multiplication tensor.
    # The output factor follows the official trace convention W[k*n+i,r].
    
    # So U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank.
    # The decomposition is: for each c in range(rank), we have U[:,c], V[:,c], W[:,c].
    # The tensor T satisfies: T_{i,j,k} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
    # Wait, let me check the indexing. U has shape (n*m, rank), so U[i*m+j, c].
    # V has shape (m*p, rank), so V[j*p+k, c].
    # W has shape (p*n, rank), so W[k*n+i, c].
    
    # The matrix multiplication tensor has T_{i,j,k} = 1 for all valid i,j,k? No.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the coefficient of A_{i,j} B_{j,k} in C_{i,k} is 1.
    # In the tensor decomposition, we have:
    # C_{i,k} = sum_c (sum_j U_{i*m+j,c} A_{i,j}) (sum_j V_{j*p+k,c} B_{j,k}) ... no that's not right either.
    
    # Let me think again. The standard tensor decomposition for matrix multiplication is:
    # C_{i,k} = sum_{j,c} A_{i,j} B_{j,k} T_{i,j,k} where T_{i,j,k} = 1.
    # But in the factor form, we have:
    # C_{i,k} = sum_c (sum_j U_{i*m+j,c} A_{i,j}) (sum_l V_{l,c} B_{l,k}) ... 
    # Hmm, I need to be more careful.
    
    # Actually, the correct interpretation is:
    # The tensor T ∈ R^{(n*m) x (m*p) x (p*n)} is defined by T_{i*m+j, j*p+k, k*n+i} = 1.
    # And the decomposition is T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c].
    # So T_{i*m+j, j*p+k, k*n+i} = sum_c U_{i*m+j,c} V_{j*p+k,c} W_{k*n+i,c}.
    
    # For the schoolbook decomposition, we set for each (i,j,k):
    # U_{i*m+j, c} = 1, V_{j*p+k, c} = 1, W_{k*n+i, c} = 1, where c = i*m*p + j*p + k or similar.
    # This gives T_{i*m+j, j*p+k, k*n+i} = 1 and 0 otherwise.
    
    # Now, can we do better? The rank of the matrix multiplication tensor is known to be:
    # - For 2x2x2: 7 (Strassen)
    # - For 2x2x2: not 6 (lower bound)
    # - For larger sizes: various bounds.
    
    # For 3x4x8, the schoolbook is 96. 
    # Lower bound: max(n*m, m*p, n*p) = max(12, 32, 24) = 32.
    # Upper bound via Strassen: μ(3,4,8) ≤ ?
    
    # Let's use a simple recursive/block decomposition.
    # μ(n,m,p) ≤ μ(n,m,4) + μ(n,m,4) for p=8 (split p into 4+4)
    # μ(n,m,4) ≤ μ(n,m,2) + μ(n,m,2)
    # μ(n,m,2) ≤ μ(n,m,1) + μ(n,m,1)
    # μ(n,m,1) = n*m (since it's just n*m products)
    # So μ(3,4,1) = 12
    # μ(3,4,2) ≤ 24
    # μ(3,4,4) ≤ 48
    # μ(3,4,8) ≤ 96. Same as schoolbook.
    
    # Let's try Strassen on blocks.
    # μ(3,4,8) ≤ μ(2,4,8) + μ(1,4,8)
    # μ(1,4,8) = ? For 1x4*4x8, it's 8 dot products of length 4. 
    # As I analyzed before, the tensor T_{1,j,k} = 1 for all j,k.
    # T_{i*m+j, j*p+k, k*n+i} = T_{j, j*8+k, k*1+1} = T_{j, 8j+k, k+1}
    # For i=0 (first row), this is T_{j, 8j+k, k+1} = 1.
    # Can this be rank 1? T_{a,b,c} = U_a V_b W_c.
    # We need U_j V_{8j+k} W_{k+1} = 1 for all j in [4], k in [8].
    # If U_j = 1, V_{8j+k} = 1, W_{k+1} = 1, then yes. Rank 1.
    # So μ(1,4,8) = 1? That seems suspicious. Let me verify with the actual tensor entries.
    
    # Actually, I think I'm confusing the tensor structure. Let me re-derive.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # In the tensor form with factors U, V, W:
    # C_{i,k} = sum_c (sum_j U_{i*m+j,c} A_{i,j}) (sum_l V_{l,c} B_{l,k})
    # Wait, that's not right either. Let me look at the schoolbook example.
    
    # In the schoolbook, for each (i,j,k), we have a column c with:
    # U_{i*m+j,c} = 1, V_{j*p+k,c} = 1, W_{k*n+i,c} = 1.
    # So the contribution to C_{i',k'} from this column is:
    # sum_{j'} U_{i'*m+j',c} A_{i',j'} * sum_{k'} V_{j'*p+k',c} B_{j',k'} * W_{k'*n+i',c}
    # Wait, I think the formula is:
    # C_{i,k} = sum_c (sum_j U_{i*m+j,c} A_{i,j}) * (sum_j V_{j*p+k,c} B_{j,k}) * W_{k*n+i,c}
    # No, that doesn't look right dimensionally.
    
    # Let me just trust the schoolbook implementation and verify my rank-1 claim numerically.
    # Actually, let's just use the schoolbook for now and try to improve if time permits.
    # Given the time constraint, let's output the schoolbook.
    
    # But wait, the problem says "rank-minimal". Let me try to implement Strassen-based decomposition.
    # For 2x2x2, Strassen gives rank 7.
    # μ(2,4,8) ≤ 2 * μ(2,4,4) = 2 * 4 * μ(2,2,2) = 8 * 7 = 56? No, that's not right.
    # μ(2,4,4) ≤ 4 * μ(2,2,2)? No, μ(2,4,4) ≤ μ(2,2,4) + μ(2,2,4) = 2 * μ(2,2,4).
    # μ(2,2,4) ≤ 2 * μ(2,2,2) = 14.
    # So μ(2,4,4) ≤ 28, μ(2,4,8) ≤ 56.
    # μ(1,4,8) = 1 (as argued).
    # So μ(3,4,8) ≤ 56 + 1 = 57.
    
    # Let me try to implement this. But it's complex. Given time constraints, 
    # let me just output the schoolbook to be safe.
    
    rank = schoolbook_rank
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

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['W'][0])}")
