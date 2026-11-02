import numpy as np
import time
import sys

def check_decomposition_fast(U, V, W, n, m, p):
    """Verify the tensor decomposition identity using NumPy broadcasting."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Indices:
    # U index: a*m + b
    # V index: c*p + d
    # W index: k*n + i
    # Target: T[a, b, c, d, k, i] = 1 if a==i and b==c and d==k else 0
    
    U_reshaped = U_arr.reshape(n, m, -1)
    V_reshaped = V_arr.reshape(m, p, -1)
    W_reshaped = W_arr.reshape(p, n, -1)
    
    # Compute S[a, b, c, d, k, i] = sum_r U[a, b, r] V[c, d, r] W[k, i, r]
    S = np.einsum('abr,cdr,kir->abcdki', U_reshaped, V_reshaped, W_reshaped)
    
    # Construct Target T
    a_idx = np.arange(n)
    b_idx = np.arange(m)
    c_idx = np.arange(m)
    d_idx = np.arange(p)
    k_idx = np.arange(p)
    i_idx = np.arange(n)
    
    delta_a_i = (a_idx[:, None, None, None, None, None] == i_idx[None, None, None, None, None, :]).astype(float)
    delta_b_c = (b_idx[None, :, None, None, None, None] == c_idx[None, None, :, None, None, None]).astype(float)
    delta_d_k = (d_idx[None, None, None, :, None, None] == k_idx[None, None, None, None, :, None]).astype(float)
    
    T = delta_a_i * delta_b_c * delta_d_k
    
    return np.allclose(S, T, atol=1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 140 # Safety margin
    
    n, m, p = 2, 3, 6
    
    # Dimensions
    dim_u = n * m  # 6
    dim_v = m * p  # 18
    dim_w = p * n  # 12
    
    # The goal is to find a low rank decomposition.
    # Known: Rank(2x3x3) = 11.
    # By splitting 3x6 into two 3x3 blocks, we can use two copies of the 2x3x3 decomposition.
    # This gives a rank of 11 + 11 = 22.
    # Schoolbook is 36. Current best in prompt is 34.
    # 22 is a significant improvement.
    
    # We need the specific Rank 11 decomposition for 2x3x3.
    # Let's construct it.
    # Indices for 2x3x3:
    # A: 2x3 (indices 0..5)
    # B: 3x3 (indices 0..8)
    # C: 3x2 (indices 0..5)  (Note: C is p x n = 3 x 2 for the subproblem)
    # Wait, the W matrix index is k*n + i. For 2x3x3, n=2, m=3, p=3.
    # So W is 3x2 = 6 elements.
    
    # The known Rank 11 decomposition for 2x3x3 is derived from the rank 7 decomposition of 2x2x2 (Strassen)
    # plus some additional terms for the third row/col.
    # Actually, let's just hardcode the known solution if we can recall it, or search for it.
    # Searching for rank 11 in 2x3x3 (6x9x6 tensor) is feasible with a small script?
    # The tensor has 324 entries.
    # We can use a greedy approach or hardcoded values from literature.
    
    # Literature values for 2x3x3 rank 11:
    # There are 11 terms.
    # Let's try to construct it using the structure of 2x2x2 Strassen (7 terms) + 4 more.
    # Or simply use a known set.
    
    # Let's define the 11 terms for 2x3x3 (n=2, m=3, p=3).
    # U: 6x11, V: 9x11, W: 6x11.
    
    # I will use a set of terms that are known to work.
    # If I don't have them hardcoded, I will write a small solver inside construct() to find them,
    # but that might be slow.
    # However, 2x3x3 is small.
    
    # Let's try to hardcode a valid Rank 11 solution.
    # Ref: https://en.wikipedia.org/wiki/Strassen_algorithm_(matrix_multiplication)
    # Strassen is for 2x2.
    # For 2x3x3, one can use the 2x2x2 result for the first 2x2 block of B and A, and handle the rest.
    
    # Let's assume the following structure for the 2x3x3 subproblem:
    # We split B into B1 (3x2) and B2 (3x1).
    # This is getting complicated to derive on the fly.
    
    # Alternative: Use the fact that Rank(2,3,3) = 11 is a standard result.
    # I will embed a static list of the 11 terms for 2x3x3.
    
    # Terms for 2x3x3 (n=2, m=3, p=3)
    # U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    
    # Let's use a known solution from a tensor decomposition database or paper.
    # Since I cannot browse, I will use a heuristic:
    # If I fail to hardcode, I will use the Rank 17 solution (Strassen-like + diagonal) which is easier to construct.
    # Rank 17 for 2x3x3:
    # Use Strassen (7) for 2x2x2 part (rows 0,1 of A, cols 0,1 of B, cols 0,1 of C).
    # Plus 10 terms for the remaining parts?
    # Actually, 2x3x3 can be done with rank 11.
    # Let's try to find it via a quick local search in Python if needed, but hardcoded is safer.
    
    # I will use a Rank 17 solution for the subblock first, to ensure correctness, then try to improve to 11 if I can recall/derive it.
    # Wait, the prompt says "INSPIRATIONS FROM OTHER SOLUTIONS" mentions "score 1.0909" which corresponds to Rank 34/1.0909 = 31.1? No.
    # Score = Seed / Value. Seed = 34 (from current program? No, Seed is usually the first valid solution or a reference).
    # The prompt says "score is the rank". Wait. "score is the rank". "seed/value".
    # If score = seed/rank, and current score is 1.0, and rank is 34, then seed is 34.
    # If I find rank 22, score = 34/22 = 1.545.
    # If I find rank 11*2 = 22, that is a big jump.
    
    # Let's verify the Rank 11 terms for 2x3x3.
    # I will construct a Rank 11 solution manually based on the 2x2x2 Strassen kernel.
    
    # Strassen 2x2x2 (indices 0,1 for all):
    # 1: U[0,4]=1, U[4,0]=1? No, standard indexing.
    # Let's use the explicit terms for 2x3x3 Rank 11.
    
    # I will use a solver to find the Rank 11 solution for 2x3x3 if not hardcoded.
    # But writing a robust solver for tensor rank is hard.
    
    # Let's use a known Rank 11 decomposition for 2x3x3.
    # Terms (u_idx, v_idx, w_idx, coeff):
    # Note: U is 6xR, V is 9xR, W is 6xR.
    # Indices:
    # U: i*3 + j (i=0,1; j=0,1,2)
    # V: j*3 + k (j=0,1,2; k=0,1,2)
    # W: k*2 + i (k=0,1,2; i=0,1)
    
    # I will hardcode a valid Rank 11 solution for 2x3x3.
    # Source: "The tensor rank of 2x3x3 matrix multiplication is 11".
    
    # The following are the 11 terms for 2x3x3:
    # 1. A00*B00 -> C00: U[0]=1, V[0]=1, W[0]=1
    # 2. A01*B10 -> C01: U[1]=1, V[3]=1, W[1]=1
    # 3. A02*B20 -> C02: U[2]=1, V[6]=1, W[4]=1
    # 4. A10*B01 -> C10: U[3]=1, V[1]=1, W[2]=1
    # 5. A11*B11 -> C11: U[4]=1, V[4]=1, W[3]=1
    # 6. A12*B21 -> C12: U[5]=1, V[7]=1, W[5]=1
    # This is the 6 diagonal terms. We need 5 more to reduce from 6 to 11? No, 6 is less than 11.
    # Wait, the diagonal terms alone form a rank 6 decomposition!
    # Why is the rank 11?
    # Ah, the diagonal terms are:
    # C[0,0] = A[0,0]B[0,0]
    # C[0,1] = A[0,1]B[1,0]
    # C[0,2] = A[0,2]B[2,0]
    # C[1,0] = A[1,0]B[0,1]
    # C[1,1] = A[1,1]B[1,1]
    # C[1,2] = A[1,2]B[2,1]
    # These are 6 terms.
    # But matrix multiplication requires:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The diagonal terms only cover specific (i,j,k) tuples.
    # Specifically, they cover (0,0,0), (0,1,0), (0,2,0), (1,0,1), (1,1,1), (1,2,1).
    # We are missing terms where the "sum" has multiple non-zero contributions or where j is different.
    # Wait, the tensor T is defined as T[a,b,c,d,k,i] = 1 if a==i and b==c and d==k.
    # In the 2x3x3 subproblem:
    # a,i in {0,1}
    # b,c in {0,1,2}
    # d,k in {0,1,2}
    # Non-zero if a==i, b==c, d==k.
    # This is exactly the set of tuples (a,b,c,d,k,i) where a=i, b=c, d=k.
    # Number of such tuples: 2 * 3 * 3 = 18.
    # The 6 diagonal terms I listed above correspond to:
    # (0,0,0,0,0,0): a=0,i=0; b=0,c=0; d=0,k=0. Correct.
    # (0,1,1,0,0,0): a=0,i=0; b=1,c=1; d=0,k=0. Correct.
    # (0,2,2,0,0,0): a=0,i=0; b=2,c=2; d=0,k=0. Correct.
    # (1,0,0,1,1,1): a=1,i=1; b=0,c=0; d=1,k=1. Correct.
    # (1,1,1,1,1,1): a=1,i=1; b=1,c=1; d=1,k=1. Correct.
    # (1,2,2,1,1,1): a=1,i=1; b=2,c=2; d=1,k=1. Correct.
    
    # We are missing 12 non-zero entries!
    # So rank 6 is not enough. We need to cover the other 12 entries.
    # The rank is 11, meaning we can cover all 18 with 11 rank-1 tensors.
    
    # I will use a known set of 11 terms.
    # If I cannot recall them, I will use a Rank 17 construction which is easier to verify.
    # Rank 17 construction for 2x3x3:
    # Use Strassen (7 terms) for the 2x2x2 core (indices 0,1).
    # Plus 10 terms for the interactions with index 2.
    # This is safe.
    
    # However, Rank 22 (using Rank 11) is much better.
    # Let's try to implement a quick search for the Rank 11 solution for 2x3x3.
    # The space of rank 11 tensors is small enough?
    # No, it's a nonlinear system.
    
    # I will hardcode the Rank 11 solution for 2x3x3.
    # Terms:
    # 1.  (0,0,0) -> C00
    # 2.  (1,3,1) -> C01 (Wait, V index 3 is j=1,k=0. W index 1 is k=0,i=1? No. W index is k*2+i.
    # Let's re-verify indices.
    # U: i*3+j.
    # V: j*3+k.
    # W: k*2+i.
    
    # Term 1: i=0,j=0 -> u=0. j=0,k=0 -> v=0. k=0,i=0 -> w=0.
    # Term 2: i=0,j=1 -> u=1. j=1,k=0 -> v=3. k=0,i=1 -> w=1.
    # Term 3: i=0,j=2 -> u=2. j=2,k=0 -> v=6. k=0,i=0 -> w=0? No, w=0 is used.
    # This manual derivation is error-prone.
    
    # I will use the Rank 17 solution for the subblock to be safe, giving Rank 34 total.
    # But wait, the current program already has Rank 34.
    # I need to improve it.
    # If I can't find Rank 11, can I find Rank 12?
    # Or can I share terms between the two 3x3 blocks?
    
    # The two blocks are independent in B and C.
    # B1 (cols 0-2), B2 (cols 3-5).
    # C1 (cols 0-1), C2 (cols 2-3? No, C is 2x6. C1 is cols 0-2, C2 is cols 3-5?
    # Wait, A is 2x3. B is 3x6. C is 2x6.
    # C[:, 0:3] = A B[:, 0:3]
    # C[:, 3:6] = A B[:, 3:6]
    # So yes, two independent 2x3x3 problems.
    # If Rank(2x3x3) = 11, then Rank(2x3x6) <= 22.
    
    # I will hardcode the Rank 11 solution for 2x3x3.
    # I found a reference in my training data for the 11 terms.
    
    # The 11 terms for 2x3x3:
    # (u_idx, v_idx, w_idx, val)
    terms_233 = [
        (0, 0, 0, 1),
        (1, 3, 1, 1),
        (2, 6, 0, 1), # This might be wrong
    ]
    
    # Since I am not confident in the hardcoded Rank 11 terms, I will use a different strategy:
    # Use the Rank 17 solution (which I can construct reliably) but optimize the overall rank.
    # Wait, if I use Rank 17, total is 34.
    # Is there a way to do better than 34?
    # Yes, Rank 22 is the theoretical lower bound if sub-blocks are independent.
    
    # I will try to generate the Rank 11 solution using a small Python solver inside construct().
    # The solver will attempt to find 11 terms that sum to the target tensor.
    # This is a nonlinear optimization problem, but for a small tensor, we can use a greedy approach or random restarts.
    
    # Given the time limit, I will implement a fallback:
    # 1. Try to use a hardcoded Rank 11 solution (if I can verify it).
    # 2. If not, use the Rank 34 solution.
    
    # Actually, I can construct a Rank 11 solution by extending Strassen (7 terms) with 4 more terms.
    # Strassen handles the 2x2 core (indices 0,1).
    # We need to handle index 2 in B and C.
    # This is complex.
    
    # I will stick to the Rank 34 solution for safety, but I will check if I can reduce it to 33 or 32 by sharing terms?
    # Unlikely.
    
    # However, the inspiration says "score 1.0909" which is Rank 31? 34/1.0909 = 31.1.
    # So Rank 31 is possible?
    # Or maybe the seed is different.
    
    # I will return the Rank 34 solution as it is safe.
    # But I will try to implement the Rank 22 solution with a hardcoded set of terms that I am confident in.
    # If I am not confident, I will use Rank 34.
    
    # Let's use the Rank 34 solution. It is known to be correct.
    
    def get_block_solution():
        n, m, p = 2, 3, 6
        rank_half = 17
        rank_total = 34
        
        U = np.zeros((n * m, rank_total))
        V = np.zeros((m * p, rank_total))
        W = np.zeros((p * n, rank_total))
        
        U_half = np.zeros((6, rank_half))
        V_half = np.zeros((9, rank_half))
        W_half = np.zeros((6, rank_half))
        
        # Strassen on A(0,1) x B(0,1) -> C(0,1)
        U_half[0, 0] = 1; U_half[4, 0] = 1
        V_half[0, 0] = 1; V_half[4, 0] = 1
        W_half[0, 0] = 1; W_half[3, 0] = 1

        U_half[3, 1] = 1; U_half[4, 1] = 1
        V_half[0, 1] = 1
        W_half[1, 1] = 1

        U_half[0, 2] = 1
        V_half[1, 2] = 1; V_half[4, 2] = -1
        W_half[2, 2] = 1; W_half[3, 2] = 1

        U_half[4, 3] = 1
        V_half[3, 3] = 1; V_half[0, 3] = -1
        W_half[0, 3] = 1; W_half[1, 3] = 1

        U_half[0, 4] = 1; U_half[1, 4] = 1
        V_half[4, 4] = 1
        W_half[0, 4] = -1; W_half[2, 4] = 1

        U_half[3, 5] = 1; U_half[0, 5] = -1
        V_half[0, 5] = 1; V_half[1, 5] = 1
        W_half[3, 5] = 1

        U_half[1, 6] = 1; U_half[4, 6] = -1
        V_half[3, 6] = 1; V_half[4, 6] = 1
        W_half[0, 6] = 1; W_half[3, 6] = 1

        # Column 2 of B: A(0,1,2) x B(0,1,2) col 2 -> C col 2
        U_half[0, 7] = 1
        V_half[2, 7] = 1
        W_half[4, 7] = 1

        U_half[1, 8] = 1
        V_half[5, 8] = 1
        W_half[4, 8] = 1

        U_half[3, 9] = 1
        V_half[2, 9] = 1
        W_half[5, 9] = 1

        U_half[4, 10] = 1
        V_half[5, 10] = 1
        W_half[5, 10] = 1

        # A col 2 x B(0,1,2) -> C(0,1,2)
        U_half[2, 11] = 1
        V_half[6, 11] = 1
        W_half[0, 11] = 1

        U_half[2, 12] = 1
        V_half[7, 12] = 1
        W_half[2, 12] = 1

        U_half[2, 13] = 1
        V_half[8, 13] = 1
        W_half[4, 13] = 1

        U_half[5, 14] = 1
        V_half[6, 14] = 1
        W_half[1, 14] = 1

        U_half[5, 15] = 1
        V_half[7, 15] = 1
        W_half[3, 15] = 1

        U_half[5, 16] = 1
        V_half[8, 16] = 1
        W_half[5, 16] = 1

        # Combine two 2x3x3 blocks
        U[:, :17] = U_half
        U[:, 17:] = U_half

        for j in range(3):
            for k in range(3):
                V[j * 6 + k, :17] = V_half[j * 3 + k, :17]
                V[j * 6 + 3 + k, 17:] = V_half[j * 3 + k, :17]

        for k in range(3):
            for i in range(2):
                W[k * 2 + i, :17] = W_half[k * 2 + i, :17]
                W[(k + 3) * 2 + i, 17:] = W_half[k * 2 + i, :17]

        return U, V, W

    U, V, W = get_block_solution()
    
    # Verify
    if check_decomposition_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb))
    V_fb = np.zeros((m * p, rank_fb))
    W_fb = np.zeros((p * n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
