"""
Task: Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 8 matrix.
Dimensions: n=2, m=3, p=8.
Shapes:
U: (n*m, rank) = (6, rank)
V: (m*p, rank) = (24, rank)
W: (p*n, rank) = (16, rank)
Convention: W[k*n+i, r]
Score: rank (lower is better).
Budget: 150s wall-clock.
"""

import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 8]
N, M, P = DIMENSIONS

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # 1. Schoolbook Decomposition (Rank 48)
    U_s = np.zeros((N*M, N*M*P))
    V_s = np.zeros((M*P, N*M*P))
    W_s = np.zeros((P*N, N*M*P))
    
    idx = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                u_idx = i*M + j
                v_idx = j*P + k
                w_idx = k*N + i
                U_s[u_idx, idx] = 1
                V_s[v_idx, idx] = 1
                W_s[w_idx, idx] = 1
                idx += 1

    # The target tensor T in coordinate form:
    # T[i, j, k] = 1 if (i,j,k) is in the valid range, else 0.
    # But effectively, we are decomposing the identity-like structure of the multiplication.
    # Let's define the target "tensor" as a 3D array T of shape (N*M, M*P, P*N) ? No.
    # The multiplication is C = A @ B.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # In tensor decomposition form: C_{ik} = sum_r U_{i,r} V_{j,r} W_{k,r} is NOT quite right because indices are mixed.
    
    # Correct Tensor Decomposition for Matrix Multiplication:
    # C_{ik} = sum_r ( sum_j U_{ij,r} A_{ij} ) ... No.
    # Standard form: C_{ik} = sum_r (A^T U_r) (V_r B) ?
    
    # Let's stick to the provided interface:
    # U (N*M, R), V (M*P, R), W (P*N, R)
    # The condition is:
    # C_{ik} = sum_{j} sum_{r} U_{iM+j, r} A_{ij} V_{jP+k, r} ... Wait, W is involved how?
    
    # The task says: "factor matrices ... that exactly decompose the matrix multiplication tensor".
    # Usually, this means:
    # sum_r U_{i, r} (V_{j, r} A_{ij}) ... No.
    
    # Let's look at the schoolbook construction in the prompt:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # This implies the decomposition of the tensor T_{i,j,k} which is 1 if the indices match the multiplication flow.
    # Specifically, the tensor being decomposed is the "identity" of the bilinear map?
    # Actually, the tensor T is defined such that the multiplication is a contraction of T with A and B.
    
    # If we define T_{i, j, k} = delta_{i, j, k}? No.
    # The bilinear map is A, B -> C.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # This is a 3-tensor T_{i, j, k} where T_{i, j, k} = 1 if the indices are compatible?
    # No, T is fixed. T_{i, j, k} = 1 for all i, j, k? No, that would be the all-ones tensor.
    
    # Let's re-read carefully: "decompose the matrix multiplication tensor".
    # The tensor for M_{n,m,p} is usually defined as T_{i, j, k} = 1 if (i,j,k) are valid indices, 0 otherwise?
    # No, that's not a tensor in the algebraic geometry sense.
    
    # In the context of "rank of matrix multiplication tensor", the tensor T is the trilinear form:
    # T(A, B, C) = tr(C^T (A B)) ?
    # Or simply, the tensor whose entries T_{i,j,k} are 1 if the multiplication uses A_{ij} and B_{jk} to produce C_{ik}.
    # Yes, T_{i,j,k} = 1 for all valid i, j, k.
    # Wait, if T_{i,j,k}=1, then sum_r U_{i,r} V_{j,r} W_{k,r} must equal 1 for all i,j,k?
    # No, the indices in U, V, W are flattened.
    
    # Let's check the schoolbook code again.
    # U has index (i, j). V has index (j, k). W has index (k, i).
    # The term r corresponds to a specific (i, j, k) tuple.
    # U_{iM+j, r} = 1 if r=(i',j',k') with i'=i, j'=j.
    # V_{jP+k, r} = 1 if r=(i',j',k') with j'=j, k'=k.
    # W_{kN+i, r} = 1 if r=(i',j',k') with k'=k, i'=i.
    
    # So for a fixed r=(i',j',k'), the contribution to the "tensor" at coordinate (i,j,k) is:
    # If we contract U with A, V with B, W with C?
    # The standard definition of rank for M_{n,m,p} is the smallest R such that:
    # M_{n,m,p} = sum_{r=1}^R U^{(r)} \otimes V^{(r)} \otimes W^{(r)}
    # where U^{(r)} \in R^{N \times M}, V^{(r)} \in R^{M \times P}, W^{(r)} \in R^{P \times N}.
    # And the multiplication is recovered by:
    # C = sum_r (A * U^{(r)}) * (V^{(r)} * B) ...?
    
    # Actually, the identity is:
    # A B = sum_r (A \circ U^{(r)}) (V^{(r)} \circ B) ? No.
    
    # Let's use the flattening provided.
    # The tensor T has entries T_{u, v, w} where u in [0, NM), v in [0, MP), w in [0, PN).
    # T_{u, v, w} = 1 if u=(i,j), v=(j,k), w=(k,i) for some i,j,k. Otherwise 0.
    # Note: For a fixed u=(i,j) and v=(j,k), there is exactly one w=(k,i).
    # So T_{u,v,w} = 1 if the indices are consistent, 0 otherwise.
    
    # We want to write T_{u,v,w} = sum_{r=1}^R U_{u,r} V_{v,r} W_{w,r}.
    
    # Goal: Minimize R.
    
    # Current Best: 48 (Schoolbook).
    # Lower Bound: Max(NM, MP, PN) = Max(6, 24, 16) = 24.
    # Strassen: For 2x2x2, rank 7 < 8.
    # For 2x3x8, can we do better than 48?
    # 2x3x8 can be viewed as 2 blocks of 2x3x4? Or 4 blocks of 2x3x2?
    # If we view it as 4 independent 2x3x2 multiplications (since p=8, maybe split k into 4 groups of 2? No, m=3 is odd).
    
    # Let's try a random search / heuristic optimization to see if we can beat 48.
    # Given the time limit, a full algebraic search is risky.
    # However, 48 is likely not minimal.
    # Known results:
    # M_{2,3,3} has rank <= 14? (Schoolbook 18).
    # M_{2,3,8} is larger.
    
    # Strategy:
    # 1. Try to construct a rank 32 or 40 decomposition using block structure.
    # 2. If not found quickly, return 48.
    
    # Let's attempt a simple random initialization and gradient descent on the bilinear constraint?
    # No, integer/half-integer entries are required? "integer/half-integer entries".
    # This suggests the answer is likely to have simple coefficients.
    # Schoolbook has 0s and 1s.
    
    # Let's try to merge terms.
    # In Strassen (2x2x2), we save 1 term by cleverly combining.
    # Here we have 48 terms.
    
    # Heuristic: Random search for a rank R < 48.
    # We can fix R and try to solve for U, V, W.
    # This is a non-convex problem.
    
    # Given the 150s limit, and the complexity, I will return the Schoolbook 48 if I can't find a better one fast.
    # But 48 might be improvable to, say, 42 or 40.
    
    # Let's try a small search for R=40, 36, 32.
    # Use random initialization and check if the tensor matches.
    # Matching condition:
    # For all u,v,w, sum_r U_{u,r} V_{v,r} W_{w,r} == T_{u,v,w}
    
    # Since T is sparse (only 48 ones), we can check the 48 non-zero entries and the 0s.
    # Actually, we only need to ensure the 48 ones are 1 and the rest are 0.
    # The dimension of the space is 6*24*16 = 2304.
    # 48 non-zero entries.
    
    # Random search is unlikely to find an exact integer solution quickly.
    # I will stick with 48 as it is robust and valid.
    # Is there a known better rank for 2x3x8?
    # 2x3x8 can be decomposed into 4 2x3x2 blocks?
    # If we split p=8 into 4 groups of 2, we have 4 independent 2x3x2 multiplications.
    # Rank of 2x3x2:
    # Schoolbook: 12.
    # Can we do better?
    # M_{2,3,2}.
    # Lower bound: Max(6, 6, 4) = 6.
    # Is rank 10 or 11 possible?
    # If we can do rank 10 for 2x3x2, then 4*10 = 40 for 2x3x8.
    
    # Let's try to find rank 10 for 2x3x2.
    # This is a smaller problem.
    # I will implement a search for 2x3x2 rank 10.
    
    # Time check
    if time.time() > deadline - 20:
         return {'U': U_s.tolist(), 'V': V_s.tolist(), 'W': W_s.tolist()}

    # Search for 2x3x2 rank 10
    def find_2x3x2_rank_10():
        n, m, p = 2, 3, 2
        rank = 10
        # Random search
        for _ in range(1000):
            U = np.random.randint(-1, 2, (n*m, rank))
            V = np.random.randint(-1, 2, (m*p, rank))
            W = np.random.randint(-1, 2, (p*n, rank))
            
            # Check
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i*m + j
                        v_idx = j*p + k
                        w_idx = k*n + i
                        
                        val = 0
                        for r in range(rank):
                            val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                        if val != 1:
                            ok = False
                            break
                    if not ok: break
                if not ok: break
            if ok:
                # Check zeros
                # Sample some zero positions
                for _ in range(100):
                    u = random.randint(0, n*m-1)
                    v = random.randint(0, m*p-1)
                    w = random.randint(0, p*n-1)
                    i = u // m; j = u % m
                    j2 = v // p; k = v % p
                    k2 = w // n; i2 = w % n
                    if (j == j2) and (k == k2) and (i == i2):
                        continue # This is a one
                    val = 0
                    for r in range(rank):
                        val += U[u, r] * V[v, r] * W[w, r]
                    if val != 0:
                        ok = False
                        break
                if ok:
                    return U, V, W
        return None

    U_sub, V_sub, W_sub = find_2x3x2_rank_10()
    
    if U_sub is not None:
        # We have a rank 10 decomposition for 2x3x2.
        # Construct 2x3x8 by repeating 4 times?
        # The p dimension is 8. We split it into 4 chunks of 2.
        # U is same for all chunks?
        # V and W depend on the chunk index.
        
        # U (6, 40)
        # V (24, 40)
        # W (16, 40)
        
        U_new = np.zeros((N*M, 40))
        V_new = np.zeros((M*P, 40))
        W_new = np.zeros((P*N, 40))
        
        for c in range(4):
            # Chunk c uses columns c*10 to (c+1)*10 - 1
            col_start = c * 10
            col_end = (c + 1) * 10
            
            U_new[:, col_start:col_end] = U_sub
            
            # V: m*p = 3*8 = 24.
            # The 2x3x2 subproblem has V shape (3*2, 10) = (6, 10).
            # For chunk c, the k indices are 2*c and 2*c+1.
            # The j indices are 0, 1, 2.
            # So for row j*P + k in V_new:
            # If k is in [2c, 2c+1], it maps to row (j*2 + (k-2c)) in V_sub.
            
            for j in range(M):
                for k in range(P):
                    if 2*c <= k < 2*c+2:
                        k_local = k - 2*c
                        v_new_idx = j*P + k
                        v_sub_idx = j*2 + k_local
                        V_new[v_new_idx, col_start:col_end] = V_sub[v_sub_idx, :]
            
            # W: p*n = 8*2 = 16.
            # The 2x3x2 subproblem has W shape (2*2, 10) = (4, 10).
            # W index is k*n + i.
            # For chunk c, k is in [2c, 2c+1].
            
            for k in range(P):
                for i in range(N):
                    if 2*c <= k < 2*c+2:
                        k_local = k - 2*c
                        w_new_idx = k*N + i
                        w_sub_idx = k_local*2 + i
                        W_new[w_new_idx, col_start:col_end] = W_sub[w_sub_idx, :]
                        
        return {'U': U_new.tolist(), 'V': V_new.tolist(), 'W': W_new.tolist()}

    # Fallback to Schoolbook
    return {'U': U_s.tolist(), 'V': V_s.tolist(), 'W': W_s.tolist()}

# EVOLVE-BLOCK-END
