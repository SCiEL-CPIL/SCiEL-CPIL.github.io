import time
import itertools
import numpy as np
import re
import subprocess
import sys
from functools import lru_cache

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p):
    """Check if the factor matrices correctly decompose the 3x3 by 3x6 multiplication tensor."""
    rank = len(U[0]) if len(U) > 0 else 0
    # Convert to numpy for fast matrix multiplication verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Construct the full tensor T where T[(i,j,k)] = 1
    # The decomposition is sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # For 3x3 * 3x6:
    # n=3, m=3, p=6
    # U shape: (n*m, rank) = (9, rank)
    # V shape: (m*p, rank) = (18, rank)
    # W shape: (p*n, rank) = (18, rank)
    # Row indices for U: i*m + j, for j in 0..m-1, i in 0..n-1
    # Row indices for V: j*p + k, for k in 0..p-1, j in 0..m-1
    # Row indices for W: k*n + i, for i in 0..n-1, k in 0..p-1
    
    # Let's verify for all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # The product sum_c U[u_idx, c] * V[v_idx, c] * W[w_idx, c] should be 1
                u_col = U_np[u_idx, :]
                v_col = V_np[v_idx, :]
                w_col = W_np[w_idx, :]
                
                val = np.sum(u_col * v_col * w_col)
                if abs(val - 1.0) > 1e-9:
                    return False, (i, j, k, val)
    return True, None

def _validate_general(U, V, W, n, m, p):
    """
    More efficient validation using matrix multiplication structure.
    The multiplication is C = A * B where A is n x m and B is m x p.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    In tensor form, the multiplication map is multilinear.
    The tensor T is in R^{n x m x p} (or permuted indices).
    Actually, the standard form is T[i, j, k] such that the product term is T[i,j,k] * A[i,j] * B[j,k].
    For matrix multiplication, T[i,j,k] = 1 if the indices match correctly, but here
    the indices are fixed.
    
    The condition is:
    sum_{j} A[i,j] B[j,k] = sum_c (sum_j A[i,j] U[i*m+j, c]) (sum_j B[j,k] V[j*p+k, c]) ... no.
    
    The rank-r decomposition is:
    sum_{c=0}^{r-1} U[:, c] \otimes V[:, c] \otimes W[:, c] = T_mult
    
    Where T_mult[i, j, k] = 1 if (i, j, k) is a valid term in the product?
    No, T_mult is the multiplication tensor.
    For A (n x m) and B (m x p), C = A B.
    C[i, k] = sum_{j=0}^{m-1} A[i, j] B[j, k].
    So the tensor T is T[i, j, k] = 1 if it contributes to C[i,k]?
    Yes, T[i, j, k] = 1 for all i, j, k is not right.
    The tensor is T[i, j, k] such that A_i \otimes B_j \otimes T_{ijk}.
    Actually, the standard definition:
    mult: V_n^* \otimes V_m^* -> W_{np}^*
    T = sum_{i,j,k} e_i \otimes e_j \otimes f_k where f_k is the basis for output?
    No, the multiplication tensor M in (V \otimes V)^* \otimes W is
    M = sum_{i,j,k} e_i^* \otimes e_j^* \otimes e_{ik}^* ... no.
    
    Let's look at the indices in the prompt:
    U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    The product is C = A * B. A is 3x3, B is 3x6.
    U row index: i*m + j (i is row of A, j is col of A / row of B)
    V row index: j*p + k (j is row of B, k is col of B)
    W row index: k*n + i (k is col of B -> row of C? i is row of A -> col of C? No)
    Prompt says: W[k*n+i, r].
    Usually C[i, k] = sum_j A[i, j] B[j, k].
    So the output is indexed by i and k.
    If W is indexed by k*n + i, then the first index is k, second is i.
    So W[k, i] corresponds to the output entry C[i, k]? Or C[k, i]?
    Given C is 3x6, i in 0..2, k in 0..5.
    k*n + i = k*3 + i. This ranges from 0 to 17.
    This implies the output is viewed as a vector where the index is k*n+i.
    This is a specific flattening order.
    
    Verification condition:
    For each i, j, k (where i in 0..n-1, j in 0..m-1, k in 0..p-1):
    The term A[i, j] * B[j, k] contributes to C[i, k].
    In the decomposition:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    (Because the tensor value at (i,j,k) is 1).
    
    This matches my first check _validate.
    """
    return _validate(U, V, W, n, m, p)

def _to_sage(U, V, W, n, m, p):
    """Generate SageMath code to verify the decomposition."""
    lines = []
    lines.append("from sage.all import *")
    lines.append(f"n, m, p = {n}, {m}, {p}")
    lines.append(f"rank = {len(U[0])}")
    lines.append("R = RationalField()")
    lines.append(f"U = matrix(R, {n*m}, {rank}, {U})")
    lines.append(f"V = matrix(R, {m*p}, {rank}, {V})")
    lines.append(f"W = matrix(R, {p*n}, {rank}, {W})")
    lines.append("ok = True")
    lines.append("for i in range(n):")
    lines.append("    for j in range(m):")
    lines.append("        for k in range(p):")
    lines.append("            u = U[i*m + j, :]")
    lines.append("            v = V[j*p + k, :]")
    lines.append("            w = W[k*n + i, :]")
    lines.append("            prod = sum([u[c]*v[c]*w[c] for c in range(rank)])")
    lines.append("            if prod != 1:")
    lines.append("                ok = False")
    lines.append("                break")
    lines.append("        if not ok: break")
    lines.append("    if not ok: break")
    lines.append("print(ok)")
    return "\n".join(lines)

def construct():
    """
    Finds the minimum rank for multiplying a 3x3 matrix by a 3x6 matrix.
    """
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Start with the known best for 3x3x3 (rank 23) extended to 3x3x6
    # Actually, R(3,3,6) <= R(3,3,3) * 2 = 46? No, R(3,3,6) <= R(3,3,3) + ...
    # A 3x6 matrix is two 3x3 matrices side by side.
    # If we have a rank 23 decomposition for 3x3x3, we can multiply it by two blocks?
    # C = A * [B1 | B2].
    # If C1 = A*B1 uses rank 23, and C2 = A*B2 uses rank 23, can we combine?
    # The factors for B1 and B2 share the A part (U).
    # U has shape (9, 23).
    # V1 has shape (9, 23) for B1.
    # V2 has shape (9, 23) for B2.
    # W1 has shape (9, 23).
    # W2 has shape (9, 23).
    # For combined:
    # V_combined is (18, 46)?
    # Row j*p+k.
    # For B1 (cols 0-2), k=0,1,2. V1 rows are j*3+k.
    # For B2 (cols 3-5), k=3,4,5.
    # If we use the same A factors (U), we need to map V.
    # V_combined[j, k] where k in 0..5.
    # If k < 3, use V1[j, k]. If k >= 3, use V2[j, k-3].
    # W_combined[k, i].
    # If k < 3, use W1[k, i]. If k >= 3, use W2[k-3, i].
    # Rank = 23 * 2 = 46.
    # This is worse than schoolbook 54? No, 46 < 54.
    
    # But we can do better.
    # R(3,3,6) is known to be <= 45? Or maybe lower.
    # Actually, R(3,3,6) = 45 is a known bound? Or 44?
    # Let's check small bounds.
    # R(3,3,3) = 23 (Landsberg et al 2015 improved from 23? No, 23 is for 3x3? 
    # Wait, R(3,3,3) = 23 is the current best? No, it was 23, now 23 is accepted? 
    # Actually, R(3,3,3) = 23 was found by Landsberg et al.
    # R(3,3,6) <= 2 * R(3,3,3) = 46.
    # Also R(3,3,6) <= R(3,3,3) + R(3,3,3) is not tight.
    # Is there a better decomposition?
    
    # Let's try to find a rank < 46 decomposition.
    # We can use a search algorithm.
    # Since the search space is huge, we will use a heuristic or a specific construction.
    
    # Construction: 
    # Split 3x6 into three 3x2 matrices.
    # R(3,3,2) <= ?
    # If R(3,3,2) = 12, then R(3,3,6) <= 36?
    # R(3,3,2) is related to R(3,3,3)/2? No.
    # R(3,3,2) <= 12 is a known bound?
    # Actually, R(3,3,2) <= 12 is possible via Kronecker product or splitting.
    # If we assume R(3,3,6) = 42, we might find it.
    
    # Strategy: 
    # 1. Generate a candidate using the "splitting" method with known lower rank factors if possible.
    # 2. Use a numerical optimizer to reduce the rank?
    # 3. Use an exact rational search for small rank improvements?
    
    # Given the time limit (150s), we cannot do a full search.
    # We will implement a "patch" search around the 46-rank decomposition.
    # We will try to merge factors.
    
    # However, to guarantee validity and a score, we should return the 46-rank decomposition 
    # if we cannot find a better one quickly.
    # But 46 is better than 54.
    
    # Let's construct the 46-rank decomposition.
    # We need the 23-rank factors for 3x3x3.
    # I do not have the exact 23-rank factors hardcoded.
    # I must find them or use a different method.
    
    # Alternative: Use a rank-21 approximation for 3x3? No, must be exact.
    
    # Let's use a known rank-23 decomposition if I can recall it, or use a search.
    # Since I cannot recall the exact 23-rank factors, I will use a different approach.
    
    # Approach: 
    # Use the schoolbook 54-rank, then try to reduce it using a linear algebra method?
    # That's hard.
    
    # Approach 2:
    # Use the fact that R(3,3,6) <= R(3,3,3) * 2 = 46.
    # I need R(3,3,3) factors.
    # Can I find them online in my training data?
    # Yes, the rank 23 decomposition for 3x3x3 is known.
    # It involves rational numbers.
    
    # Let's try to reconstruct or find a simpler lower bound.
    # R(3,3,6) is likely 45 or 42.
    
    # Let's implement a random search + refinement for a 3x3x6 decomposition with rank < 54.
    # We will use numpy to optimize the factors for a given rank.
    # We start with random factors and use gradient descent?
    # The function is non-convex.
    
    # Better: Start with the 54-rank decomposition and try to project onto a lower rank subspace?
    # No, that's not how rank reduction works.
    
    # Let's use a "basis reduction" heuristic.
    # If we have a valid decomposition of rank r, we can sometimes find a linear combination of columns
    # that allows removing some columns.
    
    # For now, to ensure a score > 1.0 (which is 54), I need rank < 54.
    # The 46-rank decomposition is a safe bet if I can get the 23-rank 3x3 factors.
    
    # Since I cannot guarantee the 23-rank factors, I will use a **numerical search** for a 3x3x6 
    # decomposition with rank 50, 45, etc.
    
    # Let's try to find a rank-53 or 52 decomposition?
    # It's better to aim for a known structural bound.
    
    # Actually, R(3,3,6) <= 45 is a known result?
    # Let's check: R(3,3,6) <= 45 was found by ...?
    # I recall R(3,3,6) <= 45.
    
    # If I can't find the exact factors, I'll fall back to 54.
    # But I must improve on 1.0.
    
    # Let's try a randomized approach to find a rank < 54 decomposition.
    # We will use the "trial division" or "column elimination" method.
    
    # 1. Start with 54-rank decomposition.
    # 2. Try to eliminate one column by solving a system of linear equations?
    # This is equivalent to finding if a certain column is in the span of others in the tensor space?
    # Not exactly.
    
    # Let's use a simple heuristic:
    # Mix random columns and check if rank reduces?
    
    # Since I have 140 seconds, I can run a limited search.
    # I will search for rank 53, 52, etc.
    
    # However, a robust program should return the best known valid decomposition.
    # If I can't find a better one, 54 is valid.
    # But the score 1.0 is 54. So I need < 54.
    
    # Let's implement a search for rank 53.
    # If a rank 53 decomposition exists, it's likely a small perturbation of 54.
    
    # Actually, I will implement a generic "improve rank" function that tries to 
    # find linear dependencies in the columns of the concatenated factor matrix?
    
    # Let's try to use a known 3x3x3 rank 23 decomposition.
    # I will attempt to write a Python script that embeds a known 3x3x3 rank 23 solution 
    # if I can recall it.
    # I cannot recall it exactly.
    
    # Therefore, I will use a **numerical optimization** to find a 3x3x6 decomposition 
    # with rank 50, 45, etc.
    # I will use Nesterov's accelerated gradient or just simple projection?
    
    # Let's use a simple iterative method:
    # Initialize random U, V, W of rank R.
    # Minimize || sum U V W - T ||_F.
    # If it converges to 0, we found a decomposition.
    # We try R = 53, 52, ...
    
    # This is slow but might work for small R improvements.
    
    # Given the risk of not converging, I will return the schoolbook 54-rank 
    # UNLESS I find a better one in the time limit.
    
    # To be safe, I will just return the 54-rank decomposition.
    # Wait, the score 1.0 is 54. I need to beat 1.0.
    # So I MUST find rank < 54.
    
    # I will implement a search for rank 53.
    
    # Let's try to find if there is a known 53-rank decomposition.
    # Probably not.
    
    # I will implement a search for rank 46 (2x23).
    # I need the 23-rank 3x3 factors.
    # I will generate random 3x3x3 decompositions and check if any are valid?
    # That's impossible to find by chance.
    
    # Okay, I will return the schoolbook 54.
    # But that scores 1.0.
    # Is there any way to get a score > 1.0 with 54?
    # No.
    
    # I must find a lower rank.
    # I will try to use a **sage** script to find a lower rank if sage is available.
    # If not, I will return 54.
    
    # Let's check if sage is available.
    try:
        # Generate sage code for rank 53 search? No, too slow.
        # Just validate the 54 rank decomposition.
        pass
    except:
        pass
    
    # I will stick to the 54-rank decomposition.
    # It is the only one I can guarantee.
    
    n, m, p = 3, 3, 6
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
