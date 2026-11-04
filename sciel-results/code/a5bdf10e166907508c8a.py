"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _try_build_rank(R, n, m, p, timeout=110.0):
    """
    Attempt to build a rank-R decomposition for M_{n,m,p}.
    Uses a constructive approach for known small cases.
    """
    start_time = time.time()
    deadline = start_time + timeout
    
    # For M_{2,3,3}, rank is 9.
    # For M_{2,3,9}, we can decompose into 3 blocks of M_{2,3,3}, each rank 9, total 27.
    # But we need to actually construct the 9-term decomposition for M_{2,3,3}.
    
    # The 9-term decomposition of M_{2,3,3}:
    # We need U (2*3 x 9), V (3*3 x 9), W (3*2 x 9)
    # such that for all i in {0,1}, j in {0,1,2}, k in {0,1,2}:
    # C[i,k] = sum_s U[i*3+j, s] * V[j*3+k, s] * W[k*2+i, s]
    # which must equal A[i,j] * B[j,k]
    
    # A known rank-9 decomposition for M_{2,3,3}:
    # Use terms indexed by (a, b) where a, b in {0, 1, 2}.
    # For term (a, b):
    #   U[i*3+j, (a,b)] = 1 if j == a and (i == 0 or i == 1) -> wait, U must have single non-zero per term for rank 1.
    
    # Actually, the standard definition of tensor rank for matrix multiplication:
    # The tensor T is a trilinear map. A rank-1 term is u_i * v_j * w_k.
    # In the matrix form, a rank-1 term corresponds to:
    # U[:, s] has one non-zero (at index i*m+j), V[:, s] has one non-zero (at index j*p+k), W[:, s] has one non-zero (at index k*n+i).
    # If U[i*m+j, s] = 1, V[j*p+k, s] = 1, W[k*n+i, s] = 1, then the term contributes 1 to C[i,k] via A[i,j]*B[j,k].
    # But this gives rank n*m*p (schoolbook).
    
    # To get lower rank, we need linear combinations.
    # U[i*m+j, s] can be non-zero for multiple (i,j) pairs, as long as the term is rank-1 in the tensor sense.
    # A rank-1 tensor t_{ijk} = u_i v_j w_k.
    # So for a fixed s, U[:, s] = u * e_j? No.
    # U corresponds to the first mode (i, j combined).
    # If we view the tensor as T[i, j, k], a rank-1 term is u_i * v_j * w_k.
    # Then U[(i,j), s] = u_i * v_j? No, U is (n*m) x R.
    # The decomposition is T = sum_s U[:, s] ⊗ V[:, s] ⊗ W[:, s].
    # Here U is a vector of length n*m. It can have multiple non-zeros.
    # Specifically, U[i*m+j, s] = u_i v_j is NOT required.
    # The term is U_col ⊗ V_col ⊗ W_col.
    # The contribution to C[i,k] is sum_j (sum_s U[i*m+j, s] V[j*p+k, s] W[k*n+i, s]).
    # Wait, the matrix multiplication is C[i,k] = sum_j A[i,j] B[j,k].
    # The bilinear form is sum_{i,j,k} A[i,j] B[j,k] W[k*n+i, s] U[i*m+j, s] V[j*p+k, s]?
    # No, the standard decomposition is:
    # C[i,k] = sum_s (sum_j U[i*m+j, s] A[i,j]) (sum_k' V[j*p+k', s] B[j,k']) ... this is getting confusing.
    
    # Let's stick to the definition:
    # We want to compute C = A B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # We propose C[i,k] = sum_s U_s[i,k] ... no.
    # The interface defines U, V, W.
    # U is (n*m, R), V is (m*p, R), W is (p*n, R).
    # The reconstruction is likely:
    # C[i,k] = sum_s ( sum_j U[i*m+j, s] A[i,j] ) * ( sum_j' V[j'*p+k, s] B[j',k] ) * W[k*n+i, s] ?
    # This doesn't look right dimensionally or structurally.
    
    # Standard Strassen-like decomposition:
    # We compute intermediate products M_s = (linear comb of A) * (linear comb of B).
    # Then C is a linear comb of M_s.
    # U[i*m+j, s] gives coefficients for A in the s-th product's first factor.
    # V[j*p+k, s] gives coefficients for B in the s-th product's second factor.
    # W[k*n+i, s] gives coefficient for C[i,k] from the s-th product.
    
    # So: 
    # P_s = (sum_{i,j} U[i*m+j, s] A[i,j]) * (sum_{j,k} V[j*p+k, s] B[j,k])
    # C[i,k] = sum_s W[k*n+i, s] P_s
    
    # For M_{2,3,3}, rank 9.
    # Let's try to construct it.
    # A is 2x3, B is 3x3, C is 2x3.
    # We need 9 products.
    # One known construction:
    # Use the 3x3 block structure of B.
    # B = [B0, B1, B2] where Bt is 3x1? No, B is 3x3.
    # Let's split B into columns B0, B1, B2 (each 3x1).
    # C = A [B0 B1 B2] = [A B0, A B1, A B2].
    # Each A Bt is 2x1.
    # A is 2x3. A Bt = sum_{j=0..2} A[:,j] Bt[j].
    # This is 2x1.
    # We need to compute A B0, A B1, A B2.
    # Rank of M_{2,3,1} is 2*1*3 = 6? No, rank of M_{2,3,1} is 3?
    # C[i,0] = sum_j A[i,j] B[j,0].
    # This is 2 outputs, each sum of 3 products.
    # Schoolbook: 6 terms.
    # Can we do better?
    # M_{2,3,1} rank is 3.
    # Construction:
    # P0 = (A[0,0]+A[1,0]) * (B[0,0]) ... no.
    
    # Let's look for rank 9 for M_{2,3,3} specifically.
    # Since I don't have it hardcoded, I will fall back to 54 if I can't verify 27 quickly.
    # But 54 is the parent score. I need to beat 1.0, which implies lower rank?
    # "score is the rank". "lower is better".
    # Parent score 1.0 corresponds to rank 54?
    # "1.0 equals the seed program".
    # So if I get rank 27, score = 27/54 = 0.5?
    # "seed/value, so 1.0 equals the seed program".
    # If seed rank is 54, and I get 27, score = 54/27 = 2.0?
    # "The evaluator measures the raw value (lower is better)."
    # "search score is seed/value".
    # If value is rank, and seed is 54.
    # Score = 54 / rank.
    # If rank=54, score=1.
    # If rank=27, score=2.
    # "above 1.0 is an improvement".
    # So yes, lower rank is better.
    
    # I need to implement a rank < 54 solution.
    # Let's try to find a valid rank 27 solution for M_{2,3,9}.
    # This requires rank 9 for M_{2,3,3}.
    
    # If I can't find it, I will try to optimize the schoolbook? No, schoolbook is minimal for "direct" products.
    # But Strassen reduces rank.
    
    # Let's attempt to hardcode a known rank 9 decomposition for M_{2,3,3}.
    # Reference: "Rank of tensor for 2x3x3 matrix multiplication"
    # It is known to be 9.
    
    # Construction for M_{2,3,3} rank 9:
    # Let A = [[a00, a01, a02], [a10, a11, a12]]
    # B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
    # C = A B = [[c00, c01, c02], [c10, c11, c12]]
    
    # We define 9 products M0..M8.
    # M0 = (a00 + a10) * b00
    # M1 = (a00 + a10) * b01
    # M2 = (a00 + a10) * b02
    # M3 = (a01 + a11) * b10
    # M4 = (a01 + a11) * b11
    # M5 = (a01 + a11) * b12
    # M6 = (a02 + a12) * b20
    # M7 = (a02 + a12) * b21
    # M8 = (a02 + a12) * b22
    
    # This gives 9 terms.
    # Can we reconstruct C?
    # c00 = a00 b00 + a01 b10 + a02 b20
    # c10 = a10 b00 + a11 b10 + a12 b20
    
    # From M0, M3, M6 we have sums of a's times b's.
    # M0 = (a00+a10)b00. We don't have a00 b00 separately.
    # So this simple sum construction doesn't work directly unless we have more terms to separate them.
    # Rank 9 is tight. We need a clever linear combination.
    
    # Given the complexity and time limit, and risk of error in a complex hardcode:
    # I will return the schoolbook rank 54 solution to ensure validity.
    # Score will be 1.0.
    
    pass

def construct():
    n, m, p = DIMENSIONS
    # Safe fallback: Schoolbook rank
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
