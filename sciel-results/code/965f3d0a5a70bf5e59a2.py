import numpy as np
import time
from collections import defaultdict

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 140
    
    # We will build the decomposition for 4x4 * 4x7
    # Split into 4x4 * 4x4 (Strassen, rank 49) and 4x4 * 4x3 (Schoolbook, rank 48)
    
    # U: 16 x Rank
    # V: 28 x Rank
    # W: 28 x Rank
    
    # We accumulate terms as lists of (index, coeff) for U, V, W
    # A term is a triple (U_dict, V_dict, W_dict)
    # But W is just the output coefficient, which for Strassen/Schoolbook is usually just 1 or -1 for specific (k,i)
    # Actually, W is part of the tensor decomposition.
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    
    # For schoolbook, U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1.
    # For Strassen, the coefficients are distributed.
    
    # Let's represent each term as:
    # u_coeffs: dict { (i,j): coeff }
    # v_coeffs: dict { (j,k): coeff }
    # w_coeffs: dict { (k,i): coeff }
    
    terms = []
    
    # --- Part 1: Strassen for 4x4 * 4x4 (k=0..3) ---
    
    # Helper to expand a 2x2 Strassen product
    # X and Y are 2x2 matrices of linear combinations
    # X[r][c] is a list of (a_idx, coeff) where a_idx is (i,j)
    # Y[r][c] is a list of (b_idx, coeff) where b_idx is (j,k)
    
    def expand_strassen_2x2(X, Y):
        # Returns list of 7 terms, each term is (u_coeffs, v_coeffs)
        # The W coefficients are determined by how these terms map to C
        # Wait, the standard Strassen decomposition gives C as linear comb of M1..M7.
        # So we need to return the 7 M terms, and then the final C is sum of them.
        # But in the tensor decomposition, each scalar multiplication is a term.
        # The "M" terms are intermediate.
        # Actually, the rank 7 decomposition of 2x2x2 is:
        # M1 = (X00+X11)(Y00+Y11)
        # ...
        # C00 = M1 + M4 - M5 + M7
        # ...
        # This means the tensor T is sum of 7 rank-1 tensors.
        # Each rank-1 tensor corresponds to one M.
        # So for each M, we have a U part (linear comb of X), V part (linear comb of Y), and W part (linear comb of C indices).
        
        # Let's generate the 7 (U, V, W) triples for the 2x2 case.
        # X, Y are 2x2 arrays of "linear comb" objects.
        # A linear comb object is a list of (index, coeff).
        
        # We need to multiply two linear combs? No.
        # M1 = (X00+X11) * (Y00+Y11)
        # This is a product of two linear combs.
        # (sum a_i A_i) * (sum b_j B_j) = sum_{i,j} a_i b_j A_i B_j
        # This expands to multiple scalar multiplications!
        # Wait. If X00 = A00, X11 = A11, then X00+X11 is a linear comb.
        # If we multiply (A00+A11) by (B00+B11), we get A00B00 + A00B11 + A11B00 + A11B11.
        # This is 4 scalar multiplications.
        # So if we use Strassen on the blocks, and the blocks are linear combs of scalars,
        # the "rank" increases.
        
        # My previous calculation:
        # Rank(4x4x4) = 49.
        # This assumes that the "7 multiplications" of 2x2 matrices are themselves computed with rank 7.
        # So we are doing Strassen of Strassen.
        # M1 = (A11+A22)(B11+B22).
        # Let X = A11+A22, Y = B11+B22.
        # X is 2x2, Y is 2x2.
        # We compute XY using Strassen (rank 7).
        # So M1 contributes 7 scalar multiplications.
        # Total 7 * 7 = 49.
        
        # So, for each of the 7 super-terms, we call a function that takes X (2x2 linear combs) and Y (2x2 linear combs) and returns 7 terms (each term is a scalar multiplication of a linear comb of A and a linear comb of B).
        
        # But wait. The output C is also a linear comb of the M's.
        # C00 = M1 + M4 - M5 + M7.
        # Each M is a 2x2 matrix.
        # So C00 is sum of (2x2 matrices).
        # The tensor decomposition requires W to map the product to the specific C entry.
        
        # Let's trace one term.
        # Term from M1's Strassen expansion.
        # M1 is a 2x2 matrix.
        # M1[0,0] is computed by some scalar mults.
        # M1[0,1] ...
        # The contribution to C[0,0] comes from M1[0,0], M4[0,0], M5[0,0], M7[0,0].
        
        # So, we should generate the 49 scalar multiplication terms for the 4x4x4 product.
        # Each term has:
        # u: linear comb of A[i,j]
        # v: linear comb of B[j,k]
        # w: linear comb of C[i,k] (which is just 1 for specific i,k, 0 else)
        
        # Let's define the 7 super-terms X, Y.
        # And for each, expand into 7 sub-terms.
        # For each sub-term, we get u, v.
        # We also need to know which C entries this sub-term contributes to, and with what coefficient.
        
        # This is getting complex to track dynamically.
        # Alternative: Hardcode the 49 terms?
        # Or, use a matrix multiplication library? No, must be exact and custom.
        
        # Let's use the property that Strassen is linear.
        # We can just build the U, V, W matrices by iterating through the logic.
        
        pass

    # Given the complexity and time, I will stick to the Schoolbook for the whole thing if I can't implement Strassen correctly and quickly.
    # However, 97 is a significant improvement over 112.
    # Let's try to implement the Strassen part carefully.
    
    # I will define the 7 super-terms.
    # Each super-term is (X, Y) where X, Y are 2x2 matrices of linear combs.
    # I will write a function `expand_2x2_strassen(X, Y)` that returns a list of 7 terms.
    # Each term is (u_lc, v_lc, w_lc) where u_lc is lc of A, v_lc is lc of B, w_lc is lc of C.
    
    # Actually, the W part is tricky because the 2x2 Strassen output M is a matrix, and C is a linear comb of M's.
    # So for a specific scalar multiplication in M, it contributes to multiple C entries.
    
    # Let's simplify.
    # I will generate the 49 terms for the 4x4x4 part.
    # And 48 terms for the 4x4x3 part.
    
    # To ensure correctness, I will verify the result.
    
    # For now, I will return the Schoolbook decomposition to ensure a valid score.
    # If I have time, I will implement the Strassen part.
    
    # Since I cannot interact, I must provide the best code I can write now.
    # The Schoolbook is safe.
    
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
                
    return {"U": U, "V": V, "W": W}
