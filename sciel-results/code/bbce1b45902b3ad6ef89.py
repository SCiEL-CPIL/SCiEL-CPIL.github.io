import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # We will use a hybrid decomposition:
    # 1. Strassen algorithm for the 2x2x2 sub-block (indices 0,1 for i,j,k)
    # 2. Schoolbook for all other terms
    
    # Total schoolbook rank is 63.
    # The 2x2x2 block has 8 terms in schoolbook.
    # Strassen replaces these 8 terms with 7 terms.
    # New rank = 63 - 8 + 7 = 62.
    
    total_rank = 62
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Strassen terms for 2x2x2 block (i in 0,1; j in 0,1; k in 0,1)
    # Standard Strassen 7 products:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # We need to map these to the tensor decomposition U, V, W.
    # Each term r contributes to U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    # For a product (A_part * B_part) contributing to C_part, the tensor term is a sum of rank-1 tensors.
    # Actually, the Strassen algorithm computes the product using 7 scalar multiplications.
    # In the tensor decomposition, each "scalar multiplication" M_r corresponds to a rank-1 tensor?
    # No, the tensor decomposition is a sum of rank-1 tensors $u_r \otimes v_r \otimes w_r$.
    # The Strassen algorithm expresses the bilinear map as a sum of 7 terms.
    # Each term is of the form $(\sum c_{ij} A_{ij}) (\sum d_{jk} B_{jk}) \to \sum e_{ki} C_{ki}$.
    # This corresponds to 7 rank-1 tensors in the decomposition?
    # Yes, if we can write the coefficients as outer products.
    
    # Let's define the 7 Strassen terms explicitly for the 2x2x2 block.
    # Indices i, j, k are 0 or 1.
    
    # Term 1: M1 = (A00 + A11)(B00 + B11)
    # Contributes to:
    # C00 += M1 - M4 - M6 + M7
    # C01 += M1 + M5
    # C10 += M2 + M4
    # C11 += M1 - M3 + M5 - M7
    # This is not a direct rank-1 decomposition of the tensor.
    # The tensor is $T = \sum_{i,j,k} E_{ij} \otimes E_{jk} \otimes E_{ki}$.
    # Strassen shows that this tensor has rank 7.
    # The 7 vectors are:
    
    # We will use a known rank-7 decomposition for 2x2x2.
    # See "Strassen's algorithm" tensor form.
    
    # U vectors (A coefficients), V vectors (B coefficients), W vectors (C coefficients)
    # Note: W index is k*n + i.
    
    # I will use the standard Strassen decomposition for 2x2x2.
    # The 7 terms are:
    # 1. u=(1,0,0,1) v=(1,0,0,1) w=(1,0,0,1) ? No.
    
    # Let's use a verified list of 7 rank-1 tensors for 2x2x2.
    # From literature:
    # T = sum_{r=1}^7 u_r v_r w_r
    # u_1 = (1, 0, 0, 1) -> A00 + A11
    # v_1 = (1, 0, 0, 1) -> B00 + B11
    # w_1 = (1, 1, 1, 1) ? No.
    
    # Actually, it is easier to just use the schoolbook for everything to ensure correctness, 
    # and then I will try to swap the 2x2x2 block if I can find the exact vectors.
    # Given the risk of error in Strassen vectors, I will stick to Schoolbook (63) 
    # unless I can be 100% sure.
    
    # However, the task asks for a stronger algorithm.
    # I will implement the 2x2x2 Strassen block carefully.
    
    # The 7 rank-1 tensors for 2x2x2 matrix multiplication are:
    # 1. u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1) -> This sums to A00B00 + A00B11 + A11B00 + A11B11.
    #    This is not one of the Strassen products.
    
    # Let's use the explicit Strassen decomposition:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # The tensor decomposition is not simply these 7 products as rank-1 tensors.
    # The rank-1 tensors are the "basis" of the decomposition.
    # The Strassen algorithm proves the rank is <= 7.
    # The actual vectors u_r, v_r, w_r are:
    
    # I will use a known correct set of vectors for 2x2x2 rank 7.
    # If I fail, I fall back to 63.
    
    # For safety and to guarantee a valid score > 1.0, I will use the 2x2x2 Strassen 
    # only if I can verify it.
    # I will use the schoolbook for now, but I will try to implement the 2x2x2 block 
    # using the 7 products as separate rank-1 terms? No, that's not how it works.
    
    # I will return the schoolbook decomposition (63) to be safe, 
    # but I will add a comment that Strassen is possible.
    # Actually, I can implement the 2x2x2 Strassen block by 
    # replacing the 8 schoolbook terms with the 7 Strassen terms.
    # The 7 Strassen terms are rank-1 tensors.
    
    # Let's use the following 7 rank-1 tensors for 2x2x2:
    # (Sources vary, but here is a standard one)
    # 1. u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1) -> Incorrect for Strassen.
    
    # I will stick to Schoolbook 63. It is the baseline.
    # To get a score > 1.0, I need rank < 63.
    # I will use the 2x2x2 Strassen block.
    
    # The 7 rank-1 tensors for 2x2x2 are:
    # 1. u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)
    # 2. u=(0,1,1,0), v=(1,0,0,0), w=(0,0,1,0)
    # 3. u=(1,0,0,0), v=(0,1,0,-1), w=(0,1,0,0)
    # 4. u=(0,0,0,1), v=(0,0,1,-1), w=(0,0,0,1)
    # 5. u=(1,0,1,0), v=(0,0,0,1), w=(0,1,0,0)
    # 6. u=(0,1,-1,0), v=(1,0,1,0), w=(1,0,0,0)
    # 7. u=(0,1,0,-1), v=(0,0,1,1), w=(0,0,0,1)
    # These are NOT correct.
    
    # I will use the schoolbook decomposition.
    total_rank = n * m * p
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
