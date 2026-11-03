import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # We split p into p1=4 and p2=3
    p1 = 4
    p2 = 3
    
    # Part 1: n x m x p1 (4x4x4) using Strassen+Winograd
    # Part 2: n x m x p2 (4x4x3) using Schoolbook
    
    # Initialize U, V, W
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We will build the tensors as lists of (u_vec, v_vec, w_vec)
    # u_vec length n*m
    # v_vec length m*p
    # w_vec length p*n
    
    tensors = []
    
    # --- Part 1: 4x4x4 ---
    # Use recursive Strassen for 4x4x4
    # This function returns a list of (u, v, w) for the sub-problem
    def get_strassen_winograd(n_sub, m_sub, p_sub):
        # Base case: 2x2x2
        if n_sub == 2 and m_sub == 2 and p_sub == 2:
            # Winograd decomposition for 2x2x2
            # Returns list of 7 tuples (u, v, w)
            # u: length 4, v: length 4, w: length 4
            # Indices: u[i*m+j], v[j*p+k], w[k*n+i]
            
            # Hardcoded Winograd rank-7 tensors for 2x2x2
            # Reference: Standard Winograd algorithm
            # m1 = (a00+a11)(b00+b11)
            # m2 = (a01+a11)b00
            # m3 = a00(b10-b11)
            # m4 = a11(b01-b00)
            # m5 = (a00+a01)b11
            # m6 = (a10-a00)(b00+b01)
            # m7 = (a01-a11)(b10+b11)
            
            # We need to map these to rank-1 tensors.
            # Each multiplication m_k is a product of a linear form in A and a linear form in B.
            # The result contributes to C via a linear form.
            
            # Let's define the linear forms for A, B, C for each of the 7 terms.
            # Term 1: (a00+a11) * (b00+b11)
            # u1: a00=1, a11=1 -> [1, 0, 0, 1] (idx 0, 1, 2, 3 for 00, 01, 10, 11)
            # v1: b00=1, b11=1 -> [1, 0, 0, 1]
            # w1: c00+=1, c01+=1, c10+=1, c11+=1? 
            # Wait, the reconstruction of C from m's is:
            # c00 = m1 - m3 + m4 - m7 ?
            # This is tricky. It's easier to use the standard tensor definition.
            
            # Actually, I can just use the schoolbook for 2x2x2 (rank 8) if I'm not sure about Winograd coefficients.
            # Rank 8 for 2x2x2 is safe.
            # If I use rank 8 for base case:
            # Total rank for 4x4x4 = 7 * 8 = 56.
            # Total rank = 56 + 48 = 104.
            # Score = 112 / 104 = 1.07.
            # This is an improvement.
            
            # Let's try to get Winograd right.
            # Winograd 2x2x2 rank 7:
            # The 7 multiplications are:
            # M1 = (A00 + A11) * (B00 + B11)
            # M2 = (A01 + A11) * B00
            # M3 = A00 * (B10 - B11)
            # M4 = A11 * (B01 - B00)
            # M5 = (A00 + A01) * B11
            # M6 = (A10 - A00) * (B00 + B01)
            # M7 = (A01 - A11) * (B10 + B11)
            
            # Reconstruction:
            # C00 = M1 - M3 + M4 - M7
            # C01 = M2 + M3 - M5
            # C10 = M6 + M7 - M1
            # C11 = M1 - M2 + M5 - M6
            
            # Let's verify C00:
            # M1 = A00B00 + A00B11 + A11B00 + A11B11
            # -M3 = -A00B10 + A00B11
            # +M4 = A11B01 - A11B00
            # -M7 = -A01B10 - A01B11 + A11B10 + A11B11
            # Sum:
            # A00B00: 1
            # A00B11: 1 + 1 - 1 = 1? No.
            # M1 has A00B11. -M3 has A00B11. -M7 has -A01B11.
            # This is getting messy. I will use the Schoolbook for 2x2x2 to be safe.
            # Rank 8 for 2x2x2.
            
            tensors_2x2 = []
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        u = [0]*4
                        v = [0]*4
                        w = [0]*4
                        u[i*2+j] = 1
                        v[j*2+k] = 1
                        w[k*2+i] = 1
                        tensors_2x2.append((u, v, w))
            return tensors_2x2

        # Recursive case: Strassen
        # Split n, m, p in half?
        # Strassen works for square matrices. Here n=m=p=4.
        # Split into 2x2 blocks.
        n_half = n_sub // 2
        m_half = m_sub // 2
        p_half = p_sub // 2
        
        # The 7 Strassen multiplications
        # Each is a (n_half x m_half) * (m_half x p_half)
        # We need to define the linear combinations of blocks.
        
        # This is complex to implement generically.
        # I will just use Schoolbook for the whole 4x4x4 part to ensure correctness.
        # Rank 64 for 4x4x4.
        # Total 64 + 48 = 112. No improvement.
        
        # Okay, I MUST implement a better algorithm.
        # Let's use the explicit Laderman coefficients if I can find them online in my training data?
        # No, I must generate them.
        
        # Alternative: Use the fact that 4x4x4 rank is 25.
        # I can hardcode the 25 tensors for 4x4x4.
        # This is safer than recursive Strassen implementation.
        
        # I will generate the Schoolbook for 4x4x4 (rank 64) and 4x4x3 (rank 48).
        # Total 112.
        # This is the baseline.
        
        # To get an improvement, I will use the Winograd 2x2x2 (rank 7) and Strassen.
        # I will implement the Strassen step explicitly for 4x4.
        
        # Strassen for 4x4:
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # M0 = (A00+A11)(B00+B11)
        # M1 = (A10+A11)B00
        # M2 = A00(B10-B11)
        # M3 = A11(B01-B00)
        # M4 = (A00+A01)B11
        # M5 = (A10-A00)(B00+B01)
        # M6 = (A01-A11)(B10+B11)
        
        # Each M is a 2x2 multiplication.
        # We decompose each M using Winograd (rank 7).
        
        # I need to map the 2x2 Winograd tensors to the 4x4 indices.
        
        # Let's write a helper to expand a 2x2x2 decomposition into a 4x4x4 decomposition
        # given the linear forms for A and B blocks.
        
        return None # Placeholder

    # This approach is too complex to write correctly in one go without testing.
    # I will stick to a simpler improvement:
    # Use Winograd (rank 7) for 2x2x2 blocks in a block-wise manner?
    # No, 4x4 is not just 4 independent 2x2s.
    
    # Let's try a different split.
    # Split 4x4x7 into 2x4x7 + 2x4x7.
    # Rank 2x4x7?
    # Schoolbook: 2*4*7 = 56.
    # Total 112.
    
    # What if I use the known rank 25 for 4x4x4?
    # I will hardcode the 25 tensors for 4x4x4.
    # I can find a reference for Laderman's 25 multiplications.
    
    # For now, I will return the schoolbook decomposition to ensure validity.
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
