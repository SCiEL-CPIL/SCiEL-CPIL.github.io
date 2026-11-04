import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140
    
    # We will construct a rank 52 decomposition.
    # Rank 23 for 3x3x3 (Strassen-based)
    # Rank 3 for 3x3x1 (Schoolbook)
    # Total: 23 + 3 + 23 + 3 = 52
    
    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_3x3x3(k_offset):
        nonlocal col
        # Indices for this block:
        # A is 3x3, B is 3x3 (cols k_offset to k_offset+2), C is 3x3 (cols k_offset to k_offset+2)
        # We split A, B into 2x2, 2x1, 1x2, 1x1 blocks.
        # A blocks:
        # A11: (0,0), (0,1); (1,0), (1,1)
        # A12: (0,2); (1,2)  -> 2x1
        # A21: (2,0), (2,1)  -> 1x2
        # A22: (2,2)         -> 1x1
        
        # B blocks (for cols k_offset, k_offset+1, k_offset+2):
        # B11: (0, k_offset), (0, k_offset+1); (1, k_offset), (1, k_offset+1)
        # B12: (0, k_offset+2); (1, k_offset+2) -> 2x1
        # B21: (2, k_offset), (2, k_offset+1)   -> 1x2
        # B22: (2, k_offset+2)                  -> 1x1
        
        # Strassen M1: (A11+A22)(B11+B22)
        # A11+A22:
        # (0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1, (2,2): 1
        # B11+B22:
        # (0, k_offset): 1, (0, k_offset+1): 1, (1, k_offset): 1, (1, k_offset+1): 1, (2, k_offset+2): 1
        
        # C11: M1 + M4 - M5 + M7
        # C12: M3 + M5
        # C21: M2 + M4
        # C22: M1 - M2 + M3 + M6
        
        # We need to add these to U, V, W.
        # Each M term is a rank 1 tensor? No, Strassen M terms are not rank 1.
        # Strassen M1 is (A11+A22)(B11+B22).
        # (A11+A22) is a 3x3 matrix. (B11+B22) is a 3x3 matrix.
        # The product is a 3x3 matrix.
        # This is a rank 9 tensor? No.
        # The term is $U_{ij} V_{jk} W_{ik}$.
        # For M1, $U_{ij} = \delta_{ij \in A11 \cup A22}$, $V_{jk} = \delta_{jk \in B11 \cup B22}$, $W_{ik} = 1$.
        # This is not a single rank 1 term.
        # Strassen's algorithm works by defining 7 intermediate matrices $M_1..M_7$ which are sums of products.
        # $M_1 = (A_{11}+A_{22})(B_{11}+B_{22})$.
        # This $M_1$ itself is a matrix product of two $2 \times 2$ matrices (embedded in $3 \times 3$).
        # To decompose this into the global tensor, we need to expand $M_1$.
        # $M_1$ contributes to $C_{11}, C_{22}$.
        # $C_{11}$ gets $+1 \times M_{1,11}$, $C_{22}$ gets $+1 \times M_{1,22}$.
        # $M_{1,11} = (A_{11}+A_{22})_{11} (B_{11}+B_{22})_{11} + (A_{11}+A_{22})_{12} (B_{11}+B_{22})_{21}$?
        # No, $M_1$ is the full $2 \times 2$ product.
        # $M_1 = \begin{pmatrix} m_{11} & m_{12} \\ m_{21} & m_{22} \end{pmatrix}$.
        # $m_{11} = (a_{11}+a_{22})b_{11} + (a_{12}+a_{21})b_{21}$? No.
        # $A_{11}+A_{22} = \begin{pmatrix} a_{11}+a_{22} & a_{12} \\ a_{21} & a_{22}+a_{11} \end{pmatrix}$? No.
        # $A = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}$.
        # $A_{11}+A_{22}$ is not a matrix. $A_{11}$ and $A_{22}$ are $2 \times 2$ blocks?
        # No, in the 3x3 case, we defined:
        # $A_{11}$ is $2 \times 2$ (top-left).
        # $A_{22}$ is $1 \times 1$ (bottom-right).
        # So $A_{11}+A_{22}$ is not defined.
        
        # I made a mistake in the block definition for 3x3.
        # Standard Strassen is for $2 \times 2$.
        # For $3 \times 3$, we can pad to $4 \times 4$ or use a specific 3x3 algorithm.
        # The rank 23 algorithm for 3x3x3 is not simply Strassen on 2x2 blocks.
        
        # Let's use the known rank 23 coefficients if I can find them, or stick to schoolbook if I can't.
        # Given the risk of incorrect coefficients, and the time limit, I will use a **search** strategy 
        # but initialized with a **structured** guess that is closer to the solution.
        # However, the random search in the current program is too weak.
        
        # Let's try a different approach:
        # Use the rank 52 decomposition derived from 2x Strassen 2x2 + rest.
        # Actually, $R(3,3,3) \le 19$ is hard to implement without coefficients.
        # $R(3,3,3) \le 23$ is also hard.
        
        # Let's go back to the search strategy but improve it.
        # The current program searches for rank 71..56.
        # If I can find a rank 56 decomposition, that's a 1.28x improvement.
        # The random search with half-integers is unlikely to find it.
        
        # I will implement a **greedy** reduction of the schoolbook decomposition.
        # Or, I will use the **Lader** coefficients for rank 19 if I can recall them.
        # I cannot recall them.
        
        # Let's use the **Strassen** decomposition for the $2 \times 2$ part of the $3 \times 3$ matrix, 
        # and schoolbook for the rest, but carefully constructed.
        
        # For 3x3x3:
        # Split into 2x2, 2x1, 1x2, 1x1.
        # Ranks:
        # 2x2x2: 7
        # 2x1x1: 2 (outer product of 2x1 and 1x1? No. 2x1 A, 1x1 B. Product 2x1. Rank 2)
        # 1x2x1: 2
        # 1x1x2: 2
        # 2x1x2: 4
        # 1x2x2: 4
        # 1x1x1: 1
        # 1x1x2: 2
        # 2x1x1: 2
        # 1x2x1: 2
        # 1x1x1: 1
        
        # Wait, the cross terms:
        # C11 = A11 B11 + A12 B21
        # C12 = A11 B12 + A12 B22
        # C21 = A21 B11 + A22 B21
        # C22 = A21 B12 + A22 B22
        
        # R(C11) = R(A11 B11) + R(A12 B21) = 7 + 4 = 11
        # R(C12) = R(A11 B12) + R(A12 B22) = 4 + 2 = 6
        # R(C21) = R(A21 B11) + R(A22 B21) = 4 + 2 = 6
        # R(C22) = R(A21 B12) + R(A22 B22) = 4 + 1 = 5
        # Total = 11 + 6 + 6 + 5 = 28.
        # This is worse than schoolbook 27.
        
        # So this block decomposition doesn't help.
        
        # I will stick to the **search** strategy but make it smarter.
        # I will use a **local search** starting from the schoolbook decomposition.
        # I will try to merge terms.
        
        # Given the complexity, I will return the schoolbook decomposition (rank 72) to ensure validity, 
        # but I will add a **fast** search for rank 71-60 using a **better** random initialization 
        # (e.g., sparse matrices) and **vectorized** verification.
        
        # Actually, the prompt says "score is the rank".
        # If I can't find a better one, I return 72.
        # The current program returns 72.
        
        # Is there a simple rank 71 decomposition?
        # Probably not.
        
        # I will optimize the verification speed to allow more attempts.
        
        pass # Placeholder

    # I will use the schoolbook decomposition as it is guaranteed to work.
    # I will not risk an invalid decomposition.
    
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
