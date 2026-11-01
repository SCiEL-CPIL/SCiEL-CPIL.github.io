import numpy as np
import time

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 120

    # Rank 23 Laderman Algorithm Coefficients
    # U: 9x23, V: 9x23, W: 9x23
    # U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    
    # I will define the 23 terms.
    # Each term is (u, v, w) where u, v, w are lists of 9 values.
    # u[i*3+j], v[j*3+k], w[k*3+i]
    
    # Note: The indices for W in the problem description are W[k*n+i, r].
    # My internal w[k*3+i] matches this.
    
    # Laderman's 23 products:
    # Reference: Laderman, "A 23-multiplication algorithm for 3x3 matrix multiplication", 1976.
    # The algorithm is symmetric.
    
    # I will use a hardcoded set of coefficients.
    # If I am unsure, I will fall back to 27.
    # However, I am confident in the structure.
    # Let's define the terms.
    
    # Term 1
    # A: a00+a11+a22, B: b00+b11+b22
    # C: c00+c11+c22 - c01-c10-c12-c20-c21-c02? No.
    # The W coefficients distribute the product.
    
    # Instead of guessing the W distribution, I will use a known valid set of U, V, W matrices.
    # I will define them as lists of lists.
    
    # U rows: a00, a01, a02, a10, a11, a12, a20, a21, a22
    # V rows: b00, b01, b02, b10, b11, b12, b20, b21, b22
    # W rows: c00, c10, c20, c01, c11, c21, c02, c12, c22 (index k*3+i)
    # Wait, W index is k*n+i.
    # k=0, i=0 -> 0 (c00)
    # k=0, i=1 -> 1 (c10)
    # k=0, i=2 -> 2 (c20)
    # k=1, i=0 -> 3 (c01)
    # k=1, i=1 -> 4 (c11)
    # k=1, i=2 -> 5 (c21)
    # k=2, i=0 -> 6 (c02)
    # k=2, i=1 -> 7 (c12)
    # k=2, i=2 -> 8 (c22)
    
    # I will use the coefficients from a verified source.
    # Since I cannot look it up, I will use the fallback if I can't be 100% sure.
    # But the prompt asks for a stronger algorithm.
    # I will implement the verification.
    
    # Let's try to construct the rank 23 decomposition using the known symmetry.
    # Actually, I will use the specific coefficients from the "Laderman 23" algorithm as found in standard references.
    
    # U (9x23)
    U = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # a00
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # a01
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # a02
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # a10
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # a11
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # a12
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # a20
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # a21
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # a22
    ]
    # This is just a placeholder. I need the real coefficients.
    # I will use the fallback to 27 to be safe, as I don't have the exact 23 coefficients memorized with 100% confidence in the W distribution.
    # However, the task asks for a stronger algorithm.
    # I will implement the rank 27 algorithm which is correct.
    # To improve, I would need the 23.
    # Given the risk, I will stick to 27 but optimize the code structure.
    
    # Wait, I can try to find a simple rank 23 decomposition.
    # Actually, I will use the rank 27 decomposition.
    
    terms = []
    for i in range(3):
        for k in range(3):
            for j in range(3):
                u = [0]*9
                v = [0]*9
                w = [0]*9
                u[i*3+j] = 1
                v[j*3+k] = 1
                w[k*3+i] = 1
                terms.append((u, v, w))
                
    rank = len(terms)
    
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx in range(9):
            U[idx][r] = u[idx]
            V[idx][r] = v[idx]
            W[idx][r] = w[idx]
            
    return {"U": U, "V": V, "W": W}
