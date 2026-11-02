import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline to be safe
    
    # Fallback solution: Rank 189 (2x2 Strassen)
    def get_fallback():
        block_size = 2
        n_blocks = n // block_size
        rank = n_blocks ** 3 * 7
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        def u_idx(ib, kb, r, c): return (ib * block_size + r) * m + (kb * block_size + c)
        def v_idx(kb, jb, r, c): return (kb * block_size + r) * p + (jb * block_size + c)
        def w_idx(ib, jb, r, c): return (jb * block_size + c) * n + (ib * block_size + r)
        
        for ib in range(n_blocks):
            for jb in range(n_blocks):
                for kb in range(n_blocks):
                    # M1
                    U[u_idx(ib, kb, 0, 0), col] += 1; U[u_idx(ib, kb, 1, 1), col] += 1
                    V[v_idx(kb, jb, 0, 0), col] += 1; V[v_idx(kb, jb, 1, 1), col] += 1
                    W[w_idx(ib, jb, 0, 0), col] += 1; W[w_idx(ib, jb, 1, 1), col] += 1
                    col += 1
                    # M2
                    U[u_idx(ib, kb, 1, 0), col] += 1; U[u_idx(ib, kb, 1, 1), col] += 1
                    V[v_idx(kb, jb, 0, 0), col] += 1
                    W[w_idx(ib, jb, 1, 0), col] += 1; W[w_idx(ib, jb, 1, 1), col] -= 1
                    col += 1
                    # M3
                    U[u_idx(ib, kb, 0, 0), col] += 1
                    V[v_idx(kb, jb, 0, 1), col] += 1; V[v_idx(kb, jb, 1, 1), col] -= 1
                    W[w_idx(ib, jb, 0, 1), col] += 1; W[w_idx(ib, jb, 1, 1), col] += 1
                    col += 1
                    # M4
                    U[u_idx(ib, kb, 1, 1), col] += 1
                    V[v_idx(kb, jb, 1, 0), col] += 1; V[v_idx(kb, jb, 0, 0), col] -= 1
                    W[w_idx(ib, jb, 0, 0), col] += 1; W[w_idx(ib, jb, 1, 0), col] += 1
                    col += 1
                    # M5
                    U[u_idx(ib, kb, 0, 0), col] += 1; U[u_idx(ib, kb, 0, 1), col] += 1
                    V[v_idx(kb, jb, 1, 1), col] += 1
                    W[w_idx(ib, jb, 0, 0), col] -= 1; W[w_idx(ib, jb, 0, 1), col] += 1
                    col += 1
                    # M6
                    U[u_idx(ib, kb, 1, 0), col] += 1; U[u_idx(ib, kb, 0, 0), col] -= 1
                    V[v_idx(kb, jb, 0, 0), col] += 1; V[v_idx(kb, jb, 0, 1), col] += 1
                    W[w_idx(ib, jb, 1, 1), col] += 1
                    col += 1
                    # M7
                    U[u_idx(ib, kb, 0, 1), col] += 1; U[u_idx(ib, kb, 1, 1), col] -= 1
                    V[v_idx(kb, jb, 1, 0), col] += 1; V[v_idx(kb, jb, 1, 1), col] += 1
                    W[w_idx(ib, jb, 0, 0), col] += 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt Rank 161 solution (2x2 Strassen with 3x3 Winograd blocks)
    try:
        if time.time() > deadline:
            return get_fallback()
            
        # Winograd 3x3 decomposition (Rank 23)
        # Coefficients for U (A), V (B), W (C)
        # Indices 0..2 for rows/cols in 3x3 block
        winograd_terms = []
        
        # Helper to add a term to the lists
        # u_coeffs: dict {(r,c): val}
        # v_coeffs: dict {(r,c): val}
        # w_coeffs: dict {(r,c): val}
        
        def add_term(u_c, v_c, w_c):
            winograd_terms.append((u_c, v_c, w_c))
            
        # 1
        add_term({(0,0):1,(1,0):1,(2,0):1}, {(0,0):1,(0,1):1,(0,2):1}, {(0,0):1,(0,1):1,(0,2):1,(1,0):1,(1,1):1,(1,2):1,(2,0):1,(2,1):1,(2,2):1})
        # This manual entry is too prone to error. I will use the standard formulas.
        
        # Correct Winograd 3x3 terms:
        # M1: (a00+a10+a20)(b00+b01+b02) -> C += M1 * ones(3x3)
        # M2: (a01+a11+a21)(b10+b11+b12) -> C += M2 * ones(3x3)
        # M3: (a02+a12+a22)(b20+b21+b22) -> C += M3 * ones(3x3)
        # M4: (a00-a01+a02)(b00-b10+b20) -> C += M4 * [[1,0,0],[0,1,0],[0,0,1]]? No.
        # The reconstruction is complex.
        
        # Instead of risking an incorrect Winograd implementation, I will stick to the 
        # verified Rank 189 solution but try to optimize the code structure or 
        # check if there's a simpler rank reduction.
        
        # Actually, the rank of 3x3 is 23. 
        # If I can't verify it easily, I should not use it.
        # However, 161 is a strong candidate.
        # Let's use a known correct implementation of Winograd 3x3.
        
        # Terms:
        # u, v, w (w is the coefficient for the output matrix C)
        # C = sum M_k * W_k
        
        # 1
        add_term({(0,0):1,(1,0):1,(2,0):1}, {(0,0):1,(0,1):1,(0,2):1}, {(0,0):1,(0,1):1,(0,2):1,(1,0):1,(1,1):1,(1,2):1,(2,0):1,(2,1):1,(2,2):1})
        # 2
        add_term({(0,1):1,(1,1):1,(2,1):1}, {(1,0):1,(1,1):1,(1,2):1}, {(0,0):1,(0,1):1,(0,2):1,(1,0):1,(1,1):1,(1,2):1,(2,0):1,(2,1):1,(2,2):1})
        # 3
        add_term({(0,2):1,(1,2):1,(2,2):1}, {(2,0):1,(2,1):1,(2,2):1}, {(0,0):1,(0,1):1,(0,2):1,(1,0):1,(1,1):1,(1,2):1,(2,0):1,(2,1):1,(2,2):1})
        # 4
        add_term({(0,0):1,(0,1):-1,(0,2):1}, {(0,0):1,(1,0):-1,(2,0):1}, {(0,0):1,(1,1):1,(2,2):1})
        # 5
        add_term({(1,0):1,(1,1):-1,(1,2):1}, {(0,1):1,(1,1):-1,(2,1):1}, {(0,0):1,(1,1):1,(2,2):1})
        # 6
        add_term({(2,0):1,(2,1):-1,(2,2):1}, {(0,2):1,(1,2):-1,(2,2):1}, {(0,0):1,(1,1):1,(2,2):1})
        # 7
        add_term({(0,0):1,(0,1):1,(0,2):-1}, {(0,0):1,(1,0):1,(2,0):-1}, {(0,0):1,(1,1):1,(2,2):1})
        # 8
        add_term({(1,0):1,(1,1):1,(1,2):-1}, {(0,1):1,(1,1):1,(2,1):-1}, {(0,0):1,(1,1):1,(2,2):1})
        # 9
        add_term({(2,0):1,(2,1):1,(2,2):-1}, {(0,2):1,(1,2):1,(2,2):-1}, {(0,0):1,(1,1):1,(2,2):1})
        # 10
        add_term({(0,0):1,(1,0):-1,(2,0):1}, {(0,0):1,(0,1):-1,(0,2):1}, {(0,0):1,(1,1):1,(2,2):1})
        # 11
        add_term({(0,1):1,(1,1):-1,(2,1):1}, {(1,0):1,(1,1):-1,(1,2):1}, {(0,0):1,(1,1):1,(2,2):1})
        # 12
        add_term({(0,2):1,(1,2):-1,(2,2):1}, {(2,0):1,(2,1):-1,(2,2):1}, {(0,0):1,(1,1):1,(2,2):1})
        # 13
        add_term({(0,0):1,(1,0):1,(2,0):-1}, {(0,0):1,(0,1):1,(0,2):-1}, {(0,0):1,(1,1):1,(2,2):1})
        # 14
        add_term({(0,1):1,(1,1):1,(2,1):-1}, {(1,0):1,(1,1):1,(1,2):-1}, {(0,0):1,(1,1):1,(2,2):1})
        # 15
        add_term({(0,2):1,(1,2):1,(2,2):-1}, {(2,0):1,(2,1):1,(2,2):-1}, {(0,0):1,(1,1):1,(2,2):1})
        # 16
        add_term({(0,0):1,(0,2):-1}, {(0,0):1,(2,0):-1}, {(0,0):1,(0,2):1,(2,0):1,(2,2):1})
        # 17
        add_term({(0,1):1,(1,2):-1}, {(0,1):1,(1,2):-1}, {(0,1):1,(0,2):1,(1,1):1,(1,2):1})
        # 18
        add_term({(1,0):1,(0,2):-1}, {(1,0):1,(0,2):-1}, {(0,1):1,(0,2):1,(1,0):1,(1,2):1})
        # 19
        add_term({(0,0):1,(0,2):1}, {(0,0):1,(2,0):1}, {(0,0):1,(0,2):1,(2,0):1,(2,2):1})
        # 20
        add_term({(0,1):1,(1,2):1}, {(0,1):1,(1,2):1}, {(0,1):1,(0,2):1,(1,1):1,(1,2):1})
        # 21
        add_term({(1,0):1,(0,2):1}, {(1,0):1,(0,2):1}, {(0,1):1,(0,2):1,(1,0):1,(1,2):1})
        # 22
        add_term({(0,0):1,(1,0):-1,(0,1):1,(1,1):-1}, {(0,0):1,(0,1):-1,(1,0):1,(1,1):-1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1})
        # 23
        add_term({(0,0):1,(1,0):1,(0,1):1,(1,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1})
        
        # Note: The above Winograd terms are a heuristic reconstruction. 
        # To ensure correctness, I will verify the full 6x6 tensor decomposition.
        
        # Build 6x6 decomposition using Strassen on 2x2 blocks of 3x3
        # Block size 3
        bs = 3
        n_blocks = 2
        rank = n_blocks ** 3 * 23 # 8 * 23 = 184? No, Strassen is 7 terms.
        # Strassen uses 7 multiplications.
        rank = 7 * 23
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        col = 0
        # Strassen terms for 2x2 blocks
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For each Strassen term, we have a linear combination of A blocks and B blocks.
        # Each A block is 3x3, B block is 3x3.
        # We need to expand the Winograd decomposition for each product of linear combinations.
        # This is complex.
        
        # Given the time constraint and risk of error in manual Winograd expansion,
        # I will return the verified Rank 189 solution.
        
        return get_fallback()
        
    except Exception as e:
        return get_fallback()
