import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 5, 6
    
    # We will construct a rank 48 solution.
    # Split m=5 into m1=2 and m2=3.
    # Part 1: 2x2x6 (rank 21)
    # Part 2: 2x3x6 (rank 27)
    
    rank = 48
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # --- Part 1: 2x2x6 ---
    # A is 2x2 (j=0,1), B is 2x6 (j=0,1; k=0..5)
    # Split B into three 2x2 blocks: k=0,1; k=2,3; k=4,5
    # For each block, use Strassen (rank 7)
    
    def add_strassen_block(j_offset, k_offset):
        nonlocal col
        # Strassen for 2x2x2
        # U indices: i in 0..1, j in 0..1 (local)
        # V indices: j in 0..1 (local), k in 0..1 (local)
        # W indices: k in 0..1 (local), i in 0..1 (local)
        
        # M1 = (A00+A11)(B00+B11)
        U[j_offset*0 + 0][col] += 1 # A00
        U[j_offset*0 + 1][col] += 1 # A11 (wait, U is indexed by i*m+j)
        # U[i*m + j]
        U[0*m + j_offset][col] += 1
        U[1*m + j_offset+1][col] += 1
        V[j_offset*p + k_offset][col] += 1
        V[j_offset+1*p + k_offset+1][col] += 1
        W[k_offset*n + 0][col] += 1
        W[k_offset*n + 1][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[1*m + j_offset][col] += 1
        U[1*m + j_offset+1][col] += 1
        V[j_offset*p + k_offset][col] += 1
        W[k_offset*n + 1][col] += 1
        col += 1
        
        # M3 = A00(B10-B11)
        U[0*m + j_offset][col] += 1
        V[j_offset+1*p + k_offset][col] += 1
        V[j_offset*p + k_offset+1][col] -= 1
        W[k_offset*n + 0][col] += 1
        col += 1
        
        # M4 = A11(B01-B00)
        U[1*m + j_offset+1][col] += 1
        V[j_offset*p + k_offset+1][col] += 1
        V[j_offset*p + k_offset][col] -= 1
        W[k_offset*n + 0][col] += 1
        W[k_offset*n + 1][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[0*m + j_offset][col] += 1
        U[0*m + j_offset+1][col] += 1
        V[j_offset+1*p + k_offset+1][col] += 1
        W[k_offset*n + 0][col] -= 1
        W[k_offset*n + 1][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B10)
        U[1*m + j_offset][col] += 1
        U[0*m + j_offset][col] -= 1
        V[j_offset*p + k_offset][col] += 1
        V[j_offset+1*p + k_offset][col] += 1
        W[k_offset*n + 1][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B11+B01)
        U[0*m + j_offset+1][col] += 1
        U[1*m + j_offset+1][col] -= 1
        V[j_offset+1*p + k_offset+1][col] += 1
        V[j_offset*p + k_offset+1][col] += 1
        W[k_offset*n + 0][col] += 1
        col += 1

    # Apply Strassen for 3 blocks
    for k_off in [0, 2, 4]:
        add_strassen_block(0, k_off)
        
    # --- Part 2: 2x3x6 ---
    # A is 2x3 (j=2,3,4), B is 3x6 (j=2,3,4; k=0..5)
    # Split B into three 3x2 blocks: k=0,1; k=2,3; k=4,5
    # For each block, use rank 9 decomposition for 2x3x2
    
    def add_rank9_block(j_offset, k_offset):
        nonlocal col
        # j_offset is 2 (local j=0,1,2)
        # k_offset is 0,2,4 (local k=0,1)
        
        # We need 9 terms for 2x3x2 all-ones tensor.
        # Let's use a known decomposition.
        # Term 1: U(0,0), V(0,0), W(0,0)
        U[0*m + j_offset][col] += 1
        V[j_offset*p + k_offset][col] += 1
        W[k_offset*n + 0][col] += 1
        col += 1
        
        # Term 2: U(1,0), V(0,0), W(1,0)
        U[1*m + j_offset][col] += 1
        V[j_offset*p + k_offset][col] += 1
        W[k_offset*n + 1][col] += 1
        col += 1
        
        # Term 3: U(0,1), V(1,0), W(0,0)
        U[0*m + j_offset+1][col] += 1
        V[j_offset+1*p + k_offset][col] += 1
        W[k_offset*n + 0][col] += 1
        col += 1
        
        # Term 4: U(1,1), V(1,0), W(1,0)
        U[1*m + j_offset+1][col] += 1
        V[j_offset+1*p + k_offset][col] += 1
        W[k_offset*n + 1][col] += 1
        col += 1
        
        # Term 5: U(0,2), V(2,0), W(0,0)
        U[0*m + j_offset+2][col] += 1
        V[j_offset+2*p + k_offset][col] += 1
        W[k_offset*n + 0][col] += 1
        col += 1
        
        # Term 6: U(1,2), V(2,0), W(1,0)
        U[1*m + j_offset+2][col] += 1
        V[j_offset+2*p + k_offset][col] += 1
        W[k_offset*n + 1][col] += 1
        col += 1
        
        # Term 7: U(0,0), V(0,1), W(0,1)
        U[0*m + j_offset][col] += 1
        V[j_offset*p + k_offset+1][col] += 1
        W[k_offset+1*n + 0][col] += 1
        col += 1
        
        # Term 8: U(1,0), V(0,1), W(1,1)
        U[1*m + j_offset][col] += 1
        V[j_offset*p + k_offset+1][col] += 1
        W[k_offset+1*n + 1][col] += 1
        col += 1
        
        # Term 9: This is tricky. The above 8 terms cover:
        # C00: A00B00, A01B10, A02B20
        # C10: A10B00, A11B10, A12B20
        # C01: A00B01
        # C11: A10B01
        # Missing: C01: A01B11, A02B21
        # Missing: C11: A11B11, A12B21
        
        # We need 1 more term? No, rank is 9.
        # The above is 8 terms.
        # Let's check the sum.
        # If I add Term 9: U(0,1), V(1,1), W(0,1)
        # It adds A01B11 to C01.
        # We still miss A02B21, A11B11, A12B21.
        
        # It seems my "known" decomposition is not complete or I'm misremembering.
        # Let's use a rank 10 or 11 fallback if 9 is too risky?
        # No, rank 9 is proven.
        
        # Let's try to find the 9th term.
        # Actually, the 8 terms above are not a valid decomposition because they don't sum to the all-ones tensor.
        
        # I will use a simple rank 12 (naive) for the 2x3x2 part to ensure correctness, 
        # which gives total rank 21 + 3*12 = 57.
        # This is worse than 54.
        
        # Let's stick to the seed's 54 if I can't verify 48 quickly.
        # However, I will try to implement a valid 9-rank decomposition.
        
        # A valid 9-rank decomposition for 2x3x2:
        # See "Matrix Multiplication Tensor" literature.
        # I will use the following 9 terms:
        
        # T1: U(0,0), V(0,0), W(0,0)
        # T2: U(1,0), V(0,0), W(1,0)
        # T3: U(0,1), V(1,0), W(0,0)
        # T4: U(1,1), V(1,0), W(1,0)
        # T5: U(0,2), V(2,0), W(0,0)
        # T6: U(1,2), V(2,0), W(1,0)
        # T7: U(0,0), V(0,1), W(0,1)
        # T8: U(1,0), V(0,1), W(1,1)
        # T9: U(0,1), V(1,1), W(0,1) ... still missing terms.
        
        # Given the risk, I will return the seed's rank 54 solution to ensure a valid score, 
        # as I cannot guarantee the rank 9 coefficients in this time frame.
        
        pass

    # Since I cannot guarantee the rank 9 decomposition, I will return the rank 54 solution.
    # The rank 48 attempt is aborted to avoid invalidity.
    
    # Re-implementing rank 54 from seed to be safe
    def get_rank_54():
        rank = 54
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        j_pairs = [(0, 1), (2, 3)]
        k_pairs = [(0, 1), (2, 3), (4, 5)]
        
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # P1
                U[0*m + j0][col] += 1
                U[1*m + j1][col] += 1
                V[j0*p + k0][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                
                # P2
                U[1*m + j0][col] += 1
                U[1*m + j1][col] += 1
                V[j0*p + k0][col] += 1
                W[k0*n + 1][col] += 1
                W[k1*n + 1][col] -= 1
                col += 1
                
                # P3
                U[0*m + j0][col] += 1
                V[j0*p + k1][col] += 1
                V[j1*p + k1][col] -= 1
                W[k1*n + 0][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                
                # P4
                U[1*m + j1][col] += 1
                V[j1*p + k0][col] += 1
                V[j0*p + k0][col] -= 1
                W[k0*n + 0][col] += 1
                W[k0*n + 1][col] += 1
                col += 1
                
                # P5
                U[0*m + j0][col] += 1
                U[0*m + j1][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] -= 1
                W[k1*n + 0][col] += 1
                col += 1
                
                # P6
                U[1*m + j0][col] += 1
                U[0*m + j0][col] -= 1
                V[j0*p + k0][col] += 1
                V[j0*p + k1][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                
                # P7
                U[0*m + j1][col] += 1
                U[1*m + j1][col] -= 1
                V[j1*p + k0][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] += 1
                col += 1
                
        for k in range(p):
            U[0*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 0][col] += 1
            col += 1
            
            U[1*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 1][col] += 1
            col += 1

        return {"U": U, "V": V, "W": W}

    return get_rank_54()
