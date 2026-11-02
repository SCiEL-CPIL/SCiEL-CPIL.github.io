"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

DIMENSIONS = [4, 4, 6]
_START_TIME = time.time()
_TIME_LIMIT = 115  # Leave margin before 120s limit

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook baseline
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(nm)]
    V = [[0] * rank_school for _ in range(mp)]
    W = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best = {'U': U, 'V': V, 'W': W, 'rank': rank_school}
    
    # Verify function
    def verify(U, V, W, rank, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Deadline guard
    deadline = _START_TIME + _TIME_LIMIT
    
    # Try to find lower rank using structured block decomposition
    # Represent 4x4x6 as blocks of 2x2, 2x4, 4x2, 4x4 multiplications
    # Use known low-rank decompositions for smaller cases
    
    random.seed(42)
    
    # Known: 2x2x2 matrix multiplication has rank 7 (Strassen)
    # We can decompose 4x4 into 4 blocks of 2x2, giving rank 4*7 = 28 for 4x4x4
    # For 4x4x6, we have 4 blocks of 2x2x2 (left 2x2, right 2x2) plus cross terms
    
    # Let's try a direct construction using Strassen-like blocks
    # Split indices into halves: i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    
    # For each (i1,j1,k1) block combination, we need a 2x2x2 decomposition
    # There are 2*2*3 = 12 such block combinations (i1 in {0,1}, j1 in {0,1}, k1 in {0,1,2})
    
    # Actually, let's think more carefully. The 4x4x6 multiplication can be seen as:
    # For each output (i,k), sum over j of A[i,j]*B[j,k]
    # Split i into (i1,i2), j into (j1,j2), k into (k1,k2) where i1,j1,k1 in {0,1}, i2,j2 in {0,1}, k2 in {0,1}
    # But k has 6 elements, so k1 in {0,1,2} and k2 in {0,1}
    
    # Let's use a simpler approach: try ranks from 48 down to 40 with structured random search
    # The structure: each column should have non-zero entries in specific patterns
    
    # Try structured random search with better initialization
    for target_rank in range(48, 39, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        attempts = 0
        max_attempts = 500000
        
        while attempts < max_attempts and not found and time.time() < deadline - 15:
            attempts += 1
            if attempts % 10000 == 0 and time.time() > deadline - 15:
                break
            
            # Generate random matrices with structured patterns
            # Use entries from {-1, 0, 1} with some sparsity
            U_r = [[0]*target_rank for _ in range(nm)]
            V_r = [[0]*target_rank for _ in range(mp)]
            W_r = [[0]*target_rank for _ in range(pn)]
            
            for c in range(target_rank):
                # Random sparse with entries in {-1, 0, 1}
                for idx in range(nm):
                    if random.random() < 0.4:
                        U_r[idx][c] = random.choice([-1, 1])
                
                for idx in range(mp):
                    if random.random() < 0.4:
                        V_r[idx][c] = random.choice([-1, 1])
                
                for idx in range(pn):
                    if random.random() < 0.4:
                        W_r[idx][c] = random.choice([-1, 1])
            
            # Quick check on a subset
            ok = True
            for _ in range(50):
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                s = 0
                for c in range(target_rank):
                    s += U_r[i*m+j][c] * V_r[j*p+k][c] * W_r[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    ok = False
                    break
            if ok:
                if verify(U_r, V_r, W_r, target_rank, n, m, p):
                    found = True
                    best = {'U': U_r, 'V': V_r, 'W': W_r, 'rank': target_rank}
                    break
    
    return {'U': best['U'], 'V': best['V'], 'W': best['W']}
# EVOLVE-BLOCK-END
