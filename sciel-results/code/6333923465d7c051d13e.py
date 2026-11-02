"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is correct."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _build_3x3x2_strassen():
    """
    Build a rank-17 decomposition for 3x3x2 matrix multiplication.
    A(3x3) * B(3x2) -> C(3x2)
    
    Index conventions:
    U[i*m+j] where m=3: index = 3i+j
    V[j*p+k] where p=2: index = 2j+k  
    W[k*n+i] where n=3: index = 3k+i
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    """
    strassen_terms = [
        # M0: (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) -> C[0,0]+=1, C[1,1]+=1
        ([1,0,0, 0,1,0, 0,0,0], [1,0, 0,1, 0,0], [1,0, 0,1, 0,0]),
        # M1: (A[1,0]+A[1,1])*B[0,0] -> C[1,0]+=1
        ([0,0,0, 1,1,0, 0,0,0], [1,0, 0,0, 0,0], [0,0, 1,0, 0,0]),
        # M2: A[0,0]*(B[1,0]-B[1,1]) -> C[0,1]+=1
        ([1,0,0, 0,0,0, 0,0,0], [0,0, 1,-1, 0,0], [0,1, 0,0, 0,0]),
        # M3: A[1,1]*(B[0,1]-B[0,0]) -> C[0,0]+=-1, C[1,1]+=-1
        ([0,0,0, 0,1,0, 0,0,0], [-1,1, 0,0, 0,0], [-1,0, 0,-1, 0,0]),
        # M4: (A[0,0]+A[0,1])*B[1,1] -> C[0,1]+=1
        ([1,1,0, 0,0,0, 0,0,0], [0,0, 0,1, 0,0], [0,1, 0,0, 0,0]),
        # M5: (A[1,0]-A[0,0])*(B[0,0]+B[0,1]) -> C[1,0]+=1
        ([-1,0,0, 1,0,0, 0,0,0], [1,1, 0,0, 0,0], [0,0, 1,0, 0,0]),
        # M6: (A[0,1]-A[1,1])*(B[1,0]+B[1,1]) -> C[0,0]+=1
        ([0,1,0, 0,-1,0, 0,0,0], [0,0, 1,1, 0,0], [1,0, 0,0, 0,0]),
    ]
    
    cross_terms = [
        # A[2,0]*B[0,0] -> C[2,0]
        ([0,0,0, 0,0,0, 1,0,0], [1,0, 0,0, 0,0], [0,0, 0,0, 1,0]),
        # A[2,0]*B[0,1] -> C[2,1]
        ([0,0,0, 0,0,0, 1,0,0], [0,1, 0,0, 0,0], [0,0, 0,0, 0,1]),
        # A[2,1]*B[1,0] -> C[2,0]
        ([0,0,0, 0,0,0, 0,1,0], [0,0, 1,0, 0,0], [0,0, 0,0, 1,0]),
        # A[2,1]*B[1,1] -> C[2,1]
        ([0,0,0, 0,0,0, 0,1,0], [0,0, 0,1, 0,0], [0,0, 0,0, 0,1]),
        # A[2,2]*B[2,0] -> C[2,0]
        ([0,0,0, 0,0,0, 0,0,1], [0,0, 0,0, 1,0], [0,0, 0,0, 1,0]),
        # A[2,2]*B[2,1] -> C[2,1]
        ([0,0,0, 0,0,0, 0,0,1], [0,0, 0,0, 0,1], [0,0, 0,0, 0,1]),
        # A[0,2]*B[2,0] -> C[0,0]
        ([0,0,1, 0,0,0, 0,0,0], [0,0, 0,0, 1,0], [1,0, 0,0, 0,0]),
        # A[0,2]*B[2,1] -> C[0,1]
        ([0,0,1, 0,0,0, 0,0,0], [0,0, 0,0, 0,1], [0,1, 0,0, 0,0]),
        # A[1,2]*B[2,0] -> C[1,0]
        ([0,0,0, 0,0,1, 0,0,0], [0,0, 0,0, 1,0], [0,0, 1,0, 0,0]),
        # A[1,2]*B[2,1] -> C[1,1]
        ([0,0,0, 0,0,1, 0,0,0], [0,0, 0,0, 0,1], [0,0, 0,1, 0,0]),
    ]
    
    return strassen_terms + cross_terms

def _build_3x3x6_from_block(block_terms, n=3, m=3, p=6, block_p=2):
    """
    Build full 3x3x6 decomposition from 3x3x2 block terms.
    Split B and C into 3 blocks of 2 columns each.
    """
    r_block = len(block_terms)
    total_rank = 3 * r_block
    
    U_full = [[0]*total_rank for _ in range(n*m)]
    V_full = [[0]*total_rank for _ in range(m*p)]
    W_full = [[0]*total_rank for _ in range(p*n)]
    
    for t in range(3):
        for c in range(r_block):
            col = t * r_block + c
            u, v, w = block_terms[c]
            
            for idx in range(n*m):
                U_full[idx][col] = u[idx]
            
            for j in range(m):
                for k in range(block_p):
                    local_v_idx = j * block_p + k
                    global_v_idx = j * p + (t * block_p + k)
                    V_full[global_v_idx][col] = v[local_v_idx]
            
            for i in range(n):
                for k in range(block_p):
                    local_w_idx = block_p * i + k
                    global_w_idx = (t * block_p + k) * n + i
                    W_full[global_w_idx][col] = w[local_w_idx]
    
    return U_full, V_full, W_full

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Deadline guard: return best result before 150s limit
    deadline = start_time + 140.0
    
    # Try the Strassen-based rank-51 decomposition
    if time.time() < deadline:
        block_terms = _build_3x3x2_strassen()
        U_block, V_block, W_block = _build_3x3x6_from_block(block_terms, n, m, p)
        
        if _verify_decomposition(U_block, V_block, W_block, n, m, p):
            return {"U": U_block, "V": V_block, "W": W_block}
    
    # Fallback: schoolbook rank 54
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
