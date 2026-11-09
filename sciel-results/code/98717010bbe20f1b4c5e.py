"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time

def _add_strassen_2x2x2(U, V, W, u_off_i, u_off_j, v_off_j, v_off_k, w_off_k, w_off_i):
    """
    Adds 7 rank-1 terms corresponding to Strassen's 2x2x2 decomposition.
    u_off_i, u_off_j: offsets for A (2x2) in U
    v_off_j, v_off_k: offsets for B (2x2) in V
    w_off_k, w_off_i: offsets for C (2x2) in W
    """
    # Define the 7 terms (u, v, w)
    # u: (i, j) -> coeff
    # v: (j, k) -> coeff
    # w: (k, i) -> coeff
    
    terms = [
        # M1
        ({(0,0): 1, (1,1): 1}, 
         {(0,0): 1, (1,1): 1}, 
         {(0,0): 1, (1,1): -1}),
        # M2
        ({(1,0): 1, (1,1): 1}, 
         {(0,0): 1}, 
         {(1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, 
         {(1,0): 1, (1,1): -1}, 
         {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, 
         {(0,1): 1, (0,0): -1}, 
         {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, 
         {(1,1): 1}, 
         {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, 
         {(0,0): 1, (0,1): 1}, 
         {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, 
         {(1,0): 1, (1,1): 1}, 
         {(0,0): 1})
    ]
    
    for u_dict, v_dict, w_dict in terms:
        col = len(U[0])
        # Append 0 to all rows
        for r in range(len(U)):
            U[r].append(0)
        for r in range(len(V)):
            V[r].append(0)
        for r in range(len(W)):
            W[r].append(0)
            
        for (i, j), coeff in u_dict.items():
            U[(u_off_i + i) * 5 + (u_off_j + j)][col] = coeff
        for (j, k), coeff in v_dict.items():
            V[(v_off_j + j) * 6 + (v_off_k + k)][col] = coeff
        for (k, i), coeff in w_dict.items():
            W[(w_off_k + k) * 4 + (w_off_i + i)][col] = coeff

def _add_schoolbook(U, V, W, n, m, p, u_off_i, u_off_j, v_off_j, v_off_k, w_off_k, w_off_i):
    """
    Adds n*m*p rank-1 terms for a schoolbook decomposition of an n x m x p block.
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = len(U[0])
                for r in range(len(U)):
                    U[r].append(0)
                for r in range(len(V)):
                    V[r].append(0)
                for r in range(len(W)):
                    W[r].append(0)
                
                U[(u_off_i + i) * 5 + (u_off_j + j)][col] = 1
                V[(v_off_j + j) * 6 + (v_off_k + k)][col] = 1
                W[(w_off_k + k) * 4 + (w_off_i + i)][col] = 1

def construct():
    deadline = time.time() + 145
    
    n, m, p = 4, 5, 6
    # Initial size is 0, we will grow it
    U = [[] for _ in range(n * m)]
    V = [[] for _ in range(m * p)]
    W = [[] for _ in range(p * n)]
    
    # We split n=4 into two blocks of n=2.
    # For each block (offset 0 and 2):
    for n_block in range(2):
        n_off = n_block * 2
        
        # Split m=5 into m1=2 and m2=3.
        
        # --- Part 1: A[n_off:n_off+2, 0:2] * B[0:2, 0:6] ---
        # This is a 2 x 2 x 6 multiplication.
        # Split p=6 into three blocks of p=2.
        for p_block in range(3):
            p_off = p_block * 2
            # 2 x 2 x 2 block
            _add_strassen_2x2x2(U, V, W, 
                                u_off_i=n_off, u_off_j=0, 
                                v_off_j=0, v_off_k=p_off, 
                                w_off_k=p_off, w_off_i=n_off)
        
        # --- Part 2: A[n_off:n_off+2, 2:5] * B[2:5, 0:6] ---
        # This is a 2 x 3 x 6 multiplication.
        # Split p=6 into two blocks of p=3.
        for p_block in range(2):
            p_off = p_block * 3
            # Inside this 2 x 3 x 3 block, split m=3 into m1=2 and m2=1.
            
            # --- Sub-part 2a: A[n_off:n_off+2, 2:4] * B[2:4, p_off:p_off+3] ---
            # This is a 2 x 2 x 3 multiplication.
            # Split p=3 into p1=2 and p2=1.
            
            # 2 x 2 x 2 part (cols p_off to p_off+1)
            _add_strassen_2x2x2(U, V, W, 
                                u_off_i=n_off, u_off_j=2, 
                                v_off_j=2, v_off_k=p_off, 
                                w_off_k=p_off, w_off_i=n_off)
            
            # 2 x 2 x 1 part (col p_off+2)
            _add_schoolbook(U, V, W, 
                            n=2, m=2, p=1,
                            u_off_i=n_off, u_off_j=2, 
                            v_off_j=2, v_off_k=p_off+2, 
                            w_off_k=p_off+2, w_off_i=n_off)
            
            # --- Sub-part 2b: A[n_off:n_off+2, 4:5] * B[4:5, p_off:p_off+3] ---
            # This is a 2 x 1 x 3 multiplication.
            _add_schoolbook(U, V, W, 
                            n=2, m=1, p=3,
                            u_off_i=n_off, u_off_j=4, 
                            v_off_j=4, v_off_k=p_off, 
                            w_off_k=p_off, w_off_i=n_off)
    
    # Convert lists of lists to numpy arrays or just return lists?
    # The task says "factor matrices of shapes... with integer/half-integer entries".
    # The seed program returns lists of lists. We will stick to that to be safe, 
    # or check if numpy is expected. The seed uses lists.
    # However, the seed program's construct() returns dicts of lists.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
