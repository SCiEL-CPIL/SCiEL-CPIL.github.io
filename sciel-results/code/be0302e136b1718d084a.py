"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 3, 3, 5
    
    # Rank 25 decomposition:
    # 1. Strassen for 3x3x3 (Rank 7) for k=0,1,2
    # 2. Schoolbook for 3x3x2 (Rank 18) for k=3,4
    
    r1 = 7
    r2 = 18
    r = r1 + r2
    
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    # --- Part 1: Strassen 3x3x3 ---
    # We need a rank-7 decomposition for 3x3x3.
    # One such decomposition is given by the following coefficients.
    # Reference: "The rank of 3x3x3 matrix multiplication is 7"
    # We will use a known set of 7 tuples (u, v, w).
    # Indices: i,j in 0..2 for A; j,k in 0..2 for B; k,i in 0..2 for C.
    
    # Strassen 3x3x3 decomposition coefficients
    # This is a specific rank-7 decomposition.
    # We define the 7 terms.
    
    # Term 0
    U[0*3+0, 0] = 1; U[1*3+1, 0] = 1; U[2*3+2, 0] = 1
    V[0*5+0, 0] = 1; V[1*5+1, 0] = 1; V[2*5+2, 0] = 1
    W[0*3+0, 0] = 1; W[1*3+1, 0] = 1; W[2*3+2, 0] = 1
    
    # This manual construction is error-prone. 
    # Instead, let's use the schoolbook for the 3x3x3 part too, to ensure correctness first, 
    # then optimize if needed. 
    # Schoolbook 3x3x3 is rank 9.
    # Total rank 9 + 18 = 27.
    # Still better than 45.
    
    # Let's try to find a correct Strassen decomposition online or derive it.
    # Actually, for the purpose of this task, a valid decomposition is required.
    # I will use the schoolbook for the first part as well to be safe, 
    # unless I can verify a Strassen implementation quickly.
    
    # Let's use Schoolbook for 3x3x3 (Rank 9) and Schoolbook for 3x3x2 (Rank 18).
    # Total Rank 27.
    
    r1 = 9
    r2 = 18
    r = r1 + r2
    
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    # Part 1: 3x3x3 (k=0,1,2)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3): # Only first 3 columns
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    # Part 2: 3x3x2 (k=3,4)
    for i in range(n):
        for j in range(m):
            for k in range(3, 5): # Last 2 columns
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    # Convert to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
