"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    Uses a structured search combining block-partitioning with integer/rational coefficients.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin before 120s kill, and some for cleanup
    
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 48
    
    # Try to find a better decomposition
    # Strategy: Use a combination of small Strassen-like decompositions on sub-blocks
    # and search for integer coefficient combinations.
    
    # First, let's try a structured approach:
    # Split the 2x4 * 4x6 into blocks.
    # A is 2x4, B is 4x6, C is 2x6.
    # Split A into A_left (2x2) and A_right (2x2).
    # Split B into B_top (4x3) and B_bottom (4x3).
    # C = A_left * B_top + A_right * B_bottom
    
    # For each 2x2 * 2x3 multiplication, we can use a Strassen-like decomposition.
    # 2x2 * 2x3: rank 6 (schoolbook) or less?
    # Actually, 2x2 * 2x2 has rank 7 (Strassen). 2x2 * 2x3 can be seen as two 2x2 * 2x2.
    # But we can do better by sharing some terms.
    
    # Let's try a direct search for small integer decompositions.
    # We'll use a randomized local search / simulated annealing approach.
    
    def generate_random_decomp(rank, trial=0):
        """Generate a random integer decomposition of given rank."""
        np.random.seed(trial)
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        
        # Initialize with random small integers
        for c in range(rank):
            # Randomly select which entries are non-zero
            u_mask = np.random.random(n*m) < 0.3
            v_mask = np.random.random(m*p) < 0.3
            w_mask = np.random.random(p*n) < 0.3
            
            U[u_mask, c] = np.random.choice([-2, -1, 1, 2], size=np.sum(u_mask))
            V[v_mask, c] = np.random.choice([-2, -1, 1, 2], size=np.sum(v_mask))
            W[w_mask, c] = np.random.choice([-2, -1, 1, 2], size=np.sum(w_mask))
        
        return U, V, W
    
    def compute_tensor(U, V, W):
        """Compute the tensor from the decomposition."""
        n, m, p = 2, 4, 6
        tensor = np.zeros((n*m*p,))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    tensor[i*m*p + j*p + k] = val
        return tensor
    
    def target_tensor():
        """The target tensor for matrix multiplication: T[i,j,k] = 1 if i,j,k form a valid product term."""
        n, m, p = 2, 4, 6
        tensor = np.zeros((n*m*p,))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # The tensor element is 1 for the standard matrix multiplication
                    # T[i*m*p + j*p + k] = 1 (since C_ik = sum_j A_ij * B_jk)
                    tensor[i*m*p + j*p + k] = 1.0
        return tensor
    
    target = target_tensor()
    
    # Try structured decomposition first
    # Use the fact that 2x4 * 4x6 can be decomposed using block matrices.
    # Let's try rank 32, 24, etc.
    
    best_U, best_V, best_W = None, None, None
    best_rank = rank_schoolbook
    
    # Start with schoolbook
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    # Try to improve with random search
    for trial_rank in [32, 28, 24, 20]:
        if time.time() > deadline:
            break
        
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            U, V, W = generate_random_decomp(trial_rank, attempt)
            tensor = compute_tensor(U, V, W)
            error = np.max(np.abs(tensor - target))
            
            if error < 1e-9:
                best_U, best_V, best_W = U, V, W
                best_rank = trial_rank
                break
        else:
            continue
        break
    
    # Convert to lists
    U_list = [[int(round(x)) if abs(x) < 0.5 else (int(round(x*2))/2 if abs(x*2 - round(x*2)) < 0.01 else float(x)) 
              for x in row] for row in best_U]
    V_list = [[int(round(x)) if abs(x) < 0.5 else (int(round(x*2))/2 if abs(x*2 - round(x*2)) < 0.01 else float(x)) 
              for x in row] for row in best_V]
    W_list = [[int(round(x)) if abs(x) < 0.5 else (int(round(x*2))/2 if abs(x*2 - round(x*2)) < 0.01 else float(x)) 
              for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
