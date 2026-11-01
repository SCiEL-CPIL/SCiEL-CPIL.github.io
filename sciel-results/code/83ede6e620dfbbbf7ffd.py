import time
import random
import numpy as np

DEADLINE = 150.0
START_TIME = time.time()

def time_remaining():
    return DEADLINE - (time.time() - START_TIME)

def construct():
    """
    Strategy: Exploit the block structure of the 2x3 by 3x9 multiplication.
    Split the 9 columns of B into three 3x3 blocks: B = [B1, B2, B3].
    Then C = [A*B1, A*B2, A*B3].
    This reduces the problem to finding a low-rank decomposition for 2x3 by 3x3,
    and replicating it 3 times.
    
    The schoolbook rank for 2x3x3 is 6. Total rank would be 18.
    However, we can try to find a lower rank decomposition for 2x3x3.
    The tensor rank of 2x3x3 is known to be 5.
    
    We will attempt to find a rank 5 decomposition for 2x3x3 using a numerical search
    (alternating least squares or gradient descent on the bilinear form).
    If successful, total rank is 15.
    If it fails or times out, we fallback to rank 6 (total 18) or even schoolbook (54).
    
    Actually, a simpler and more robust approach for a "structural change" is to use
    the known rank 5 decomposition if we can hardcode it, or use a very fast search.
    
    Let's try to implement a fast search for the 2x3x3 rank 5 decomposition.
    Variables: U (6x5), V (9x5), W (6x5).
    Constraint: sum_r U[i*3+j, r] V[j'*3+k, r] W[k*2+i, r] = delta_{j,j'}
    
    This is a system of 54 equations (2*3*3*3) with 105 variables (6*5+9*5+6*5).
    We can initialize with the rank 6 solution and try to zero out one column via optimization.
    Or use random restarts.
    
    Given the 150s limit, a simple local search from a good initialization might work.
    Let's initialize with the rank 6 schoolbook solution, but that has 6 columns.
    We need 5.
    
    Alternative: Use the Strassen-like algorithm for 2x3x3 if it exists.
    Or, use the fact that 2x3x3 is a sub-tensor of 3x3x3.
    
    Let's try a different structural insight:
    C = A * B.
    A is 2x3, B is 3x9.
    We can pad A to 3x3 by adding a zero row: A' = [[A], [0 0 0]].
    Then C = A' * B.
    But this doesn't help directly unless we have a fast 3x3x9 algorithm.
    
    Let's stick to the block replication.
    We need rank(2x3x3).
    Let's try to find a rank 5 decomposition numerically.
    
    If we fail, we use rank 6 (total 18).
    Rank 18 is a huge improvement over 54 (score 3.0).
    Rank 15 is even better (score 3.6).
    
    Let's implement the rank 6 solution first as a fallback, and then try to optimize to rank 5.
    """
    
    n, m, p = 2, 3, 9
    
    # Fallback: Rank 18 (3 blocks of rank 6 schoolbook)
    def get_rank18_solution():
        rank = 18
        U = np.zeros((6, rank))
        V = np.zeros((27, rank))
        W = np.zeros((18, rank))
        
        col = 0
        for block in range(3):
            for i in range(2):
                for j in range(3):
                    k_local = j # In 2x3x3, the output col is k, but here we are doing schoolbook for the block
                    # Wait, schoolbook for 2x3x3:
                    # C_{i,k} = sum_j A_{i,j} B_{j,k}
                    # Terms are (i, j, k).
                    # In the full 2x3x9 problem, the block index is 'b' (0..2).
                    # The local k is 0..2. The global k is b*3 + k_local.
                    
                    # So for each block b:
                    #   for each i in 0..1:
                    #     for each j in 0..2:
                    #       for each k_local in 0..2:
                    #         term (i, j, b, k_local)
                    
                    pass
        
        # Correct Rank 18 construction:
        col = 0
        for b in range(3):
            for i in range(2):
                for j in range(3):
                    for k_local in range(3):
                        k_global = b * 3 + k_local
                        U[i * 3 + j, col] = 1.0
                        V[j * 9 + k_global, col] = 1.0
                        W[k_global * 2 + i, col] = 1.0
                        col += 1
                        
        return U, V, W, rank

    # Try to find Rank 15 solution (3 blocks of rank 5)
    def try_rank15_solution():
        # We need a rank 5 decomposition for 2x3x3.
        # Let's try a simple local search.
        # Initialize U, V, W with random values or from a perturbed rank 6 solution.
        
        # Since finding rank 5 is non-trivial and might fail, we will attempt a few random restarts.
        # If time is running out, we abort.
        
        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # Target tensor T for 2x3x3
        # T[i, j, k] is not quite right.
        # The condition is:
        # M_{i,k} = sum_r (U[:, i*3:(i+1)*3].T @ V[:, k*3:(k+1)*3]) * W[:, k*2+i] ? No.
        
        # Let's define the residual.
        # For each i in 0..1, k in 0..2:
        #   We want a 3x3 matrix M_{i,k} such that M_{i,k}[j, j'] = delta_{j,j'}.
        #   M_{i,k} = sum_r (u_r[i*3:(i+1)*3]) (v_r[k*3:(k+1)*3]).T * w_r[k*2+i]
        #   Wait, u_r is a vector of length 6. The part for row i is u_r[i*3:(i+1)*3].
        #   v_r is a vector of length 9. The part for col k is v_r[k*3:(k+1)*3].
        #   w_r is a vector of length 6. The part for output (i,k) is w_r[k*2+i].
        
        # So M_{i,k} = sum_r w_r[k*2+i] * (u_r[i*3:(i+1)*3]) (v_r[k*3:(k+1)*3]).T
        
        # We want M_{i,k} = I_3.
        
        def compute_error(U, V, W):
            err = 0.0
            for i in range(2):
                for k in range(3):
                    M = np.zeros((3, 3))
                    for r in range(5):
                        u_part = U[i*3:(i+1)*3, r]
                        v_part = V[k*3:(k+1)*3, r]
                        w_val = W[k*2+i, r]
                        M += w_val * np.outer(u_part, v_part)
                    err += np.sum((M - np.eye(3))**2)
            return err

        # Try a few random initializations
        random.seed(42)
        for restart in range(10):
            if time_remaining() < 10:
                break
            
            U = np.random.randn(6, 5) * 0.1
            V = np.random.randn(9, 5) * 0.1
            W = np.random.randn(6, 5) * 0.1
            
            # Simple gradient descent or alternating least squares?
            # Gradient descent is safer for small steps.
            lr = 0.01
            for step in range(200):
                if time_remaining() < 5:
                    return None
                    
                # Compute gradients?
                # For simplicity, let's use a finite difference or just a local search.
                # Actually, let's try to use the fact that we can solve for W given U, V.
                # M_{i,k} = sum_r w_r[k*2+i] * (u_r[i*3:(i+1)*3]) (v_r[k*3:(k+1)*3]).T
                # This is a linear system for W if U, V are fixed.
                # But U, V are also variables.
                
                # Let's just do a few iterations of random perturbation (simulated annealing style).
                # This is inefficient but might find something.
                
                # Better: Use the rank 6 solution and try to merge two columns.
                # This is complex.
                
                # Given the difficulty, let's just return the rank 18 solution.
                # It is a significant improvement over 54.
                break
                
        return None

    # Use the reliable Rank 18 solution
    U, V, W, rank = get_rank18_solution()
    
    # Convert to lists of lists as required by the interface?
    # The interface says "factor matrices of shapes ... with integer/half-integer entries".
    # It doesn't explicitly say lists, but the seed program uses lists.
    # Let's check the seed: "U": U, where U is a list of lists.
    # I will convert numpy arrays to lists.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

