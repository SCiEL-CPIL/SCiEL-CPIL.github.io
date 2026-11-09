"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    def time_left():
        return deadline - time.time()
    
    # Schoolbook solution (always valid)
    def make_schoolbook():
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
    
    best_result = make_schoolbook()
    best_rank = n * m * p
    
    # Fast verification using numpy
    def verify_np(U, V, W):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = U_np[i * m + j]
                    vj = V_np[j * p + k]
                    wk = W_np[k * n + i]
                    s = np.dot(ui * vj, wk)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Strassen-style block decomposition
    # Split 3x4 * 4x6 into blocks
    # A (3x4) split into 2x2 blocks of (3/2, 2/2) x (4/2, 4/2)
    # Actually, let's use a simpler approach:
    # Split the middle dimension 4 into 2+2
    # A = [A1 | A2] where A1 is 3x2, A2 is 3x2
    # B = [B1; B2] where B1 is 2x6, B2 is 2x6
    # C = A1*B1 + A2*B2
    # Each product is 3x2 * 2x6, which has schoolbook rank 3*2*6 = 36
    # Total rank: 72 (same as full schoolbook)
    
    # But we can try to use Strassen-like tricks on the 3x2 * 2x6 products
    # For 3x2 * 2x6, we can potentially reduce the rank
    
    # Let's try a different approach: use known low-rank decompositions
    # For 2x2 matrix multiplication, Strassen uses rank 7 instead of 8
    # For 3x3, the best known is rank 23 (Laser method) vs schoolbook 27
    
    # For 3x4 * 4x6, let's try to exploit structure
    
    # Approach 1: Try to find a decomposition using the fact that
    # the tensor has symmetries
    
    # Approach 2: Use random search with structured initialization
    
    # Let's try a greedy construction approach
    def greedy_construct(target_rank, max_iter=1000):
        """Try to construct a valid decomposition of given rank greedily."""
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        # Start with schoolbook pattern but try to merge terms
        # For each (i,j,k), we need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        
        # Try to assign values to minimize rank
        # Use a simple heuristic: for each rank slot, try to cover multiple (i,j,k) terms
        
        for r in range(target_rank):
            if time_left() < 5:
                break
            
            # Choose random (i,j,k) to focus on
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            
            # Set U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
            U[i*m+j][r] = 1
            V[j*p+k][r] = 1
            W[k*n+i][r] = 1
            
            # Try to extend this to cover other (i',j',k') where possible
            # For this to work, we need U[i'*m+j',r] * V[j'*p+k',r] * W[k'*n+i',r] 
            # to contribute to the sum for (i',j',k')
            
            # This is complex; let's just do random search instead
        
        return U, V, W
    
    # Random search with better initialization
    def random_search(target_rank, num_attempts=100):
        """Random search for a valid decomposition of given rank."""
        for attempt in range(num_attempts):
            if time_left() < 10:
                break
            
            # Initialize with sparse random values
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Fill with random values from a small set
            values = [-2, -1, 0, 1, 2]
            for r in range(target_rank):
                # Randomly choose which (i,j), (j,k), (k,i) to activate
                num_activations = random.randint(1, 3)
                
                # Activate random (i,j) pairs
                for _ in range(num_activations):
                    i = random.randint(0, n-1)
                    j = random.randint(0, m-1)
                    U[i*m+j][r] = random.choice(values)
                
                # Activate random (j,k) pairs
                for _ in range(num_activations):
                    j = random.randint(0, m-1)
                    k = random.randint(0, p-1)
                    V[j*p+k][r] = random.choice(values)
                
                # Activate random (k,i) pairs
                for _ in range(num_activations):
                    k = random.randint(0, p-1)
                    i = random.randint(0, n-1)
                    W[k*n+i][r] = random.choice(values)
            
            if verify_np(U, V, W):
                return U, V, W
        
        return None
    
    # Try to reduce rank from schoolbook
    # Start with schoolbook and try to merge columns
    
    def try_merge_schoolbook():
        """Try to merge schoolbook columns to reduce rank."""
        U = [row[:] for row in best_result["U"]]
        V = [row[:] for row in best_result["V"]]
        W = [row[:] for row in best_result["W"]]
        rank = len(U[0])
        
        # Try to find pairs of columns that can be merged
        # Two columns c1, c2 can be merged if there exists a new column c
        # such that the tensor equation still holds
        
        # This is equivalent to finding a linear dependence
        # For now, just try random perturbations
        
        for _ in range(50):
            if time_left() < 10:
                break
            
            # Pick two random columns
            c1 = random.randint(0, rank-1)
            c2 = random.randint(0, rank-1)
            if c1 == c2:
                continue
            
            # Try to merge by setting one to zero and adjusting the other
            # This is heuristic and may not work
            
            # Instead, try a different approach:
            # Look for columns that have support on disjoint sets of (i,j), (j,k), (k,i)
            
            # Get supports
            support_u1 = set()
            support_u2 = set()
            for i in range(n*m):
                if U[i][c1] != 0:
                    support_u1.add(i)
                if U[i][c2] != 0:
                    support_u2.add(i)
            
            if support_u1 & support_u2:  # Overlapping supports, can't simply merge
                continue
            
            # Try to merge
            U_new = [row[:] for row in U]
            V_new = [row[:] for row in V]
            W_new = [row[:] for row in W]
            
            # Add column c2 to column c1
            for i in range(n*m):
                U_new[i][c1] += U_new[i][c2]
            for i in range(m*p):
                V_new[i][c1] += V_new[i][c2]
            for i in range(p*n):
                W_new[i][c1] += W_new[i][c2]
            
            # Remove column c2
            for i in range(n*m):
                U_new[i].pop(c2)
            for i in range(m*p):
                V_new[i].pop(c2)
            for i in range(p*n):
                W_new[i].pop(c2)
            
            if verify_np(U_new, V_new, W_new):
                return U_new, V_new, W_new
        
        return None
    
    # Try various target ranks
    for target_rank in range(best_rank - 1, max(20, best_rank - 30), -1):
        if time_left() < 20:
            break
        
        # Try random search
        result = random_search(target_rank, num_attempts=20)
        if result is not None:
            U, V, W = result
            if verify_np(U, V, W):
                best_rank = target_rank
                best_result = {"U": U, "V": V, "W": W}
                break
        
        # Try merging from schoolbook if we haven't found anything yet
        if best_rank == n * m * p and target_rank == best_rank - 1:
            merged = try_merge_schoolbook()
            if merged is not None:
                U, V, W = merged
                if verify_np(U, V, W):
                    best_rank = target_rank
                    best_result = {"U": U, "V": V, "W": W}
                    break
    
    # Final verification
    if not verify_np(best_result["U"], best_result["V"], best_result["W"]):
        best_result = make_schoolbook()
    
    return best_result
# EVOLVE-BLOCK-END
