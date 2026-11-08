"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Tensor dimensions: U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    dim_u = n * m  # 6
    dim_v = m * p  # 9
    dim_w = p * n  # 6
    
    # Build the target tensor T[i,j,k] = 1 for matrix multiplication
    # T_{(i,j),(j,k),(k,i)} where i in [0,n), j in [0,m), k in [0,p)
    # The multiplication: (AB)_{i,k} = sum_j A_{i,j} * B_{j,k}
    # So tensor T has T[(i,j),(j,k),(k,i)] = 1
    
    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        # Check shapes
        if len(U) != dim_u or len(V) != dim_v or len(W) != dim_w:
            return False
        for c in range(rank):
            if len(U[c]) != rank or len(V[c]) != rank or len(W[c]) != rank:
                return False
        
        # Reconstruct and check
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # For each (i,j,k), check sum_c U[(i,j),c] * V[(j,k),c] * W[(k,i),c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Faster verification using numpy."""
        rank = len(U[0]) if U else 0
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # T[(i,j),(j,k),(k,i)] should equal 1
        # Reshape: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # For each c, we have outer product contribution
        # Sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # Build full tensor and check
        T = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U_arr[:, c].reshape(n, m)
            v_c = V_arr[:, c].reshape(m, p)
            w_c = W_arr[:, c].reshape(p, n)
            # Contribution: u_c[i,j] * v_c[j,k] * w_c[k,i] summed over k for each (i,j)
            # Actually we need: for each (i,j,k): u_c[i,j] * v_c[j,k] * w_c[k,i]
            # Sum over c of these should be 1 for all (i,j,k)
            # Let's compute the full 3D tensor
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
        
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Strategy: randomized search with local refinement
    
    def random_decomposition(rank, seed=None):
        """Generate a random decomposition with rank factors."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        # Use small integer/half-integer values
        values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        
        U = [[rng.choice(values) for _ in range(rank)] for _ in range(dim_u)]
        V = [[rng.choice(values) for _ in range(rank)] for _ in range(dim_v)]
        W = [[rng.choice(values) for _ in range(rank)] for _ in range(dim_w)]
        
        return U, V, W
    
    def refine_decomposition(U, V, W, max_iters=10000, seed=None):
        """Try to refine a decomposition to satisfy the constraints exactly."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        rank = len(U[0])
        
        values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        
        for iteration in range(max_iters):
            # Check if we're done
            if verify_fast(U, V, W):
                return U, V, W, True
            
            # Pick a random factor and a random entry to change
            factor = rng.choice(['U', 'V', 'W'])
            if factor == 'U':
                idx = rng.randrange(dim_u)
                col = rng.randrange(rank)
                old_val = U[idx][col]
            elif factor == 'V':
                idx = rng.randrange(dim_v)
                col = rng.randrange(rank)
                old_val = V[idx][col]
            else:
                idx = rng.randrange(dim_w)
                col = rng.randrange(rank)
                old_val = W[idx][col]
            
            # Try a different value
            new_val = rng.choice(values)
            if new_val == old_val:
                continue
            
            # Apply change
            if factor == 'U':
                U[idx][col] = new_val
            elif factor == 'V':
                V[idx][col] = new_val
            else:
                W[idx][col] = new_val
            
            # Check if improved
            if verify_fast(U, V, W):
                return U, V, W, True
            
            # Keep the change if it doesn't make things worse (simple heuristic)
            # For now, just keep it (random walk)
            # Could add backtracking here
        
        return U, V, W, False
    
    # Try ranks from low to high
    # Lower bound: at least max(n*p, m*p, n*m) = max(6,9,6) = 9
    # But realistically for exact rank, we need more
    # Let's try from 9 upward
    
    for rank in range(9, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 50 if rank < 12 else 20
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            U, V, W = random_decomposition(rank, seed=rank * 1000 + attempt)
            
            # Try to refine
            U_ref, V_ref, W_ref, success = refine_decomposition(
                U, V, W, max_iters=2000, seed=rank * 1000 + attempt
            )
            
            if success:
                best_result = {"U": U_ref, "V": V_ref, "W": W_ref}
                best_rank = rank
                break  # Found a solution at this rank, no need to try more at this rank
        
        if best_rank == rank:
            break  # Found solution at this rank, we're done
    
    return best_result
# EVOLVE-BLOCK-END
