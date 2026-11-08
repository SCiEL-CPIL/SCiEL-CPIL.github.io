"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    best = schoolbook()
    
    # Try to find a lower rank decomposition using known structures
    # For M_{2,m,p}, try rank = 2*m*p - m - p + 2 or similar
    # Let's try a few candidate ranks and see if we can construct a valid decomposition
    
    # Known: M_{2,2,2} has rank 7 (Strassen)
    # Try to build M_{2,5,6} from Strassen blocks
    
    # Alternative: use a random search with small rank
    # The tensor equation: for all i,j,k, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k'*n+i',r] = delta conditions
    
    # Actually, the condition is that the sum over r of the outer product equals the tensor
    # Let's verify a decomposition by checking the tensor equation
    
    def verify(U, V, W):
        """Check if U,V,W form a valid decomposition."""
        rank = len(U[0]) if U else 0
        # Build the tensor from U,V,W
        # T[i*m+j, j*p+k, k'*n+i'] should match the multiplication tensor
        # The multiplication tensor has T[i,j,k] = 1 for the coefficient
        # Actually, we need: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
        # Wait, that's not right. Let me think...
        
        # The tensor M_{n,m,p} = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
        # So we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
        # And 0 otherwise? No, the tensor only has 1s at (i*m+j, j*p+k, k*n+i) for valid i,j,k
        
        # Actually, the tensor is: T[a,b,c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' 
        # with the same j=j' and the relation that this corresponds to the multiplication
        # I think T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise
        
        # So we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
        # And for any (a,b,c) not of this form, the sum should be 0, but since our indices
        # are always of the form (i*m+j, j'*p+k', k''*n+i''), we just need the above
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-10:
                        return False
        return True
    
    # Try Strassen-based decomposition for M_{2,2,2} blocks
    # For M_{2,5,6}, we can partition the 5 columns into 2+2+1 and 6 rows into 2+2+2
    # But this is complex. Let's try a simpler approach.
    
    # Try rank 20, 25, 30, etc. with random initialization and check
    # Given time constraints, just return schoolbook if we can't find better quickly
    
    if time.time() > deadline - 10:
        return best
    
    # Try a few specific ranks
    for target_rank in [20, 25, 30, 35, 40, 45, 50, 55]:
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Convert to lists for verification
            U_list = [[U[i, r] for r in range(target_rank)] for i in range(n * m)]
            V_list = [[V[i, r] for r in range(target_rank)] for i in range(m * p)]
            W_list = [[W[i, r] for r in range(target_rank)] for i in range(p * n)]
            
            if verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        if time.time() > deadline - 10:
            break
    
    return best
# EVOLVE-BLOCK-END
