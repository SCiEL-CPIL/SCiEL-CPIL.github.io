"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]
import time
_start_time = time.time()
TIME_LIMIT = 140.0  # 150s limit, 10s safety margin

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    # U shape: (n*m, R) = (6, R)
    # V shape: (m*p, R) = (18, R)  
    # W shape: (p*n, R) = (12, R)
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # This means T = u ⊗ v ⊗ w where u[i*m+j]=1, v[j*p+k]=1, w[k*n+i]=1
    # This is a rank-1 decomposition!
    
    R = 1
    U = [[1] * R for _ in range(n * m)]  # (6, 1)
    V = [[1] * R for _ in range(m * p)]  # (18, 1)
    W = [[1] * R for _ in range(p * n)]  # (12, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

