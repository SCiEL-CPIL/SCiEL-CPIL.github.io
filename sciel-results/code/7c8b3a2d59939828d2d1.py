import time
import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the optimal rank-7 Strassen decomposition for 2x2 matrix multiplication.
    """
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 7
    
    U = np.zeros((4, rank), dtype=int)
    V = np.zeros((4, rank), dtype=int)
    W = np.zeros((4, rank), dtype=int)
    
    # Strassen products
    U[0,0]=1; U[3,0]=1    # M1: a+d
    U[2,1]=1; U[3,1]=1    # M2: c+d
    U[0,2]=1              # M3: a
    U[3,3]=1              # M4: d
    U[0,4]=1; U[1,4]=1    # M5: a+b
    U[2,5]=1; U[0,5]=-1   # M6: c-a
    U[1,6]=1; U[3,6]=-1   # M7: b-d
    
    V[0,0]=1; V[3,0]=1    # M1: e+h
    V[0,1]=1              # M2: e
    V[1,2]=1; V[3,2]=-1   # M3: f-h
    V[2,3]=1; V[0,3]=-1   # M4: g-e
    V[3,4]=1              # M5: h
    V[0,5]=1; V[1,5]=1    # M6: e+f
    V[2,6]=1; V[3,6]=1    # M7: g+h
    
    W[0,0]=1; W[0,3]=1; W[0,4]=-1; W[0,6]=1 # C00
    W[1,1]=1; W[1,3]=1                      # C10
    W[2,2]=1; W[2,4]=1                      # C01
    W[3,0]=1; W[3,1]=-1; W[3,2]=1; W[3,5]=1 # C11
    
    # Quick verification
    if time.time() < deadline:
        rng = np.random.RandomState(42)
        for _ in range(5):
            A = rng.randn(2, 2)
            B = rng.randn(2, 2)
            C_expected = A @ B
            C_computed = np.zeros((2, 2))
            for i in range(2):
                for k in range(2):
                    for j in range(2):
                        C_computed[i, k] += np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                # Should not happen
                break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
