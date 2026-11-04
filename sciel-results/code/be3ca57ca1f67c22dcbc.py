"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    total = n*m*p  # 60
    
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j][c]*V[j*p+k][c]*W[k*n+i][c] for c in range(rank))
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Schoolbook
    rank0 = total
    U0 = [[0]*rank0 for _ in range(nm)]
    V0 = [[0]*rank0 for _ in range(mp)]
    W0 = [[0]*rank0 for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j][col] = 1
                V0[j*p+k][col] = 1
                W0[k*n+i][col] = 1
                col += 1
    
    # Try rank 59: use 59 terms, 58 from schoolbook + 1 combined
    # The combined term must compensate for the missing schoolbook term
    # Missing term (i0,j0,k0): need U[i0*m+j0,58]*V[j0*p+k0,58]*W[k0*n+i0,58] = 1
    # But this term also contributes to all (i,j,k) where U[i*m+j,58]*V[j*p+k,58]*W[k*n+i,58] != 0
    # For the 58 schoolbook terms, each (i,j,k) has exactly one term contributing 1.
    # If we remove term (i0,j0,k0), then (i0,j0,k0) has sum 0, and all others have sum 1.
    # The new term must add 1 to (i0,j0,k0) and 0 to all others.
    # So we need: for all (i,j,k) != (i0,j0,k0): U[i*m+j,58]*V[j*p+k,58]*W[k*n+i,58] = 0
    # And: U[i0*m+j0,58]*V[j0*p+k0,58]*W[k0*n+i0,58] = 1
    # This means the new term must be "orthogonal" to all other (i,j,k) in a multiplicative sense.
    # This is only possible if the new term is exactly the schoolbook term, which doesn't help.
    
    # So rank 59 with this structure won't work. We need a more global restructuring.
    
    # For now, return schoolbook. A proper search would require more sophisticated methods.
    return {"U": U0, "V": V0, "W": W0}
# EVOLVE-BLOCK-END

