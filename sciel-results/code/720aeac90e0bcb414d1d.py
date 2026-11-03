import numpy as np
import time

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115

    n, m, p = 4, 4, 9

    # Try to construct rank 102 decomposition
    # Split B into B1 (4x4), B2 (4x4), B3 (4x1)
    # C1 = A B1, C2 = A B2, C3 = A B3
    
    # We need U, V, W for 4x4x4 (rank 49) and 4x4x1 (rank 4)
    
    # 4x4x1 is just dot products.
    # U_dot: 4x4 identity? No.
    # A (4x4) * B (4x1) -> C (4x1)
    # C_i = sum_j A_ij B_j
    # Tensor T_{i,j,k} where k=0.
    # T_{i,j,0} = 1.
    # U_{i,j, r}, V_{j,0, r}, W_{0,i, r}.
    # Rank 4.
    # U[r] has 1 at (i,j) where i=r, j=0? No.
    # Term r: A_{r,0} B_{0,0} C_{r,0}? No.
    # Term r: A_{r, j} B_{j, 0} C_{r, 0}.
    # We need sum_j A_{r,j} B_{j,0} = C_{r,0}.
    # So for each r, we need terms for all j.
    # This is not rank 4.
    # Rank of 4x4x1 is 4.
    # U is 4x4 (A indices), V is 4x4 (B indices), W is 4x4 (C indices).
    # T_{i,j,0} = 1.
    # Decomposition: sum_{r=0}^3 U_{i, r} V_{j, r} W_{0, i} ?
    # No.
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
    # For 4x4x1, k=0.
    # T_{i,j,0} = 1.
    # Let r=j.
    # U_{i,j,j} = 1 if i=j? No.
    # We want sum_j A_{i,j} B_{j,0}.
    # This is not a tensor decomposition of the form sum U V W.
    # Wait, matrix multiplication IS a tensor.
    # The rank of M_{n,m,1} is n*m?
    # No, rank of M_{n,m,p} is at least max(n*m, m*p, p*n).
    # For 4,4,1: max(16, 4, 4) = 16.
    # So rank is 16.
    # So 49 + 49 + 16 = 114.
    
    # Still better than 144.
    
    # I will implement rank 114.
    # 49 for 4x4x4, 49 for 4x4x4, 16 for 4x4x1.
    
    # I need the 4x4x4 factors.
    # I will use the 2x2x2 Strassen factors I found, assuming they are correct.
    # I will verify them in code.
    
    pass
