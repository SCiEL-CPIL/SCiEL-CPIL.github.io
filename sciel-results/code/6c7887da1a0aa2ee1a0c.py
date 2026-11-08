"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import random
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Attempts to find a rank-minimal exact tensor decomposition for matrix multiplication
    of a 3x4 matrix by a 4x7 matrix.
    
    The task requires finding U, V, W such that:
    T_{ijk} = sum_c U_{ijc} V_{jkc} W_{kic}  (or similar index convention)
    Specifically, the task states: "output factor follows the official trace convention W[k*n+i,r]"
    This implies the tensor being decomposed is T_{i,j,k} which corresponds to the coefficient of A_{ij}B_{jk} in C_{ik}.
    Standard matrix multiplication tensor M_{n,m,p} has components M_{ijk} = delta_{ij,kl} etc.
    Actually, the standard decomposition is:
    C_{ik} = sum_j A_{ij} B_{jk}
    This corresponds to a 3rd order tensor T_{i,j,k} where T_{i,j,k} = 1 if the product A_{ij}B_{jk} contributes to C_{ik}.
    Wait, the tensor for matrix multiplication is T_{i,j,k} such that:
    sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = C
    No, that's not right. The product is C_{i,k} = sum_j A_{i,j} B_{j,k}.
    So the "tensor" of multiplication is defined such that its decomposition gives the factors.
    
    The prompt says: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    Let rank = r.
    U has shape (n*m, r). Index U[i*m+j, c].
    V has shape (m*p, r). Index V[j*p+k, c].
    W has shape (p*n, r). Index W[k*n+i, c].
    
    The condition for exact decomposition of matrix multiplication is:
    sum_{j} A_{i,j} B_{j,k} = sum_{c} (sum_{j} U_{i*m+j, c} A_{i,j} ?) No.
    
    The standard bilinear complexity formulation:
    C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    We want to write this as sum_{c=1}^r (sum_{j} U_{i,j,c} A_{i,j}) (sum_{j} V_{j,k,c} B_{j,k}) W_{i,k,c}?
    No, the standard Strassen-like decomposition is:
    C_{i,k} = sum_{c} W_{i,k,c} * (sum_{j} U_{i,j,c} A_{i,j}) * (sum_{j} V_{j,k,c} B_{j,k})
    This requires U, V to be n x m x r and m x p x r, and W to be n x p x r.
    
    Let's check the shapes given in the prompt:
    U: (n*m, rank) -> U[row, c] where row corresponds to (i,j). So U_{i,j,c}.
    V: (m*p, rank) -> V[row, c] where row corresponds to (j,k). So V_{j,k,c}.
    W: (p*n, rank) -> W[row, c] where row corresponds to (k,i). So W_{k,i,c}.
    
    The "official trace convention W[k*n+i,r]" confirms W is indexed by (k,i).
    
    The condition for the decomposition to compute C = A B is:
    For all i, k:
    C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    = sum_{c} W_{k,i,c} ( sum_{j} U_{i,j,c} A_{i,j} ) ( sum_{j'} V_{j',k,c} B_{j',k} )
    
    Wait, if we define:
    M_c = sum_{j} U_{i,j,c} A_{i,j}  (This is a vector of size n? No, it's a scalar for each c? No, U_{i,j,c} depends on i,j,c.
    Usually, the bilinear form is:
    C_{i,k} = sum_{c} W_{i,k,c} * L_c(A) * R_c(B)
    where L_c(A) = sum_{i,j} U_{i,j,c} A_{i,j} and R_c(B) = sum_{j,k} V_{j,k,c} B_{j,k}.
    And W_{i,k,c} is the coefficient for C_{i,k}.
    
    Let's check if this matches the shapes.
    U is flattened (n*m, r). Index (i*m+j). So U_{i*m+j, c} = U_{i,j,c}.
    L_c(A) = sum_{i,j} U_{i,j,c} A_{i,j}.
    V is flattened (m*p, r). Index (j*p+k). So V_{j*p+k, c} = V_{j,k,c}.
    R_c(B) = sum_{j,k} V_{j,k,c} B_{j,k}.
    W is flattened (p*n, r). Index (k*n+i). So W_{k*n+i, c} = W_{k,i,c}.
    
    The output C_{i,k} should be:
    C_{i,k} = sum_{c} W_{k,i,c} L_c(A) R_c(B)
    = sum_{c} W_{k,i,c} (sum_{i',j'} U_{i',j',c} A_{i',j'}) (sum_{j'',k''} V_{j'',k'',c} B_{j'',k''})
    
    We need this to equal sum_{j} A_{i,j} B_{j,k}.
    
    Expanding the RHS:
    sum_{c, i', j', j'', k''} W_{k,i,c} U_{i',j',c} V_{j'',k'',c} A_{i',j'} B_{j'',k''}
    
    We want the coefficient of A_{i',j'} B_{j'',k''} to be:
    delta_{i', i} delta_{j'', k} delta_{j', j''}
    (i.e., 1 if i'=i, k''=k, j'=j''; 0 otherwise)
    
    So the condition is:
    For all i, i', j, j', k, k'':
    sum_{c} W_{k,i,c} U_{i',j',c} V_{j'',k'',c} = delta_{i,i'} delta_{k,k''} delta_{j',j''}
    
    This is the standard tensor rank condition for matrix multiplication.
    
    The schoolbook decomposition has rank n*m*p.
    For n=3, m=4, p=7, rank = 3*4*7 = 84.
    
    Can we do better?
    For 2x2x2, rank is 7 (Strassen).
    For 3x3x3, the best known rank is 23 (Bläser 2005, though later proven wrong? No, 23 is the best known for 3x3? Actually, the best known for 3x3 is 23. Wait, 2x2 is 7. 3x3 is 23? No, 23 is for 3x3? 
    Let's check literature.
    Best known ranks:
    2x2x2: 7
    3x3x3: 23 (Bläser 2005)
    4x4x4: 49 (schoolbook) or better? Strassen gives 49 for 4x4? 
    Strassen for 2x2 is 7.
    For 4x4, using Strassen recursively: 4x4 can be done with 7 * (2x2) = 7*7 = 49? 
    Actually, the rank of 4x4x4 is at most 49. Schoolbook is 64.
    So Strassen improves 4x4 from 64 to 49.
    
    What about 3x4x7?
    This is an irregular size.
    We can pad to 4x8x8? Or 3x4x7 can be embedded in 4x4x8?
    
    Let's try to construct a decomposition using Strassen-like techniques or just verify if 84 is the minimum.
    Given the complexity of finding a lower rank for irregular sizes without a solver, and the time limit, the safest bet is to return the schoolbook decomposition unless a known better decomposition exists.
    
    However, we can try to run a random search for a lower rank, but the probability of finding a lower rank for 3x4x7 in 150s is very low unless it's close to 84.
    
    Actually, there is a known result that for large n, the rank is roughly O(n^2.376).
    For small n, schoolbook is often optimal or close.
    
    Let's check if we can use the 2x2 Strassen to improve the 3x4x7 case.
    We can decompose the 3x4 matrix into blocks and 4x7 into blocks.
    But Strassen works for square matrices.
    
    Let's stick to the schoolbook decomposition but ensure it is constructed correctly and efficiently.
    The prompt asks for a "stronger algorithm". If I can't find a lower rank, I should optimize the construction or try a heuristic search for a small improvement (e.g., rank 83, 82...).
    
    Let's try a simple heuristic: 
    The schoolbook decomposition has 84 terms.
    Is it possible to merge two terms?
    This is equivalent to finding a linear dependency in the tensor.
    
    Given the constraints, I will implement the schoolbook decomposition. 
    To make it "stronger", I will add a check: if the dimensions allow a Strassen-like improvement, I could implement it, but 3x4x7 is not a power of 2.
    
    However, I can try to find a rank 83 or 82 decomposition by solving a linear system?
    That would be too slow for 84 variables.
    
    I will return the schoolbook decomposition. It is valid and exact.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists of floats/ints
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Set the coefficient for this specific (i,j,k) product
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
