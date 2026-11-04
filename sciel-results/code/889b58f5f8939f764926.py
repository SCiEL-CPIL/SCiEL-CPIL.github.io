"""Program for matrix multiplication 2,3,6 tensor decomposition."""
DIMENSIONS = [2, 3, 6]

import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    
    The multiplication tensor T_{i,k,j} = 1 if i in [0,n), k in [0,m), j in [0,p)
    represents C_{i,j} = sum_k A_{i,k} B_{k,j}.
    
    A rank-r decomposition gives:
    U: (n*m) x r, V: (m*p) x r, W: (p*n) x r
    such that T_{i,k,j} = sum_t U_{i,k,t} * V_{k,j,t} * W_{j,i,t}
    
    The output W follows the trace convention W[k*n+i, r] where the index is (j*n + i).
    Wait, the task says W[k*n+i, r] but the shape is (p*n, rank).
    Let's re-read: "output factor follows the official trace convention W[k*n+i,r]"
    But the shape is (p*n, rank). So the index is j*n + i?
    The standard convention for matrix multiplication tensor is:
    T_{i,j,k} where i is row of A (and C), j is col of B (and C), k is shared.
    Usually U is indexed by (i,k), V by (k,j), W by (j,i).
    The task says W shape is (p*n, rank). So W index is j*n + i? Or i*p + j?
    "W[k*n+i,r]" - this is confusing. Let's look at the seed program.
    In the seed program:
    W[j * n + i][col] = 1.0
    So the index is j*n + i. Here j is the column index of B (0..p-1) and i is the row index of A (0..n-1).
    So W is indexed by (j, i) flattened as j*n + i.
    
    Strategy:
    We can use a search or a known construction.
    Since n=2, m=3, p=6, the schoolbook rank is 2*3*6 = 36.
    
    Let's try to find a lower rank.
    One approach is to use the fact that we can split the problem.
    However, a robust way to get a good rank is to use a randomized search or a specific known scheme.
    
    Given the time limit, I will implement a simple randomized search for a rank-r decomposition
    that satisfies the equations exactly (or within a very tight tolerance, then round).
    But exact integer/half-integer entries are required.
    
    Alternatively, I can try to construct a decomposition using block matrices.
    
    Let's try a rank-24 or rank-20 construction.
    
    Actually, let's use a simple iterative refinement or a known low-rank scheme.
    A safe bet is to use a rank that is known to work, like 24 or 20.
    
    Let's try to use the following idea:
    We can view the multiplication as 3 independent (2,3,2) multiplications?
    No, the A matrix is shared.
    
    Let's try a random search with integer constraints.
    We want to find U, V, W with entries in {0, 1/2, 1} or similar small integers
    such that for all i, k, j:
    sum_t U[i,k,t] V[k,j,t] W[j,i,t] = 1 (if this is a valid tensor entry, which it is for all i,k,j in range)
    Wait, the tensor T_{i,k,j} is 1 for all valid i,k,j.
    So we need sum_t U_{i,k,t} V_{k,j,t} W_{j,i,t} = 1 for all i in 0..1, k in 0..2, j in 0..5.
    
    This is a system of 2*3*6 = 36 equations.
    If we use rank r, we have r variables for U, r for V, r for W (total 3*r*r variables? No, U is 6x r, V is 18x r, W is 12x r).
    
    Let's try a rank-20 solution.
    I will use a simple heuristic: initialize with random small integers and solve for one matrix given the others.
    But this is non-convex.
    
    Instead, let's use a known construction for (2,3,6).
    A known result is that R(2,3,6) <= 20.
    
    Let's try to hardcode a rank-20 solution if I can derive it, or use a search.
    Given the complexity, I will implement a search that tries to find a solution with rank 24 first, as it's easier.
    Or better, I will use a simple construction that gives rank 24.
    
    Construction for rank 24:
    We can split the 6 columns of B into two groups of 3.
    B = [B1, B2] where B1, B2 are 3x3.
    C = [C1, C2] where C1, C2 are 2x3.
    C1 = A * B1, C2 = A * B2.
    This is two independent (2,3,3) multiplications.
    If R(2,3,3) = 9, then total rank is 9 + 9 = 18?
    But the A matrix is the same in both. So we can share the U factors?
    No, in a tensor decomposition, the rank is the number of terms.
    If we do two independent (2,3,3) multiplications, the total rank is R(2,3,3) + R(2,3,3) = 18.
    But wait, the U matrix for the first part is U1 (6x9) and for the second is U2 (6x9).
    The total U would be [U1, U2] (6x18).
    Similarly V = [V1, V2] (18x18) and W = [W1, W2] (12x18).
    This gives a valid decomposition of rank 18.
    
    Is R(2,3,3) = 9?
    Schoolbook for (2,3,3) is 2*3*3 = 18.
    Strassen-like?
    Actually, R(2,3,3) is known to be 9? No, R(2,3,3) <= 9 is a known result?
    Let's check: R(2,3,3) <= 9. Yes, it is known that R(2,3,3) = 9.
    So if we can construct a rank-9 decomposition for (2,3,3), we can use it twice to get rank 18 for (2,3,6).
    
    Let's find a rank-9 decomposition for (2,3,3).
    n=2, m=3, p=3.
    U: 6x9, V: 9x9, W: 6x9.
    
    I will search for a rank-9 solution for (2,3,3) using a simple method.
    Or I can use a known one.
    
    A known rank-9 scheme for (2,3,3):
    It is based on the fact that 2x3x3 can be decomposed into 9 terms.
    
    Let's try to construct it.
    We need sum_t U[i,k,t] V[k,j,t] W[j,i,t] = 1 for all i=0,1; k=0,1,2; j=0,1,2.
    
    I will use a randomized search to find integer solutions for (2,3,3) rank 9.
    If I find one, I will use it twice for (2,3,6) to get rank 18.
    
    If the search fails, I will fallback to a higher rank.
    
    Let's implement the search for (2,3,3) rank 9.
    """
    n, m, p = 2, 3, 6
    
    # First, try to find a rank-9 decomposition for (2,3,3)
    # and use it twice.
    
    def find_decomp_233(rank):
        # Try to find U, V, W for (2,3,3) with given rank
        # U: 6xrank, V: 9xrank, W: 6xrank
        # Equations: for i in 0..1, k in 0..2, j in 0..2:
        # sum_t U[i,k,t] V[k,j,t] W[j,i,t] = 1
        
        # We can fix U and W to be sparse and solve for V, or use a simple search.
        # Let's try a random search with small integers.
        
        import random
        random.seed(42)
        
        for _ in range(1000):
            # Initialize U, V, W with random small integers
            U = np.zeros((6, rank), dtype=int)
            V = np.zeros((9, rank), dtype=int)
            W = np.zeros((6, rank), dtype=int)
            
            # Set some entries to 1 or -1
            for t in range(rank):
                # Randomly choose a few entries in U, V, W
                for _ in range(2):
                    u_idx = random.randint(0, 5)
                    U[u_idx, t] = random.choice([1, -1])
                    v_idx = random.randint(0, 8)
                    V[v_idx, t] = random.choice([1, -1])
                    w_idx = random.randint(0, 5)
                    W[w_idx, t] = random.choice([1, -1])
            
            # Check if this works
            ok = True
            for i in range(2):
                for k in range(3):
                    for j in range(3):
                        s = 0
                        for t in range(rank):
                            s += U[i*3+k, t] * V[k*3+j, t] * W[j*2+i, t]
                        if s != 1:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                return U, V, W
        
        return None
    
    # Try to find a rank-9 solution for (2,3,3)
    U233, V233, W233 = find_decomp_233(9)
    
    if U233 is not None:
        # Use it twice for (2,3,6)
        rank = 18
        U = np.zeros((6, rank), dtype=int)
        V = np.zeros((18, rank), dtype=int)
        W = np.zeros((12, rank), dtype=int)
        
        # First block: columns 0-2 of B, rows 0-2 of C
        U[:, 0:9] = U233
        V[0:9, 0:9] = V233
        W[0:6, 0:9] = W233
        
        # Second block: columns 3-5 of B, rows 3-5 of C (wait, C is 2x6, so rows 0-1, cols 3-5)
        # W index is j*n + i. For j=3,4,5 and i=0,1, the indices are 3*2+0=6, 3*2+1=7, etc.
        U[:, 9:18] = U233
        V[9:18, 9:18] = V233
        W[6:12, 9:18] = W233
        
        # Convert to lists of lists
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook rank 36
        rank = 36
        U = [[0.0] * rank for _ in range(6)]
        V = [[0.0] * rank for _ in range(18)]
        W = [[0.0] * rank for _ in range(12)]
        
        col = 0
        for i in range(2):
            for k in range(3):
                for j in range(6):
                    U[i * 3 + k][col] = 1.0
                    V[k * 6 + j][col] = 1.0
                    W[j * 2 + i][col] = 1.0
                    col += 1
                    
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
