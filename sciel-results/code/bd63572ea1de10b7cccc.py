import numpy as np
import time
from itertools import product, combinations
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _verify_mod(U, V, W, n, m, p, mod):
    """Fast verification using modular arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += (U[i*m+j][c] % mod) * (V[j*p+k][c] % mod) * (W[k*n+i][c] % mod)
                if s % mod != 1 % mod:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook rank n*m*p decomposition."""
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

def _strassen_2x2x2():
    """Rank-7 Strassen decomposition for 2x2 * 2x2."""
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[2][5] = 1
    
    # M7 = (A01-A11)(B01+B11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return {"U": U, "V": V, "W": W}

def _construct_3x3x4_strassen_hybrid():
    """
    Construct 3x3 * 3x4 decomposition using Strassen for 2x2 block.
    
    A (3x3) = [[A11 (2x2), A12 (2x1)],
               [A21 (1x2), A22 (1x1)]]
    B (3x4) = [[B11 (2x2), B12 (2x2)],
               [B21 (1x2), B22 (1x2)]]
    C (3x4) = [[C11 (2x2), C12 (2x2)],
               [C21 (1x2), C22 (1x2)]]
    
    C11 = A11*B11 + A12*B21  (2x2 = 2x2*2x2 + 2x1*1x2)
    C12 = A11*B12 + A12*B22  (2x2 = 2x2*2x2 + 2x1*1x2)
    C21 = A21*B11 + A22*B21  (1x2 = 1x2*2x2 + 1x1*1x2)
    C22 = A21*B12 + A22*B22  (1x2 = 1x2*2x2 + 1x1*1x2)
    
    Total schoolbook: 4 + 4 + 2 + 2 + 2 + 2 + 1 + 1 = 18 for the "small" parts
    Plus C11 needs A11*B11 (4 mults) and C12 needs A11*B12 (4 mults)
    Total schoolbook = 28 + 8 = 36
    
    Using Strassen for A11*B11: saves 4-7 = -3 (worse!)
    Actually Strassen 2x2*2x2 uses 7 vs 4 schoolbook, so it's worse for small sizes.
    
    Let me think differently. The total is 36. Can we do better?
    """
    # Actually for 3x3*3x4, let me try a different approach
    # Use the fact that we can share some computations
    
    n, m, p = 3, 3, 4
    rank = 36
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _genetic_search(target_rank, n, m, p, deadline, seed=42):
    """
    Genetic algorithm to find sparse support patterns.
    Each individual is a set of (row, col) non-zero positions for U, V, W.
    We evolve to satisfy the tensor identity.
    """
    rng = np.random.RandomState(seed)
    
    # Represent each factor as a list of non-zero (row, value) pairs per column
    # For now, try random sparse patterns with small integers
    
    best = None
    
    def evaluate(U, V, W):
        """Return number of satisfied tensor equations (out of n*m*p)."""
        satisfied = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(target_rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s == 1:
                        satisfied += 1
        return satisfied
    
    def mutate(U, V, W, rate=0.1):
        """Randomly perturb entries."""
        U_new = [row[:] for row in U]
        V_new = [row[:] for row in V]
        W_new = [row[:] for row in W]
        
        for _ in range(int(target_rank * rate * 3)):
            mat = rng.randint(3)
            r = rng.randint(U[0].__len__() if mat==0 else (V[0].__len__() if mat==1 else W[0].__len__()))
            c = rng.randint(target_rank)
            if mat == 0:
                U_new[r][c] = rng.randint(-2, 3)
            elif mat == 1:
                V_new[r][c] = rng.randint(-2, 3)
            else:
                W_new[r][c] = rng.randint(-2, 3)
        
        return U_new, V_new, W_new
    
    # Initialize population
    pop_size = 100
    population = []
    for _ in range(pop_size):
        U = [[0]*target_rank for _ in range(n*m)]
        V = [[0]*target_rank for _ in range(m*p)]
        W = [[0]*target_rank for _ in range(p*n)]
        
        for c in range(target_rank):
            # Sparse: 1-3 nonzeros per column
            nz_u = rng.randint(1, 4)
            rows_u = rng.choice(n*m, size=nz_u, replace=False)
            for r in rows_u:
                U[r][c] = rng.randint(-2, 3)
            
            nz_v = rng.randint(1, 4)
            rows_v = rng.choice(m*p, size=nz_v, replace=False)
            for r in rows_v:
                V[r][c] = rng.randint(-2, 3)
            
            nz_w = rng.randint(1, 4)
            rows_w = rng.choice(p*n, size=nz_w, replace=False)
            for r in rows_w:
                W[r][c] = rng.randint(-2, 3)
        
        score = evaluate(U, V, W)
        population.append((score, U, V, W))
    
    # Check if any is perfect
    for score, U, V, W in population:
        if score == n * m * p:
            return {"U": U, "V": V, "W": W}
    
    # Evolution
    generations = 500
    for gen in range(generations):
        if time.time() > deadline:
            break
        
        # Sort by fitness
        population.sort(key=lambda x: x[0], reverse=True)
        
        # Check best
        if population[0][0] == n * m * p:
            return {"U": population[0][1], "V": population[0][2], "W": population[0][3]}
        
        # Crossover and mutation
        new_pop = [population[0]]  # Elitism
        while len(new_pop) < pop_size:
            if time.time() > deadline:
                break
            
            # Tournament selection
            idx1 = rng.randint(0, min(20, pop_size))
            idx2 = rng.randint(0, min(20, pop_size))
            parent1 = population[idx1]
            parent2 = population[idx2]
            
            # Crossover: mix columns
            U_child = [row[:] for row in parent1[1]]
            V_child = [row[:] for row in parent1[2]]
            W_child = [row[:] for row in parent1[3]]
            
            for c in range(target_rank):
                if rng.random() < 0.5:
                    for r in range(n*m):
                        U_child[r][c] = parent2[1][r][c]
                    for r in range(m*p):
                        V_child[r][c] = parent2[2][r][c]
                    for r in range(p*n):
                        W_child[r][c] = parent2[3][r][c]
            
            # Mutate
            U_child, V_child, W_child = mutate(U_child, V_child, W_child, rate=0.15)
            
            score = evaluate(U_child, V_child, W_child)
            new_pop.append((score, U_child, V_child, W_child))
        
        population = new_pop
    
    # Return best found
    population.sort(key=lambda x: x[0], reverse=True)
    if population[0][0] == n * m * p:
        return {"U": population[0][1], "V": population[0][2], "W": population[0][3]}
    return None

def _algebraic_search(target_rank, n, m, p, deadline):
    """
    Try to find a decomposition by solving linear equations.
    Fix U and V, then solve for W using the tensor identity.
    """
    rng = np.random.RandomState(42)
    
    # Strategy: fix U and V to be sparse with small integers,
    # then for each (i,j,k), the equation is:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is linear in W entries.
    
    # For a given (i,j,k), the constraint involves W[k*n+i,c] for all c.
    # But W[k*n+i,c] appears in multiple constraints (for different j).
    
    # Let's try: fix U and V, then the system is linear in W.
    # Number of W variables: p*n*target_rank = 12*target_rank
    # Number of constraints: n*m*p = 36
    
    # We need target_rank >= 36/12 = 3 for feasibility (in principle)
    # But the structure makes it much harder.
    
    for trial in range(500):
        if time.time() > deadline:
            break
        
        # Generate sparse U and V
        U = np.zeros((n*m, target_rank), dtype=int)
        V = np.zeros((m*p, target_rank), dtype=int)
        
        for c in range(target_rank):
            nz_u = rng.randint(1, 4)
            rows_u = rng.choice(n*m, size=nz_u, replace=False)
            for r in rows_u:
                U[r, c] = rng.randint(-2, 3)
            
            nz_v = rng.randint(1, 4)
            rows_v = rng.choice(m*p, size=nz_v, replace=False)
            for r in rows_v:
                V[r, c] = rng.randint(-2, 3)
        
        # Build linear system for W
        # Variables: W[k*n+i, c] for i in 0..n-1, k in 0..p-1, c in 0..target_rank-1
        # Total variables: p*n*target_rank
        num_w_vars = p * n * target_rank
        
        # For each (i, j, k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This constrains W[k*n+i, c] weighted by U[i*m+j,c]*V[j*p+k,c]
        
        # Build A matrix and b vector
        A = np.zeros((n*m*p, num_w_vars), dtype=int)
        b = np.ones(n*m*p, dtype=int)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(target_rank):
                        coeff = U[i*m+j, c] * V[j*p+k, c]
                        if coeff != 0:
                            var_idx = k * n * target_rank + i * target_rank + c
                            A[eq_idx, var_idx] += coeff
                    eq_idx += 1
        
        # Solve using least squares (integer solution may not exist)
        try:
            # Use np.linalg.lstsq
            sol, residuals, rank_mat, sv = np.linalg.lstsq(A.astype(float), b.astype(float), rcond=None)
            
            # Check if solution is close to integer
            W_sol = np.round(sol).astype(int)
            W_mat = W_sol.reshape(p * n, target_rank)
            
            # Verify
            W_list = W_mat.tolist()
            U_list = U.tolist()
            V_list = V.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
        except:
            continue
    
    return None

def _try_rank32_construction():
    """
    Try to construct a rank-32 decomposition.
    
    For 3x3 * 3x4:
    - The tensor has 36 components
    - Lower bound: max(9, 12, 12) = 12 (too optimistic)
    - Better lower bound from border rank considerations
    
    Let me try a block approach:
    Split into 2+1 in first dimension, 2+1 in second, 2+2 in third.
    
    Actually, let me try to combine Strassen with schoolbook more cleverly.
    
    A (3x3): rows split as 2+1, cols split as 2+1
    B (3x4): rows split as 2+1, cols split as 2+2
    C (3x4): rows split as 2+1, cols split as 2+2
    
    C11 (2x2) = A11(2x2)*B11(2x2) + A12(2x1)*B21(1x2)
    C12 (2x2) = A11(2x2)*B12(2x2) + A12(2x1)*B22(1x2)
    C21 (1x2) = A21(1x2)*B11(2x2) + A22(1x1)*B21(1x2)
    C22 (1x2) = A21(1x2)*B12(2x2) + A22(1x1)*B22(1x2)
    
    Schoolbook costs:
    C11: 4 + 2 = 6
    C12: 4 + 2 = 6
    C21: 2 + 1 = 3
    C22: 2 + 1 = 3
    Total: 18
    
    Wait, that's only 18? No, I'm miscounting.
    
    A11*B11: 2x2 * 2x2 = 4 multiplications
    A12*B21: 2x1 * 1x2 = 2 multiplications
    A11*B12: 2x2 * 2x2 = 4 multiplications
    A12*B22: 2x1 * 1x2 = 2 multiplications
    A21*B11: 1x2 * 2x2 = 2 multiplications
    A22*B21: 1x1 * 1x2 = 1 multiplication
    A21*B12: 1x2 * 2x2 = 2 multiplications
    A22*B22: 1x1 * 1x2 = 1 multiplication
    
    Total: 4+2+4+2+2+1+2+1 = 18
    
    But wait, the full schoolbook is 3*3*4 = 36. Where's the discrepancy?
    
    Oh I see, I'm computing C = A*B where C is 3x4.
    C[i,k] = sum_j A[i,j]*B[j,k]
    
    So C has 3*4 = 12 entries, each requiring 3 multiplications = 36 total.
    
    In my block decomposition:
    C11 (2x2): 4 entries, each needs sum over j=0,1,2 of A[i,j]*B[j,k]
    = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    If I split j into {0,1} and {2}:
    = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + A[i,2]*B[2,k]
    
    The first part is 2x2*2x2 (4 mults for all 4 entries of C11)
    The second part is 2x1*1x2 (2 mults for all 4 entries of C11)
    
    So C11 costs 4+2 = 6. Similarly C12 costs 6.
    C21 (1x2): 2 entries, each needs 3 mults = 6 total
    = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k]
    = (A[2,0]*B[0,k] + A[2,1]*B[1,k]) + A[2,2]*B[2,k]
    = 1x2*2x2 + 1x1*1x2 = 2 + 1 = 3... 
    
    Wait, 1x2 * 2x2: A21 is 1x2, B11 is 2x2. Result is 1x2.
    C21[k] = A21[0]*B11[0,k] + A21[1]*B11[1,k] for k=0,1
    That's 2 multiplications for k=0 and 2 for k=1 = 4 total? No.
    
    For a 1x2 * 2x2 product giving 1x2:
    Result[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]  (2 mults)
    Result[1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]  (2 mults)
    Total: 4 multiplications
    
    I was wrong. Let me recount:
    A11(2x2) * B11(2x2) -> 2x2: 4 mults
    A12(2x1) * B21(1x2) -> 2x2: 2*1*2 = 4 mults (each of 4 outputs needs 1 mult)
    
    Actually for outer product style: (2x1)*(1x2) = 2x2, each element C[i,k] = A[i,0]*B[0,k], so 4 mults.
    
    So:
    C11: A11*B11 (4) + A12*B21 (4) = 8
    C12: A11*B12 (4) + A12*B22 (4) = 8
    C21: A21*B11 (4) + A22*B21 (2) = 6  [A22(1x1)*B21(1x2) = 1x2, 2 mults]
    C22: A21*B12 (4) + A22*B22 (2) = 6
    
    Total: 8+8+6+6 = 28
    
    Hmm, still not 36. Let me recount schoolbook:
    3*3*4 = 36. Yes.
    
    Block: 8+8+6+6 = 28. That's less than 36, which means I'm double-counting or missing something.
    
    Oh wait, I think the issue is that in schoolbook, each C[i,k] = sum_{j=0}^{2} A[i,j]*B[j,k] uses 3 separate multiplications. Total 12*3 = 36.
    
    In the block approach, I'm grouping:
    C11[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    
    In blocks: A11*B11 gives A[0,0]*B[0,0] + A[0,1]*B[1,0] (2 mults for this element)
    A12*B21 gives A[0,2]*B[2,0] (1 mult)
    Total for this element: 3 mults. Correct.
    
    For all 4 elements of C11: 4*3 = 12 mults in schoolbook.
    In blocks: A11*B11 = 4 mults (Strassen could do 7, but schoolbook is 4), A12*B21 = 4 mults.
    Total: 8 mults for 4 elements = 2 per element on average.
    
    But each element needs 3 mults! So 4 elements need 12 mults, not 8.
    
    I see my error. A11(2x2)*B11(2x2) gives a 2x2 result, which accounts for 4 of the 12 products needed for C11. The remaining 4 products come from A12*B21. So total is 4+4=8, but we need 12.
    
    Wait no. C11[i,k] = A11[i,0]*B11[0,k] + A11[i,1]*B11[1,k] + A12[i,0]*B21[0,k]
    
    So for each of the 4 elements (i,k) in C11:
    - 2 products from A11*B11
    - 1 product from A12*B21
    Total: 3 products per element, 12 total.
    
    A11*B11 as a matrix product: computing the full 2x2*2x2->2x2 takes 4 multiplications (schoolbook).
    A12*B21 as 2x1*1x2->2x2: takes 4 multiplications.
    Total: 8 multiplications to compute all 4 elements of C11.
    
    But we need 12 scalar multiplications! The discrepancy is because matrix multiplication "A11*B11" in 4 scalar mults computes 4 outputs each as a sum of 2 products. So it's doing 8 scalar multiplications in 4 "multiplication operations" if we count each bilinear form as one.
    
    I'm confusing scalar multiplications with bilinear forms. In tensor rank, each "rank-1 term" is one bilinear form. Schoolbook 3x3*3x4 uses 36 bilinear forms.
    
    OK so the block decomposition with schoolbook within blocks gives:
    C11: 4 (for A11*B11) + 4 (for A12*B21) = 8
    C12: 4 + 4 = 8
    C21: 4 (for A21*B11) + 2 (for A22*B21) = 6
    C22: 4 + 2 = 6
    Total: 28
    
    But this should equal 36! Let me recount.
    
    A21(1x2) * B11(2x2) -> 1x2: C21[k] = A21[0]*B11[0,k] + A21[1]*B11[1,k]
    For k=0: 2 mults. For k=1: 2 mults. Total: 4 bilinear forms. ✓
    
    A22(1x1) * B21(1x2) -> 1x2: C21[k] += A22[0]*B21[0,k]
    For k=0: 1 mult. For k=1: 1 mult. Total: 2 bilinear forms. ✓
    
    So C21 = 4 + 2 = 6. ✓
    
    Sum: 8+8+6+6 = 28. But schoolbook is 36.
    
    The issue: in schoolbook, C[i,k] = sum_{j=0}^{2} A[i,j]*B[j,k]. Each term A[i,j]*B[j,k] is one bilinear form. Total 3*4*3 = 36.
    
    In my block count:
    C11 has 4 elements, each needs 3 bilinear forms = 12. But I counted 8.
    
    A11*B11: This is a 2x2*2x2 matrix product. Schoolbook uses 4 bilinear forms to compute all 4 outputs. Each output C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] uses 2 bilinear forms. 4 outputs * 2 = 8 bilinear forms? No, schoolbook 2x2*2x2 uses 4 bilinear forms total (one per output, each being a sum of 2 products).
    
    I think the confusion is: in tensor rank, a "bilinear form" u^T A v * w^T B is one rank-1 term. The schoolbook algorithm for m*n*p matrix multiplication uses m*n*p such terms.
    
    For 2x2*2x2: schoolbook uses 4 terms. Each term contributes to one output element.
    Term 1: A[0,0]*B[0,0] -> C[0,0]
    Term 2: A[0,1]*B[1,0] -> C[0,0]
    Term 3: A[1,0]*B[0,1] -> C[1,1]
    Term 4: A[1,1]*B[1,1] -> C[1,1]
    
    Wait that's only 4 terms but C[0,1] and C[1,0] are missing!
    
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] (2 terms)
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] (2 terms)
    
    So total: 4+2+2 = 8 terms for 2x2*2x2? No, 2*2*2 = 8. Yes!
    
    So schoolbook 2x2*2x2 uses 8 bilinear forms, not 4. I was wrong.
    
    Strassen uses 7 for 2x2*2x2, saving 1.
    
    OK so for 3x3*3x4: schoolbook = 36.
    
    Block decomposition with schoolbook within blocks:
    C11: A11*B11 (2*2*2=8) + A12*B21 (2*1*2=4) = 12
    C12: A11*B12 (8) + A12*B22 (4) = 12
    C21: A21*B11 (1*2*2=4) + A22*B21 (1*1*2=2) = 6
    C22: A21*B12 (4) + A22*B22 (2) = 6
    Total: 12+12+6+6 = 36 ✓
    
    Now using Strassen for A11*B11 (7 instead of 8) and A11*B12 (7 instead of 8):
    C11: 7 + 4 = 11
    C12: 7 + 4 = 11
    C21: 4 + 2 = 6
    C22: 4 + 2 = 6
    Total: 11+11+6+6 = 34
    
    So rank 34 is achievable! That's an improvement over 36.
    
    Can we do better? Use Strassen for A21*B11? No, 1x2*2x2 is too small.
    
    What about a different split? 3 = 1+1+1?
    Or use more sophisticated methods?
    
    For now, let's implement rank 34.
    """
    n, m, p = 3, 3, 4
    rank = 34
    
    U = [[0]*rank for _ in range(n*m)]  # 9 x 34
    V = [[0]*rank for _ in range(m*p)]  # 12 x 34
    W = [[0]*rank for _ in range(p*n)]  # 12 x 34
    
    # Block indices:
    # A: rows 0-1 are block 1, row 2 is block 2; cols 0-1 are block 1, col 2 is block 2
    # B: rows 0-1 are block 1, row 2 is block 2; cols 0-1 are block 1, cols 2-3 are block 2
    # C: rows 0-1 are block 1, row 2 is block 2; cols 0-1 are block 1, cols 2-3 are block 2
    
    # A[i,j] -> U[i*3+j]
    # B[j,k] -> V[j*4+k]
    # C[i,k] -> W[k*3+i]
    
    # C11 (2x2) = A11(2x2)*B11(2x2) + A12(2x1)*B21(1x2)
    # A11: A[0,0],A[0,1],A[1,0],A[1,1] -> U[0],U[1],U[3],U[4]
    # B11: B[0,0],B[0,1],B[1,0],B[1,1] -> V[0],V[1],V[4],V[5]
    # A12: A[0,2],A[1,2] -> U[2],U[5]
    # B21: B[2,0],B[2,1] -> V[8],V[9]
    # C11: C[0,0],C[0,1],C[1,0],C[1,1] -> W[0],W[1],W[3],W[4]
    
    # Use Strassen for A11*B11 (7 terms, cols 0-6)
    # Strassen for 2x2*2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 - M4 + M5 + M7
    # C01 = M2 + M4
    # C10 = M1 + M3 - M5 - M7
    # C11 = M1 - M2 + M3 + M6
    
    # A00=A[0,0]=U[0], A01=A[0,1]=U[1], A10=A[1,0]=U[3], A11=A[1,1]=U[4]
    # B00=B[0,0]=V[0], B01=B[0,1]=V[1], B10=B[1,0]=V[4], B11=B[1,1]=V[5]
    # C00=C[0,0]=W[0], C01=C[0,1]=W[1], C10=C[1,0]=W[3], C11=C[1,1]=W[4]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[3][0] = 1; W[4][0] = 1
    # C00 += M1, C01 += M1, C10 += M1, C11 += M1
    
    # M2 = (A10+A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[4][1] = -1
    # C01 += M2, C11 -= M2
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[4][2] = 1; V[5][2] = -1
    W[3][2] = 1; W[4][2] = 1
    # C10 += M3, C11 += M3
    
    # M4 = A11(B01-B00)
    U[4][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[3][3] = 1
    # C00 -= M4, C10 += M4
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = 1; W[3][4] = -1
    # C00 += M5, C10 -= M5
    
    # M6 = (A10-A00)(B00+B10)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[4][5] = 1
    W[4][5] = 1
    # C11 += M6
    
    # M7 = (A01-A11)(B01+B11)
    U[1][6] = 1; U[4][6] = -1
    V[1][6] = 1; V[5][6] = 1
    W[0][6] = 1; W[3][6] = -1
    # C00 += M7, C10 -= M7
    
    # Now add A12*B21 (4 terms, cols 7-10)
    # A12: A[0,2]=U[2], A[1,2]=U[5]
    # B21: B[2,0]=V[8], B[2,1]=V[9]
    # C11: C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[3], C[1,1]=W[4]
    # C[i,k] += A12[i,0]*B21[0,k]
    
    # C[0,0] += A[0,2]*B[2,0]
    U[2][7] = 1; V[8][7] = 1; W[0][7] = 1
    # C[0,1] += A[0,2]*B[2,1]
    U[2][8] = 1; V[9][8] = 1; W[1][8] = 1
    # C[1,0] += A[1,2]*B[2,0]
    U[5][9] = 1; V[8][9] = 1; W[3][9] = 1
    # C[1,1] += A[1,2]*B[2,1]
    U[5][10] = 1; V[9][10] = 1; W[4][10] = 1
    
    # C12 (2x2) = A11(2x2)*B12(2x2) + A12(2x1)*B22(1x2)
    # B12: B[0,2],B[0,3],B[1,2],B[1,3] -> V[2],V[3],V[6],V[7]
    # B22: B[2,2],B[2,3] -> V[10],V[11]
    # C12: C[0,2],C[0,3],C[1,2],C[1,3] -> W[6],W[7],W[9],W[10]
    
    # Use Strassen for A11*B12 (7 terms, cols 11-17)
    # A00=U[0], A01=U[1], A10=U[3], A11=U[4]
    # B00=V[2], B01=V[3], B10=V[6], B11=V[7]
    # C00=W[6], C01=W[7], C10=W[9], C11=W[10]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][11] = 1; U[4][11] = 1
    V[2][11] = 1; V[7][11] = 1
    W[6][11] = 1; W[7][11] = 1; W[9][11] = 1; W[10][11] = 1
    
    # M2 = (A10+A11)B00
    U[3][12] = 1; U[4][12] = 1
    V[2][12] = 1
    W[7][12] = 1; W[10][12] = -1
    
    # M3 = A00(B10-B11)
    U[0][13] = 1
    V[6][13] = 1; V[7][13] = -1
    W[9][13] = 1; W[10][13] = 1
    
    # M4 = A11(B01-B00)
    U[4][14] = 1
    V[3][14] = 1; V[2][14] = -1
    W[6][14] = -1; W[9][14] = 1
    
    # M5 = (A00+A01)B11
    U[0][15] = 1; U[1][15] = 1
    V[7][15] = 1
    W[6][15] = 1; W[9][15] = -1
    
    # M6 = (A10-A00)(B00+B10)
    U[3][16] = 1; U[0][16] = -1
    V[2][16] = 1; V[6][16] = 1
    W[10][16] = 1
    
    # M7 = (A01-A11)(B01+B11)
    U[1][17] = 1; U[4][17] = -1
    V[3][17] = 1; V[7][17] = 1
    W[6][17] = 1; W[9][17] = -1
    
    # Add A12*B22 (4 terms, cols 18-21)
    # C[0,2] += A[0,2]*B[2,2]
    U[2][18] = 1; V[10][18] = 1; W[6][18] = 1
    # C[0,3] += A[0,2]*B[2,3]
    U[2][19] = 1; V[11][19] = 1; W[7][19] = 1
    # C[1,2] += A[1,2]*B[2,2]
    U[5][20] = 1; V[10][20] = 1; W[9][20] = 1
    # C[1,3] += A[1,2]*B[2,3]
    U[5][21] = 1; V[11][21] = 1; W[10][21] = 1
    
    # C21 (1x2) = A21(1x2)*B11(2x2) + A22(1x1)*B21(1x2)
    # A21: A[2,0]=U[6], A[2,1]=U[7]
    # A22: A[2,2]=U[8]
    # B11: V[0],V[1],V[4],V[5]
    # B21: V[8],V[9]
    # C21: C[2,0]=W[12], C[2,1]=W[13]
    
    # A21*B11 (4 terms, cols 22-25)
    # C[2,0] += A[2,0]*B[0,0]
    U[6][22] = 1; V[0][22] = 1; W[12][22] = 1
    # C[2,0] += A[2,1]*B[1,0]
    U[7][23] = 1; V[4][23] = 1; W[12][23] = 1
    # C[2,1] += A[2,0]*B[0,1]
    U[6][24] = 1; V[1][24] = 1; W[13][24] = 1
    # C[2,1] += A[2,1]*B[1,1]
    U[7][25] = 1; V[5][25] = 1; W[13][25] = 1
    
    # A22*B21 (2 terms, cols 26-27)
    # C[2,0] += A[2,2]*B[2,0]
    U[8][26] = 1; V[8][26] = 1; W[12][26] = 1
    # C[2,1] += A[2,2]*B[2,1]
    U[8][27] = 1; V[9][27] = 1; W[13][27] = 1
    
    # C22 (1x2) = A21(1x2)*B12(2x2) + A22(1x1)*B22(1x2)
    # B12: V[2],V[3],V[6],V[7]
    # B22: V[10],V[11]
    # C22: C[2,2]=W[14], C[2,3]=W[15]
    
    # A21*B12 (4 terms, cols 28-31)
    # C[2,2] += A[2,0]*B[0,2]
    U[6][28] = 1; V[2][28] = 1; W[14][28] = 1
    # C[2,2] += A[2,1]*B[1,2]
    U[7][29] = 1; V[6][29] = 1; W[14][29] = 1
    # C[2,3] += A[2,0]*B[0,3]
    U[6][30] = 1; V[3][30] = 1; W[15][30] = 1
    # C[2,3] += A[2,1]*B[1,3]
    U[7][31] = 1; V[7][31] = 1; W[15][31] = 1
    
    # A22*B22 (2 terms, cols 32-33)
    # C[2,2] += A[2,2]*B[2,2]
    U[8][32] = 1; V[10][32] = 1; W[14][32] = 1
    # C[2,3] += A[2,2]*B[2,3]
    U[8][33] = 1; V[11][33] = 1; W[15][33] = 1
    
    return {"U": U, "V": V, "W": W}

def _try_rank33_construction():
    """
    Try to get rank 33 by finding one more optimization.
    
    From rank 34, can we save one more multiplication?
    
    Looking at the cross terms:
    A12*B21: 4 terms
    A12*B22: 4 terms
    A21*B11: 4 terms
    A21*B12: 4 terms
    A22*B21: 2 terms
    A22*B22: 2 terms
    Total cross: 20 terms
    
    Can we combine any of these? The A12 terms (cols 7-10, 18-21) involve
    U[2] and U[5] (A[0,2] and A[1,2]).
    
    Similarly A21 terms involve U[6] and U[7].
    
    It's hard to save more without a fundamentally different approach.
    Let's try rank 33 with a search.
    """
    return None

def _try_rank32_construction():
    """Try rank 32."""
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140 second budget, leaving 10s margin
    
    # First, try the rank-34 Strassen hybrid construction
    result = _try_rank32_construction()
    if result is not None:
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    result = _try_rank33_construction()
    if result is not None:
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    # Rank 34 Strassen hybrid
    result = _try_rank32_construction()  # placeholder, actually rank 34
    # Let me just use the rank 34 construction directly
    result = _construct_3x3x4_strassen_hybrid()
    if result is not None:
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    # Try algebraic search for lower rank
    for target_rank in [33, 32, 31, 30]:
        if time.time() > deadline:
            break
        result = _algebraic_search(target_rank, n, m, p, deadline)
        if result is not None:
            return result
    
    # Try genetic search
    for target_rank in [33, 32]:
        if time.time() > deadline:
            break
        result = _genetic_search(target_rank, n, m, p, deadline, seed=target_rank)
        if result is not None:
            return result
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)
