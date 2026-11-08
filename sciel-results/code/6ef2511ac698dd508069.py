"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower-rank decomposition using structured search
    # For 3x3 * 3x5, we can try ranks below 45
    
    def verify_decomposition(U, V, W, rank):
        """Verify that U, V, W form a valid matrix multiplication decomposition."""
        # Check: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 if this is a valid term, else 0
        # Actually the tensor condition is: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For matrix mult, T[i,j,k] = 1 for all i,j,k (the multiplication tensor has all-ones)
        # Wait no - the matrix multiplication tensor M_{i,j,k} where C[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all i,j,k in their ranges
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True
    
    def verify_decomposition_fast(U, V, W, rank):
        """Faster verification using the tensor structure."""
        # T[i,j,k] should equal 1 for all i in [0,n), j in [0,m), k in [0,p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        u_val = U[i * m + j][c]
                        v_val = V[j * p + k][c]
                        w_val = W[k * n + i][c]
                        s += u_val * v_val * w_val
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Search for structured decompositions
    # Strategy: Try small integer/half-integer values for U, V, W entries
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # For 3x3*3x5, known bounds: rank is at most 45 (schoolbook)
    # Lower bound: at least max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # Let's try to find decompositions with rank < 45
    
    # Try a structured approach: use the fact that matrix multiplication can be
    # decomposed using block structures
    
    # Approach: Try all possible "sparse" patterns for small ranks
    # For rank r, each column c defines a triple of vectors (u_c, v_c, w_c)
    # where u_c is in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    # and sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1 for all i,j,k
    
    # Try ranks from lower bound up
    min_possible_rank = max(n*m, m*p, p*n)  # 15
    
    # Try a few candidate ranks with structured search
    candidate_ranks = [r for r in range(min_possible_rank, best_rank + 1) if r <= 40]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 30:
            break
        
        # For each rank, try to find a decomposition
        # Use a simple randomized + local search approach
        
        def random_decomposition(rank, seed_offset=0):
            """Generate a random sparse decomposition attempt."""
            import random
            rng = random.Random(seed_offset)
            
            # Try to build a valid decomposition
            # Each "factor" c has support on some subset of (i,j), (j,k), (k,i)
            
            # Simple approach: try to cover all (i,j,k) triples
            # with products of sparse vectors
            
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Try different strategies
            strategy = rng.randint(0, 2)
            
            if strategy == 0:
                # Dense random with small integers
                for c in range(rank):
                    for i in range(n):
                        for j in range(m):
                            if rng.random() < 0.3:
                                U[i*m+j][c] = rng.choice([-1, 1])
                    for j in range(m):
                        for k in range(p):
                            if rng.random() < 0.3:
                                V[j*p+k][c] = rng.choice([-1, 1])
                    for k in range(p):
                        for i in range(n):
                            if rng.random() < 0.3:
                                W[k*n+i][c] = rng.choice([-1, 1])
                
                # Try to fix up
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            target = 1
                            current = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                            if current != target:
                                # Find a column to adjust
                                for c in range(rank):
                                    if U[i*m+j][c] != 0 and V[j*p+k][c] != 0 and W[k*n+i][c] != 0:
                                        # Adjust W
                                        W[k*n+i][c] += (target - current) / (U[i*m+j][c] * V[j*p+k][c])
                                        break
            elif strategy == 1:
                # Block-structured: group indices
                # For 3x3*3x5, try 3x3 blocks on the 5 dimension
                # Split p=5 into 3+2
                for c in range(rank):
                    # Assign each factor to a specific (i, j, k_block) pattern
                    i_val = c % n
                    j_val = (c // n) % m
                    k_val = (c // (n*m)) % p
                    U[i_val*m + j_val][c] = 1
                    V[j_val*p + k_val][c] = 1
                    W[k_val*n + i_val][c] = 1
            else:
                # Strassen-like: use sums/differences
                # For 2x2*2x2, Strassen uses 7 factors
                # For 3x3, we can try to generalize
                for c in range(rank):
                    # Use a structured pattern
                    i1 = c % 2
                    i2 = (c // 2) % 2
                    j1 = (c // 4) % 2
                    j2 = (c // 8) % 2
                    k1 = (c // 16) % 2
                    k2 = (c // 32) % 2
                    
                    # This is getting complex; use a simpler approach
                    # Just try all-ones for a subset
                    if c < n*m:
                        U[c][c] = 1
                    if c < m*p:
                        V[c][c] = 1
                    if c < p*n:
                        W[c][c] = 1
        
        # Try a few random initializations and local optimization
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and time.time() < deadline - 30:
            attempts += 1
            
            # Generate initial guess
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Initialize with a structured pattern
            for c in range(rank):
                # Each factor handles a subset of the multiplication
                # Try to make it work
                for i in range(n):
                    for j in range(m):
                        U[i*m+j][c] = 1 if (i*m+j) % rank == c else 0
                for j in range(m):
                    for k in range(p):
                        V[j*p+k][c] = 1 if (j*p+k) % rank == c else 0
                for k in range(p):
                    for i in range(n):
                        W[k*n+i][c] = 1 if (k*n+i) % rank == c else 0
            
            # Local optimization: try to fix errors
            for iteration in range(10):
                if time.time() > deadline - 30:
                    break
                
                # Check current state
                errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                            if abs(s - 1) > 1e-6:
                                errors.append((i, j, k, s))
                
                if not errors:
                    # Found a valid decomposition!
                    # Convert to exact fractions
                    U_frac = [[Fraction(U[i][c]) for c in range(rank)] for i in range(n*m)]
                    V_frac = [[Fraction(V[i][c]) for c in range(rank)] for i in range(m*p)]
                    W_frac = [[Fraction(W[i][c]) for c in range(rank)] for i in range(p*n)]
                    
                    # Verify with exact arithmetic
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = sum(U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c] for c in range(rank))
                                if s != 1:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if not valid:
                        break
                    
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {
                            "U": [[float(x) for x in row] for row in U_frac],
                            "V": [[float(x) for x in row] for row in V_frac],
                            "W": [[float(x) for x in row] for row in W_frac]
                        }
                    break
                
                # Pick a random error and try to fix it
                i, j, k, s = errors[0]
                # Find a column where we can adjust
                for c in range(rank):
                    if U[i*m+j][c] != 0 and V[j*p+k][c] != 0:
                        # Adjust W
                        needed = 1 - s
                        current_w = W[k*n+i][c]
                        new_w = current_w + needed / (U[i*m+j][c] * V[j*p+k][c])
                        W[k*n+i][c] = new_w
                        break
                    elif U[i*m+j][c] != 0 and W[k*n+i][c] != 0:
                        # Adjust V
                        needed = 1 - s
                        current_v = V[j*p+k][c]
                        new_v = current_v + needed / (U[i*m+j][c] * W[k*n+i][c])
                        V[j*p+k][c] = new_v
                        break
                    elif V[j*p+k][c] != 0 and W[k*n+i][c] != 0:
                        # Adjust U
                        needed = 1 - s
                        current_u = U[i*m+j][c]
                        new_u = current_u + needed / (V[j*p+k][c] * W[k*n+i][c])
                        U[i*m+j][c] = new_u
                        break
            
            # Check if we found a solution
            if rank < best_rank:
                if verify_decomposition_fast(U, V, W, rank):
                    # Convert to fractions for exactness
                    U_frac = [[Fraction(U[i][c]).limit_denominator(1000) for c in range(rank)] for i in range(n*m)]
                    V_frac = [[Fraction(V[i][c]).limit_denominator(1000) for c in range(rank)] for i in range(m*p)]
                    W_frac = [[Fraction(W[i][c]).limit_denominator(1000) for c in range(rank)] for i in range(p*n)]
                    
                    # Verify exact
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = sum(U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c] for c in range(rank))
                                if s != 1:
                                    valid = False
                                    break
                        if not valid:
                            break
                    
                    if valid:
                        best_rank = rank
                        best_result = {
                            "U": [[float(x) for x in row] for row in U_frac],
                            "V": [[float(x) for x in row] for row in V_frac],
                            "W": [[float(x) for x in row] for row in W_frac]
                        }
                        break
        
        if rank < best_rank:
            continue  # Try next rank
    
    return best_result
# EVOLVE-BLOCK-END
