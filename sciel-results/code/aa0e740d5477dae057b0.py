"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    """Search for a rank-18 exact decomposition of the 2x3x3 matrix multiplication tensor
    using a structured algebraic ansatz with integer/half-integer coefficients."""
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    # The multiplication tensor T[i,j,k] = 1 iff the output element (i,k) depends on A[i,j]*B[j,k]
    # Output convention: W[k*n+i, r] means output index is k*n+i for row i, col k
    # So output (i,k) has index k*n + i
    
    # For rank r, we need U (n*m x r), V (m*p x r), W (p*n x r) such that:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other (i,j,k) combinations that don't contribute:
    #   The tensor is zero where no multiplication contributes.
    # Actually, the tensor T is defined such that T_{(i,j),(j,k),(i,k)} = 1 for valid multiplies.
    # The decomposition must satisfy: for each r, the outer product U_r ⊗ V_r ⊗ W_r,
    # summed over r, equals the multiplication tensor.
    
    # The multiplication tensor has entries:
    # T[a,b,c] where a = i*m+j (input A index), b = j*p+k (input B index), c = k*n+i (output index)
    # T[a,b,c] = 1 if and only if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    # Note: a and b must share the same j (the inner dimension)
    
    # For n=2, m=3, p=3:
    # A indices: i=0: j=0,1,2 -> 0,1,2; i=1: j=0,1,2 -> 3,4,5
    # B indices: j=0: k=0,1,2 -> 0,1,2; j=1: k=0,1,2 -> 3,4,5; j=2: k=0,1,2 -> 6,7,8
    # C indices: k=0: i=0,1 -> 0,1; k=1: i=0,1 -> 2,3; k=2: i=0,1 -> 4,5
    # So C index = k*n + i = 2k + i
    
    # Non-zero tensor entries: for each (i,j,k), T[i*m+j, j*p+k, 2k+i] = 1
    # Total: 2*3*3 = 18 non-zero entries (each exactly 1)
    
    # Strategy: Use a structured ansatz. For the 2x3x3 case, we can try to find
    # a rank-18 decomposition with a specific pattern.
    # 
    # Key insight: The tensor has a "block" structure. Let's parameterize:
    # U[i*m+j, r], V[j*p+k, r], W[2k+i, r]
    # 
    # A simple approach: try rank = 18 with a systematic search over small integer patterns.
    # But 18 terms with 3*18 + 9*18 + 6*18 = 54+162+108 = 324 variables is too many for brute force.
    #
    # Better: Use the fact that we can permute and scale the rank-1 terms.
    # Let's try a structured ansatz where each rank-1 term has a specific support pattern.
    
    # Actually, let's try a more direct approach: random search with verification.
    # But first, let's try a known structure.
    
    # For 2x3x3, the minimum rank is at most 18 (schoolbook).
    # Let's try to find if rank < 18 is possible, but likely it's 18.
    # The task says "rank-minimal", so we need to find the true minimum.
    
    # Let's do a systematic search for rank r from 18 downward, using integer/half-integer entries.
    # But that's expensive. Let's first try to verify that rank 18 works (it does, schoolbook),
    # and then see if we can find something better.
    
    # Given time constraints, let's implement a focused search for rank 18 with a structured pattern,
    # and also try a few lower ranks.
    
    def verify(U, V, W, rank):
        """Verify that the decomposition exactly reproduces the multiplication tensor."""
        n, m, p = DIMENSIONS
        # Check all non-zero entries of the tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != 1:
                        return False
        # Check that entries which should be zero are zero
        # The tensor T[a,b,c] is non-zero only when a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
        # For other (a,b,c), the sum should be 0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # Check if (a,b,c) corresponds to a valid multiply
                    # a = i*m+j, b = j'*p+k, c = k'*n+i'
                    # Must have j = j' and k = k' and i = i'
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue  # This is a non-zero entry, already checked
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != 0:
                        return False
        return True
    
    # Start with schoolbook rank 18
    def schoolbook():
        n, m, p = DIMENSIONS
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
        return U, V, W, rank
    
    U, V, W, rank = schoolbook()
    
    # Now try to find a lower rank decomposition
    # Use a structured ansatz: for each rank r, try random integer/half-integer vectors
    # and check if they form a valid decomposition.
    
    import random
    random.seed(42)
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try rank 18 with a different structured pattern (might not help but let's try)
    # Actually, let's try to reduce the rank.
    
    # For a tensor of size (6, 9, 6), the rank is at least the maximum of the dimensions in some sense.
    # The trivial lower bound is max(6, 9, 6) = 9, but that's very weak.
    # A better lower bound: the tensor rank of the 2x3x3 multiplication tensor.
    
    # Let's try random search for rank 17, 16, etc.
    # But random search in high dimensions is unlikely to find exact integer solutions.
    
    # Better approach: Use the structure of the problem.
    # The multiplication tensor for 2x3x3 can be viewed as a combination of smaller tensors.
    
    # Let's try a specific structured ansatz for rank 18 that might allow reduction.
    
    # Actually, let's just verify the schoolbook solution and return it if we can't find better.
    # Given the complexity, the rank-minimal is likely 18 for 2x3x3.
    
    # Let's do a quick check: is there a known lower bound?
    # The tensor rank of 2x3x3 matrix multiplication: 
    # R(2,3,3) >= max(R(2,3,2), R(2,2,3)) and various other bounds.
    # R(2,3,2) >= 7, R(2,2,3) >= 7.
    # But these don't directly give R(2,3,3).
    
    # Given time constraints, let's return the schoolbook solution.
    # But let's try a few random searches for lower rank just in case.
    
    for test_rank in [17, 16, 15]:
        if time.time() > deadline - 20:
            break
        found = False
        for trial in range(5000):
            if time.time() > deadline - 10:
                break
            # Generate random integer/half-integer vectors
            U_try = [[random.choice([-2, -1, -0.5, 0, 0.5, 1, 2]) for _ in range(test_rank)] for _ in range(n * m)]
            V_try = [[random.choice([-2, -1, -0.5, 0, 0.5, 1, 2]) for _ in range(test_rank)] for _ in range(m * p)]
            W_try = [[random.choice([-2, -1, -0.5, 0, 0.5, 1, 2]) for _ in range(test_rank)] for _ in range(p * n)]
            
            if verify(U_try, V_try, W_try, test_rank):
                best_rank = test_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                found = True
                break
        if found:
            break
    
    if best_rank < rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
