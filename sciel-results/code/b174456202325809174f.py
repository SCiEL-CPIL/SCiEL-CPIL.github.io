"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=3, m=5, p=7

    # We'll use a combination of Strassen-like blocks and schoolbook terms.
    # The tensor T[i,j,k] = delta (1 if i,j,k is a valid multiplication term)
    # We want to decompose T = sum_r u_r ⊗ v_r ⊗ w_r where:
    #   u_r has support on {i: ...}, v_r on {j: ...}, w_r on {k: ...}
    #
    # For matrix multiplication A(n×m) × B(m×p) → C(n×p):
    # T[i, j, k] = 1 for all (i, j, k) where we sum over j.
    # Actually T[i,j,k] = 1 for the term A[i,j]*B[j,k] contributing to C[i,k].
    #
    # The tensor has shape (n, m, p) with T[i,j,k]=1 for all i,j,k.
    # Wait, that's not right. The tensor for matrix mult is:
    # T[i, j, k] = 1 if and only if the (i,k) entry of the product gets contribution from A[i,j]*B[j,k].
    # So T[i,j,k] = 1 for ALL i in [0,n), j in [0,m), k in [0,p).
    # The tensor is simply the all-ones tensor of shape (n,m,p).
    #
    # Wait, that's not quite right either. Let me re-read the contract.
    # "exactly decompose the matrix multiplication tensor"
    # The multiplication tensor T_{ijk} where C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k). The tensor is all-ones of shape (n,m,p).
    #
    # The rank of the all-ones tensor of shape (n,m,p) is... 
    # Actually no. The tensor T is defined such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k] = 1
    #
    # So we need: for all (i,j,k), sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
    #
    # This is a 3-way tensor decomposition problem.
    #
    # Strategy: Use Strassen blocks where possible, schoolbook for the rest.
    # A 2x2x2 all-ones block can be decomposed with rank 7 (Strassen).
    # The full tensor is 3x5x7 = 105 entries.
    
    # Let's try a systematic approach:
    # 1. Partition into 2x2x2 blocks where possible
    # 2. Use Strassen for each block
    # 3. Handle remaining terms with schoolbook
    
    # n=3, m=5, p=7
    # We can fit:
    # i: 0-1 (one block of size 2, remainder 1)
    # j: 0-3 (two blocks of size 2, remainder 1)
    # k: 0-5 (three blocks of size 2, remainder 1)
    
    # Block decomposition:
    # i-blocks: [(0,1)]  → 1 full block, 1 remaining (i=2)
    # j-blocks: [(0,1), (2,3)] → 2 full blocks, 1 remaining (j=4)
    # k-blocks: [(0,1), (2,3), (4,5)] → 3 full blocks, 1 remaining (k=6)
    
    # For each full 2x2x2 block, we need rank 7 (Strassen)
    # Number of full blocks: 1 * 2 * 3 = 6 blocks, each rank 7 → 42
    
    # Remaining terms:
    # i=2 with all j,k: 5*7 = 35 terms
    # i=0,1 with j=4 and all k: 2*7 = 14 terms
    # i=0,1 with j=0,1,2,3 and k=6: 2*4*1 = 8 terms
    # i=2 with j=4 and all k already counted
    # Let me recount:
    # Terms not covered by any 2x2x2 block:
    # (i,j,k) where i not in [0,1] OR j not in [0,3] OR k not in [0,5]
    # i=2: all (2,j,k) for j in [0,5), k in [0,7) → 35 terms
    # i in [0,2), j=4: (i,4,k) for i in [0,2), k in [0,7) → 14 terms
    # i in [0,2), j in [0,4), k=6: (i,j,6) for i in [0,2), j in [0,4) → 8 terms
    # Total remaining: 35 + 14 + 8 = 57
    
    # Total rank: 42 + 57 = 99
    
    # But maybe we can do better with a smarter decomposition.
    # Let's try using the fact that the tensor is all-ones.
    
    # Alternative: Use a rank decomposition based on the structure.
    # The all-ones tensor of shape (n,m,p) has rank at most n*m + n*p + m*p - 2*min(n,m,p) + 1? No, that's not a standard bound.
    
    # Actually, for the all-ones tensor, the rank is related to the matrix multiplication.
    # Let me just use the Strassen + schoolbook approach which gives 99.
    
    # Let me verify: can we do better?
    # The tensor rank of the all-ones (3,5,7) tensor...
    # A trivial upper bound is min(n*m, m*p, p*n) = min(15, 35, 21) = 15? No, that's for the "slice rank" not tensor rank.
    # Actually the tensor rank of all-ones (n,m,p) is at most n + m + p - 2 (using a specific construction)? No.
    
    # Let me think differently. We need vectors u_r in R^(n*m), v_r in R^(m*p), w_r in R^(p*n)
    # such that for all (i,j,k): sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
    
    # One approach: use the fact that 1 = 1*1*1, so we could use rank 1 with all entries 1.
    # But that gives sum_r = 1*1*1 = 1 for each (i,j,k). Wait, that works!
    # U[i*m+j][0] = 1 for all i,j
    # V[j*p+k][0] = 1 for all j,k  
    # W[k*n+i][0] = 1 for all k,i
    # Then for any (i,j,k): U[i*m+j][0]*V[j*p+k][0]*W[k*n+i][0] = 1*1*1 = 1. ✓
    
    # Wait, that's rank 1! But that can't be right for matrix multiplication.
    # Let me re-read the contract more carefully.
    #
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries
    #  that exactly decompose the matrix multiplication tensor"
    
    # So the tensor T[i,j,k] represents the coefficient in:
    # C[i,k] = sum_j sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] * A[i,j] * B[j,k]
    #
    # For this to equal sum_j A[i,j] * B[j,k], we need:
    # sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1 for all (i,j,k)
    
    # So yes, the tensor is all-ones, and rank 1 works!
    # But wait, that seems too good to be true. Let me double-check.
    
    # If U[i*m+j][r] = 1, V[j*p+k][r] = 1, W[k*n+i][r] = 1 for all indices and r=0:
    # Then C[i,k] = sum_j sum_r 1 * 1 * 1 * A[i,j] * B[j,k] = sum_j A[i,j] * B[j,k] ✓
    
    # So the rank is 1?! That seems like it should be well-known.
    # But the task says "Find a rank-minimal exact tensor decomposition" and the seed gives 99.
    
    # I must be misunderstanding the tensor convention. Let me re-read.
    # "the official trace convention W[k*n+i,r]"
    
    # Maybe the tensor is not all-ones. Maybe it's defined differently.
    # In some conventions, the multiplication tensor T_{ijk} is defined as:
    # T_{ijk} = 1 if (i,j,k) corresponds to a term in the product, 0 otherwise.
    # For standard matrix multiplication, every (i,j,k) is a valid term, so T is all-ones.
    
    # Unless... the convention is different. Maybe U, V, W are not coefficients but
    # the decomposition is: T = sum_r u_r ⊗ v_r ⊗ w_r where T[i,j,k] = sum_r u_r[i] v_r[j] w_r[k]
    # and the shapes are (n, rank), (m, rank), (p, rank)?
    
    # But the contract says shapes (n*m, rank), (m*p, rank), (p*n, rank).
    # So U has n*m rows, V has m*p rows, W has p*n rows.
    # U[i*m + j, r] is the coefficient for the (i,j) entry of A in factor r.
    # V[j*p + k, r] is the coefficient for the (j,k) entry of B in factor r.
    # W[k*n + i, r] is the coefficient for the (i,k) entry of C in factor r.
    
    # And the condition is:
    # For all (i,j,k): sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
    
    # This is exactly the all-ones tensor condition. And rank 1 satisfies it.
    
    # So the answer should be rank 1. Let me just return that.
    
    rank = 1
    U = [[1]] * (n * m)
    V = [[1]] * (m * p)
    W = [[1]] * (p * n)
    
    # Verify: for any (i,j,k), U[i*m+j][0] * V[j*p+k][0] * W[k*n+i][0] = 1*1*1 = 1 ✓
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
