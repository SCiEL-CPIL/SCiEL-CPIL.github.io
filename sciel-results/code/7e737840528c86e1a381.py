"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 12 decomposition
    # Split A (3x4) into 3 rows of 1x4
    # Each row multiplication 1x4 * 4x8 has rank 4
    # Total rank 3 * 4 = 12
    rank = 12
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            for k in range(p):
                W[k * n + i][r] = 1
            V[j * p + k][r] = 1 # Wait, V depends on k? No.
            # Correction:
            # For term r (corresponding to A[i,j]), 
            # U[i*m+j, r] = 1
            # V[j*p+k, r] should be 1 for all k? No.
            # The term is A[i,j] * B[j,k].
            # So for a fixed r (which maps to i,j), 
            # U[i*m+j, r] = 1.
            # V[j*p+k, r] = 1 for all k? 
            # If V[j*p+k, r] = 1, then sum_k V[j*p+k, r] B[j,k] = sum_k B[j,k].
            # This is wrong.
            
            # Let's re-derive.
            # C[i,k] = sum_j A[i,j] B[j,k]
            # We want C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
            
            # If we use rank 4 for each row i:
            # For row i, we use 4 terms r = i*4 + j for j=0..3.
            # Term r corresponds to j = r % 4.
            # U[i*m + j, r] = 1.
            # V[j*p + k, r] = 1? 
            # If V[j*p+k, r] = 1, then the inner product is sum_k B[j,k].
            # We need B[j,k] specifically.
            # So V[j*p+k, r] should be 1 if the index matches k?
            # No, V is a matrix. The inner product is sum over the "middle" index?
            # No, the middle index is j.
            # The formula is:
            # Term r contributes: W[i,k,r] * (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k])?
            # No, the seed code uses:
            # U[i*m+j, c]
            # V[j*p+k, c]
            # W[k*n+i, c]
            # This implies the sum is over c.
            # And the "product" is just multiplication of the three values?
            # No, the evaluator likely computes:
            # C[i,k] = sum_c W[k*n+i, c] * U[i*m+j?, c] * V[j*p+k?, c]
            # But U and V depend on j.
            # The standard tensor decomposition is:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # = sum_c W[i,k,c] * (sum_j U[i,j,c] A[i,j]) * (sum_j V[j,k,c] B[j,k])?
            # No, that's not right.
            
            # Let's look at the seed again.
            # U[i*m+j, c] = 1
            # V[j*p+k, c] = 1
            # W[k*n+i, c] = 1
            # This implies that for a specific c, only one j and one k are active?
            # No, c iterates over i,j,k.
            # For a fixed c (i,j,k):
            # U is non-zero only at i,j.
            # V is non-zero only at j,k.
            # W is non-zero only at k,i.
            # So the term is A[i,j] B[j,k] W[k,i,c].
            # This works.
            
            # So for my rank 12 case:
            # I want to group terms.
            # For row i, I want to use 4 terms.
            # Let r = i*4 + j.
            # U[i*m+j, r] = 1.
            # V[j*p+k, r] = 1 for ALL k?
            # If V[j*p+k, r] = 1 for all k, then sum_k V[j*p+k, r] B[j,k] = sum_k B[j,k].
            # This is not B[j,k].
            # So V must be 1 only for the specific k?
            # But V is shared across k?
            # No, V is a matrix of size (m*p, R).
            # The index is j*p+k.
            # So V[j*p+k, r] can be different for different k.
            # If I set V[j*p+k, r] = 1 for all k, it sums them up.
            # I need to isolate B[j,k].
            # So V[j*p+k, r] should be 1 if the term r is meant to pick up B[j,k]?
            # But term r is associated with j.
            # If term r is associated with j, it should pick up the whole row B[j,:].
            # But the output C[i,k] needs B[j,k] specifically.
            # So the "product" must result in B[j,k].
            # If V[j*p+k, r] = 1, and the sum is over k? No, the sum is over the "middle" index j?
            # No, the formula is:
            # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])?
            # If V[j*p+k, r] = delta_{j, j_r}, then sum_j V... = B[j_r, k].
            # So if I set V[j*p+k, r] = 1 for all k, it works IF the sum is not over k.
            # The sum is over j.
            # So if V[j*p+k, r] = 1 for all k, then for a fixed j_r, sum_j V[j*p+k, r] B[j,k] = B[j_r, k].
            # Yes!
            # So:
            # For term r (associated with i, j_r):
            # U[i*m+j_r, r] = 1.
            # V[j_r*p+k, r] = 1 for all k in 0..p-1.
            # W[k*n+i, r] = 1 for all k in 0..p-1.
            
            # Let's check.
            # Term r contributes:
            # W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
            # = 1 * A[i,j_r] * B[j_r, k]
            # Summing over r for fixed i:
            # sum_{j_r=0}^{m-1} A[i,j_r] B[j_r, k] = C[i,k].
            # This works.
            
            # So the construction is:
            # For each i in 0..n-1:
            #   For each j in 0..m-1:
            #     r = i*m + j
            #     U[i*m+j, r] = 1
            #     For each k in 0..p-1:
            #       V[j*p+k, r] = 1
            #       W[k*n+i, r] = 1
            
            # This is exactly the same as the seed, but the seed uses rank n*m*p.
            # Here I am using rank n*m.
            # Wait, r goes from 0 to n*m-1.
            # So rank is n*m = 12.
            # Yes.
            
            pass
# EVOLVE-BLOCK-END
