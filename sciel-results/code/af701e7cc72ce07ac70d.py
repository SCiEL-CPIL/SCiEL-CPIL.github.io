# For 3x2 * 2x2 -> 3x2, we need 6 outputs
# Each output C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
# That's 6 dot products of length 2, so minimum rank is 6 (if we could share)
# But since each uses different A[i,j] and B[j,k], the minimum is actually 6
# We can do it in 6 terms: one for each (i,j,k) triple
# But we can share: for fixed i, C[i,0] and C[i,1] share A[i,0] and A[i,1]
# So for each i, we can compute A[i,0]*B[0,:] and A[i,1]*B[1,:] in 2 terms
# Total: 3 rows * 2 terms = 6 terms
