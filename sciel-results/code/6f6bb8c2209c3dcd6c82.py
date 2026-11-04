# Build the tensor evaluation matrix
# For each (i,j,k), compute the vector [T_r(i,j,k) for r in range(98)]
# This gives a 128 x 98 matrix
# If its rank is < 98, we can reduce

# Actually, we need to be more careful. The condition is:
# sum_r U[i*4+j, r] * V[j*8+k, r] * W[k*4+i, r] = 1 for all (i,j,k)
# 
# This means the sum of the 98 rank-1 tensors equals the all-ones tensor.
# A linear dependency sum_r c_r * T_r = 0 means we can replace the set of terms
# with a different set that has fewer terms, IF we can express the removed terms
# as combinations of remaining ones.
#
# More precisely: if T_s = sum_{r != s} c_r * T_r, then we can remove term s
# and adjust... no, that's not how it works. We can't just remove a term unless
# its coefficient can be set to 0.
#
# Actually, for tensor rank, the terms are not independent in that way.
# The correct statement: the rank of the tensor is the minimum number of rank-1
# tensors that sum to it. If we have a representation with 98 terms, and there's
# a linear dependency among the 98 rank-1 tensors, it doesn't directly mean we
# can reduce the count.
#
# However, if one of the rank-1 tensors is in the span of the others, we CAN
# replace it. Specifically, if T_0 = sum_{r=1}^{97} c_r * T_r, then we can
# write the sum as: sum_{r=1}^{97} (1 + c_r) * T_r, which uses only 97 terms.
#
# So the algorithm is:
# 1. Build the 128 x 98 matrix M where M[idx, r] = T_r(idx)
# 2. Find the null space of M (or rather, find linear dependencies)
# 3. For each null vector c, we have sum_r c_r * T_r = 0
# 4. Pick a term s with c_s != 0, and rewrite: T_s = -sum_{r != s} (c_r/c_s) * T_r
# 5. This means the original sum sum_r T_r = sum_{r != s} (1 + c_r/c_s) * T_r + T_s
#    = sum_{r != s} (1 + c_r/c_s) * T_r + (-sum_{r != s} (c_r/c_s) * T_r)
#    = sum_{r != s} T_r
# Wait, that's not right. Let me redo this.
#
# We have: sum_{r=0}^{97} T_r = M (the target tensor)
# And: sum_r c_r * T_r = 0 (linear dependency)
# So: T_0 = -sum_{r=1}^{97} (c_r/c_0) * T_r (assuming c_0 != 0)
# Then: sum_{r=0}^{97} T_r = T_0 + sum_{r=1}^{97} T_r
#   = -sum_{r=1}^{97} (c_r/c_0) * T_r + sum_{r=1}^{97} T_r
#   = sum_{r=1}^{97} (1 - c_r/c_0) * T_r
# This uses 97 terms! So we can reduce by 1.
#
# But wait, this only works if we can absorb the coefficients. The new "terms" are
# (1 - c_r/c_0) * T_r for r = 1..97. Each is still a rank-1 tensor (just scaled),
# so this is valid!
#
# So the algorithm is:
# 1. Build M (128 x 98)
# 2. Find null space vectors
# 3. For each null vector, pick a nonzero component s
# 4. Remove term s and scale the others
# 5. Repeat until no more dependencies
