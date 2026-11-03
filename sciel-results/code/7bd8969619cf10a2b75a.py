import numpy as np
np.random.seed(0)
A = np.random.rand(6, 6)
B = np.random.rand(6, 6)
C = A @ B
# Check decomposition
U_np = np.array(U)
V_np = np.array(V)
W_np = np.array(W)
# C_recon[i,j] = sum_k U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] ?
# Wait, the convention is W[k*n+i, r] in the prompt?
# "output factor follows the official trace convention W[k*n+i,r]"
# Usually W is (p*n, rank).
# The reconstruction is:
# C = sum_r (U[:,r] * V[:,r]) * W[:,r] ? No.
# Tensor decomposition: T = sum_r u_r \otimes v_r \otimes w_r.
# T[i, k, j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
# Here T[i,k,j] corresponds to C[i,j] = sum_k A[i,k] B[k,j].
# So we need: sum_k A[i,k] B[k,j] = sum_r U[i*m+k, r] V[k*p+j, r] W[j*n+i, r]
# This implies W is indexed by (j, i) in the output?
# Let's check the prompt: "W[k*n+i,r]". This is ambiguous.
# Usually W is just a vector for each term, or a matrix.
# In standard tensor decomposition for matmul:
# U: (n*m, rank)
# V: (m*p, rank)
# W: (p*n, rank)
# The term is U[:,r] \otimes V[:,r] \otimes W[:,r].
# The sum is over r.
# The resulting tensor is T[i, k, j] = sum_r U[i*m+k, r] V[k*p+j, r] W[j*n+i, r].
# We want T[i, k, j] to be 1 if we are computing C[i,j] using A[i,k]B[k,j]?
# No, the tensor T is the multiplication tensor.
# T[i, k, j] = 1 if the product A[i,k]B[k,j] contributes to C[i,j].
# Actually, the decomposition is of the tensor that defines the bilinear map.
# The map is (A, B) -> C.
# C[i,j] = sum_k A[i,k] B[k,j].
# This can be written as C = sum_r (u_r(A)) (v_r(B)) w_r(C)? No.
# It is C = sum_r u_r(A) \otimes v_r(B) \otimes w_r^* where w_r is a linear functional on C?
# No, the standard form is:
# C_{ij} = \sum_r U_{i, m+k, r} V_{k, p+j, r} W_{j, n+i, r} ?
# Let's look at the seed code:
# `W` is `(p*n, rank)`.
# The loop does: `W[w_row][r_idx] += gamma`.
# `w_row = j_glob * n + i_glob`.
# So W is indexed by `j*n + i`.
# This matches the convention `W[k*n+i, r]` if we identify the first index with `j` and second with `i`?
# The prompt says `W[k*n+i,r]`. This likely means row index is `k*n+i`.
# In the seed code, `w_row = j_glob * n + i_glob`. So `k` in prompt is `j` in code?
# Or `i` in prompt is `i` in code?
# Let's assume the seed code's indexing is correct because it scored 1.1429.
# The seed code uses `w_row = j_glob * n + i_glob`.
# And `U[u_row]` where `u_row = i_glob * m + k_glob`.
# And `V[v_row]` where `v_row = k_glob * p + j_glob`.

# So the check is:
# For all i, j:
# C[i,j] == sum_k sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
