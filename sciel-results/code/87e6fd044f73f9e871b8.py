# Rank 23 Decomposition
# Indices 0..8 for 3x3 matrices
# A: 0,1,2 (row 0), 3,4,5 (row 1), 6,7,8 (row 2)
# B: 0,1,2 (row 0), 3,4,5 (row 1), 6,7,8 (row 2)
# C: 0,1,2 (col 0? No, W[k*n+i] -> k is row, i is col? 
# Prompt: W[k*n+i]. Usually C[i][j] is i*3+j.
# If W[k*n+i], then index is k*3+i.
# In standard C[i][j], i is row, j is col.
# So W index = row*3 + col.
# Let's assume C[i][j] = sum U[i,j,k] V[j,k,l] W[l,i,k]? No.
# Tensor T_{i,j,k} = A_{i,j} B_{j,k} C_{k,i}? No.
# T_{i,j,k} = A_{i,j} B_{j,k} C_{i,k}.
# U[i,j] * V[j,k] * W[i,k] = C[i,k].
# So W index is i*3+k? Or k*3+i?
# Prompt: W[k*n+i]. So W index = k*3+i.
# This implies W[k,i] corresponds to C[i,k]? Or C[k,i]?
# If W[k*n+i] is the coefficient for C[i,k], then the index is k*3+i.
# Let's verify with schoolbook:
# C[i,k] = sum_j A[i,j] B[j,k].
# Term (i,j,k): U[i,j]=1, V[j,k]=1, W[?,?]=1.
# Current code: W[k*n+i] = 1.
# So W index is k*3+i.
# This means W[k,i] = 1.
# So the output matrix C has C[i,k] = sum_k W[k,i] * (product)?
# No, the decomposition is $C = \sum_r U_r V_r W_r^T$?
# The tensor is $\sum_r u_r \otimes v_r \otimes w_r$.
# $C_{i,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$?
# The indices in U are $(i,j)$, V are $(j,k)$, W are $(k,i)$.
# So $C_{i,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$.
# This matches the schoolbook implementation.
