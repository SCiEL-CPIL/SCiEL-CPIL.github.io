# BCLR Rank 21 decomposition for 3x3 matrix multiplication
# Terms are (u, v, w) where u, v, w are 9-vectors (flattened 3x3 matrices)
# u corresponds to A, v to B, w to C
# The product is sum(u_i * v_i * w_i) for each term? No.
# The tensor is sum_r u_r \otimes v_r \otimes w_r.
# So C = sum_r (u_r^T A) (v_r^T B) w_r? No.
# C_{ij} = sum_k sum_r U_{ik}^{(r)} V_{kj}^{(r)} W_{ij}^{(r)}?
# In the global decomposition, we just add the contributions.
