# The 23 rank-1 components for 3x3x3 matrix multiplication
# U, V, W are 3x3 matrices
# T = sum(U_r \otimes V_r \otimes W_r)
# This means C_ik = sum_r W_r[i,k] * (sum_ij U_r[i,j] A_ij) * (sum_jk V_r[j,k] B_jk)
