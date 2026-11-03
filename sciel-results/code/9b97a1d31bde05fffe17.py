# Target is 1 if indices form a cycle?
# The code iterates i, j, k.
# The term is U[i,j] V[j,k] W[k,i].
# This term corresponds to the tensor entry T[(i,j), (j,k), (k,i)].
# For the matrix multiplication tensor, this entry is 1.
# So s should be 1.
