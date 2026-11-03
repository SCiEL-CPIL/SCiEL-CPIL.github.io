# C sub-block: k=4, i in [0,1]
for i_local in range(2):
    c_local = i_local  # 2x1: index is just row (k*n+i with k=0 local)
    c_global = 4 * 2 + i_local
