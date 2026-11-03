# For the delta computation, precompute:
# b_old_pi = b[old_pi, :]  (row of b at old_pi)
# b_old_pj = b[old_pj, :]  (row of b at old_pj)
# Then the sum over m != i,j of a[i,m]*(b[old_pj,p[m]] - b[old_pi,p[m]])
# = sum_m a[i,m]*(b[old_pj,p[m]] - b[old_pi,p[m]]) - a[i,i]*(b[old_pj,p[i]] - b[old_pi,p[i]]) - a[i,j]*(b[old_pj,p[j]] - b[old_pi,p[j]])
# = sum_m a[i,m]*(b[old_pj,p[m]] - b[old_pi,p[m]]) - a[i,i]*(b[old_pj,old_pi] - b[old_pi,old_pi]) - a[i,j]*(b[old_pj,old_pj] - b[old_pi,old_pj])
