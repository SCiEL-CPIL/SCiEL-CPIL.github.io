# For (2,2,p), use rank = p + 6
# Terms 0-5: shared Strassen terms
# Terms 6 to 6+p-1: per-column terms

# U has shape (4, rank), V has shape (2p, rank), W has shape (2p, rank)
# U[i*m+j, r] where i,j in {0,1}, so rows 0-3
# V[j*p+k, r] where j in {0,1}, k in {0,...,p-1}, so rows 0-2p-1
# W[k*n+i, r] where k in {0,...,p-1}, i in {0,1}, so rows 0-2p-1
