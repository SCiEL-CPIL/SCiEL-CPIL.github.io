from scipy.sparse import csr_matrix

# Inside construct
rows = []
cols = []
for i in range(N):
    for j in neighbors[i]:
        rows.append(i)
        cols.append(j)

M_sparse = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(N, N))

# In greedy
gains = M_sparse.dot(uncovered)
