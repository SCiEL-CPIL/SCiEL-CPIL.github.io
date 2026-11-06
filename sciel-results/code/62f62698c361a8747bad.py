   xs = np.arange(N)
   ys = np.arange(N)
   X, Y = np.meshgrid(xs, ys, indexing="ij")  # [x][y]
   P = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.int64)
   