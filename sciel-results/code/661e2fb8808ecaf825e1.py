    all_vectors = np.array([np.array([ (i >> k) & 1 for k in range(N) ]) for i in range(2**N)], dtype=np.uint8)
