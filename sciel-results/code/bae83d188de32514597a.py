inn = np.array([i for i in range(47) if (mask>>i)&1], dtype=np.int64)
out = np.array([i for i in range(47) if not (mask>>i)&1], dtype=np.int64)
a_arr = np.array(a, dtype=object)  # big ints, can't use numpy int64
