Ua = np.asarray(U, dtype=np.float64).reshape(n, m, r)  # (3, 4, 1)
Va = np.asarray(V, dtype=np.float64).reshape(m, p, r)  # (4, 4, 1)
Wa = np.asarray(W, dtype=np.float64).reshape(p, n, r)  # (4, 3, 1)
WaT = Wa.transpose(1, 0, 2)  # (3, 4, 1)

Ue = Ua[:, :, None, :]      # (3, 4, 1, 1)
Ve = Va[None, :, :, :]      # (1, 4, 4, 1)
We = WaT[:, None, :, :]     # (3, 1, 4, 1)

T = np.sum(Ue * Ve * We, axis=3)  # (3, 4, 4)
