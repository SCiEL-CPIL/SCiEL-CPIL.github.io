def _flip(s, A, i):
    si = s[i]
    j1 = i + _KS
    j2 = i - _KS
    t1 = np.where((j1 >= 0) & (j1 < N), s[np.clip(j1, 0, N - 1)], 0.0)
    t2 = np.where((j2 >= 0) & (j2 < N), s[np.clip(j2, 0, N - 1)], 0.0)
    upd = (t1 + t2)
    upd[N - 1] = 0.0  # k=0 term unchanged
    A += (-2.0 * si) * upd
    s[i] = -si
