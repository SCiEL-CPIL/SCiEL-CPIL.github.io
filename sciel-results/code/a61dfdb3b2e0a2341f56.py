T = np.einsum('ar,br,cr->abc', U, V, W)
for i in range(n):
    for k in range(p):
        for j in range(m):
            if T[i * m + j, j * p + k, k * n + i] != 1:
                return False
