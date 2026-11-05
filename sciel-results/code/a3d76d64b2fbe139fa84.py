def _verify(U, V, W, n, m, p, trials=20, tol=1e-6):
    """Check that the decomposition computes C[i,k] = sum_j A[i,j] B[j,k]."""
    rank = U.shape[1]
    rng = np.random.RandomState(0)
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        C = np.zeros((n, p))
        for t in range(rank):
            a = U[:, t].reshape(n, m)
            b = V[:, t].reshape(m, p)
            w = W[:, t].reshape(p, n)
            C += np.einsum('ij,jk,ki->ik', a, b, w)
        if np.max(np.abs(C - C_expected)) > tol:
            return False
    return True
