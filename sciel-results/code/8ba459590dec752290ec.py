def snap_to_half_integers(M, tol=1e-5):
    # Check if M is close to half-integers
    # M * 2 should be close to integers
    M2 = M * 2
    diff = M2 - np.round(M2)
    if np.all(np.abs(diff) < tol):
        return np.round(M2) / 2.0
    return M
