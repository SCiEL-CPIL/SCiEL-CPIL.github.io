<<<<<<< SEARCH
def _compute_error(a_list, b_list, A, B, t_values):
    """Compute mean spectral norm error over t_values."""
    dim = A.shape[0]
    H = A + B
    errors = []
    for t in t_values:
        U_exact = np.linalg.matrix_exp(-1j * t * H)
        U_prod = np.eye(dim, dtype=complex)
        for k in range(len(a_list)):
            U_prod = U_prod @ np.linalg.matrix_exp(-1j * b_list[k] * t * B)
            U_prod = U_prod @ np.linalg.matrix_exp(-1j * a_list[k] * t * A)
        err = np.linalg.norm(U_prod - U_exact, 2)
        errors.append(err)
    return np.mean(errors)
=======
def _compute_error(a_list, b_list, A, B, t_values):
    """Compute mean spectral norm error over t_values."""
    dim = A.shape[0]
    H = A + B
    errors = []
    for t in t_values:
        U_exact = expm(-1j * t * H)
        U_prod = np.eye(dim, dtype=complex)
        for k in range(len(a_list)):
            U_prod = U_prod @ expm(-1j * b_list[k] * t * B)
            U_prod = U_prod @ expm(-1j * a_list[k] * t * A)
        err = np.linalg.norm(U_prod - U_exact, 2)
        errors.append(err)
    return np.mean(errors)
>>>>>>> REPLACE

