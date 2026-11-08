s = np.dot(U_arr[ui, :], V_arr[vi, :] * W_arr[wi, :])
if abs(s - 1.0) > 1e-9:
    return False
