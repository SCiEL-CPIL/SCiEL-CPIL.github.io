def als_step_U(V, W, r):
    A = np.einsum('br,cr->bcr', V, W).reshape(-1, r)  # (mp*pn, r)
    B = T.reshape(nm, -1)  # (nm, mp*pn)
    # U @ A.T = B  =>  U = lstsq(A.T, B.T).T
    U = np.linalg.lstsq(A.T, B.T, rcond=None)[0].T
    return U

def als_step_V(U, W, r):
    A = np.einsum('ar,cr->acr', U, W).reshape(-1, r)  # (nm*pn, r)
    B = np.transpose(T, (1, 0, 2)).reshape(mp, -1)  # (mp, nm*pn)
    V = np.linalg.lstsq(A.T, B.T, rcond=None)[0].T
    return V

def als_step_W(U, V, r):
    A = np.einsum('ar,br->abr', U, V).reshape(-1, r)  # (nm*mp, r)
    B = np.transpose(T, (2, 0, 1)).reshape(pn, -1)  # (pn, nm*mp)
    W = np.linalg.lstsq(A.T, B.T, rcond=None)[0].T
    return W
