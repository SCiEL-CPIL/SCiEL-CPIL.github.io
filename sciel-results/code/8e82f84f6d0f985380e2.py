    def quick_residual(U, V, W, sample_pairs):
        R = U.shape[1]
        U_mat = U.reshape(n, m, R)
        V_mat = V.reshape(m, p, R)
        W_mat = W.reshape(p, n, R)
        err = 0.0
        eye = np.eye(m)
        for (i, k) in sample_pairs:
            Wki = W_mat[k, i, :]
            Ui = U_mat[i, :, :]
            Vk = V_mat[:, k, :]
            A = Ui * Wki[None, :]
            Mik = A.T @ Vk
            diff = Mik - eye
            err += float(np.sum(diff * diff))
        return err
