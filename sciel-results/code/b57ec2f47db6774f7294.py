    def compute_jacobian(t_val, y_val):
        eps = 1e-8
        f_base = f(t_val, y_val)
        J = np.zeros((2, 2))
        
        y_pert = y_val.copy()
        y_pert[0] += eps
        f_pert = f(t_val, y_pert)
        J[:, 0] = (f_pert - f_base) / eps
        
        y_pert = y_val.copy()
        y_pert[1] += eps
        f_pert = f(t_val, y_pert)
        J[:, 1] = (f_pert - f_base) / eps
        return J
