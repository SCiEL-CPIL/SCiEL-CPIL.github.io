    def compute_kernel(X1, X2, hyperparams, add_noise=False):
        log_sigma_f, log_l, log_sigma_n = hyperparams
        sigma_f = np.exp(log_sigma_f)
        l = np.exp(log_l)
        sigma_n = np.exp(log_sigma_n)
        
        dists = cdist(X1, X2, metric='sqeuclidean')
        d = np.sqrt(dists) # Euclidean distance
        sqrt5_d = np.sqrt(5.0) * d / l
        K_matern = sigma_f**2 * (1.0 + sqrt5_d + (5.0/3.0)*sqrt5_d**2) * np.exp(-sqrt5_d)
        
        # Linear kernel
        K_lin = X1 @ X2.T * sigma_f**2
        
        K = K_matern + K_lin
        
        if add_noise:
            K += sigma_n**2 * np.eye(X1.shape[0])
            
        return K
