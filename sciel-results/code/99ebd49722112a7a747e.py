            if r % 3 == 0:
                # Pure random
                U = np.random.randn(n*m, R)
                V = np.random.randn(m*p, R)
                W = np.random.randn(p*n, R)
            elif r % 3 == 1:
                # Structured: initialize with some pattern
                U = np.random.randn(n*m, R) * 0.5 + 0.5
                V = np.random.randn(m*p, R) * 0.5 + 0.5
                W = np.random.randn(p*n, R) * 0.5 + 0.5
            else:
                # Sparse-like initialization
                U = np.random.randn(n*m, R)
                V = np.random.randn(m*p, R)
                W = np.random.randn(p*n, R)
                # Zero out some entries to encourage sparsity
                mask_u = np.random.rand(n*m, R) > 0.5
                mask_v = np.random.rand(m*p, R) > 0.5
                mask_w = np.random.rand(p*n, R) > 0.5
                U = U * mask_u
                V = V * mask_v
                W = W * mask_w
