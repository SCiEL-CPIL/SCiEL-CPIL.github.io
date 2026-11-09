        mask_eff = np.zeros(MAX_K + 1, dtype=bool)
        mask_eff[:K_eff+1] = True
        mask_eff &= (mu != 0)
        
        domain = np.nonzero(mask_eff)[0]
        domain = domain[domain > 0]
        values = mu[domain]
