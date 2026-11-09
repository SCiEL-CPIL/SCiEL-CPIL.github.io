            # Update p_c
            h_sigma = np.linalg.norm(p_sigma) < 1.63
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * sigma * z_w
            
            # Update C
            C_new = (1 - c_1 - c_mu) * C
            C_new += c_1 * (p_c @ p_c.T)
            if not h_sigma:
                C_new += c_1 * (1 - h_sigma) * c_c * (2 - c_c) * C
            C_new += c_mu * (sigma ** 2) * np.sum(weights[:mu, None, None] * (z_parents[:, None, :] * z_parents[None, :, :]), axis=0)
