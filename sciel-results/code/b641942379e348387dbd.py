if hsig == 0:
    sigma_update_term = 0
else:
    sigma_update_term = norm_ps / hsig - 1
sigma *= np.exp(c_sigma * sigma_update_term)
