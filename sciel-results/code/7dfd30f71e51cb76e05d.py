# c_vals: (n_frac,)
# m: scalar
# m_pre: (n_frac,) = R * (m - c_vals)

cont = np.zeros(n_frac)
for zp in range(Z):
    p = transition[z, zp]
    if p < 1e-15:
        continue
    # m_prime for this z' and all c_vals
    m_prime_vec = m_pre + income[zp] # shape (n_frac,)
    m_prime_clipped = np.clip(m_prime_vec, fine_grid[0], fine_grid[-1])
    v_interp = np.interp(m_prime_clipped, fine_grid, V[t + 1, zp, :]) # shape (n_frac,)
    cont += p * v_interp
