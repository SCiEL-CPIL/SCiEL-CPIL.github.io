result = minimize(objective, current, method='Nelder-Mead',
                  options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-10,
                           'initial_simplex': current + scale * np.eye(3)})
