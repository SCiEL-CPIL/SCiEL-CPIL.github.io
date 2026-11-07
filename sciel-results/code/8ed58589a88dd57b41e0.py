<<<<<<< SEARCH
def _basin_hopping(positions, kind, n_exp, m_exp, c, rng, max_iterations=50):
    """Simple basin hopping: perturb, relax, keep best."""
    best_energy, best_positions = sutton_chen_energy_gradient(positions, kind)
    current_energy, current_positions = best_energy, best_positions.copy()
    
    for it in range(max_iterations):
        if _time_left() < 10.0:
            break
        
        # Perturb
        perturbation_scale = 0.1 + 0.3 * (it / max_iterations)
        perturbed = current_positions + rng.normal(0, perturbation_scale, current_positions.shape)
        
        # Relax
        energy, relaxed = relax(perturbed, 'sutton_chen', kind)
        
        if energy < best_energy:
            best_energy, best_positions = energy, relaxed
        current_positions = relaxed
    
    return best_energy, best_positions
=======
def _bfgs_refine(positions, kind, n_exp, m_exp, c):
    """BFGS refinement after L-BFGS-B to escape shallow local minima."""
    try:
        def fun_and_grad(x):
            pos = x.reshape(-1, 3)
            e, g = sutton_chen_energy_gradient(pos, kind)
            return e, g.ravel()
        
        res = minimize(fun_and_grad, positions.ravel(), method='BFGS',
                       jac=True, options={'maxiter': 50, 'gtol': 1e-6})
        if res.success or res.fun < sutton_chen_energy_gradient(positions, kind)[0]:
            return res.fun, res.x.reshape(-1, 3)
    except:
        pass
    return sutton_chen_energy_gradient(positions, kind)


def _basin_hopping(positions, kind, n_exp, m_exp, c, rng, max_iterations=50):
    """Basin hopping: perturb, relax, optionally BFGS refine, keep best."""
    best_energy, best_positions = sutton_chen_energy_gradient(positions, kind)
    current_positions = best_positions.copy()
    
    for it in range(max_iterations):
        if _time_left() < 10.0:
            break
        
        perturbation_scale = 0.08 + 0.25 * (it / max_iterations)
        perturbed = current_positions + rng.normal(0, perturbation_scale, current_positions.shape)
        
        energy, relaxed = relax(perturbed, 'sutton_chen', kind)
        
        # BFGS refinement every 3rd iteration
        if it % 3 == 2 and _time_left() > 15.0:
            try:
                e2, p2 = _bfgs_refine(relaxed, kind, n_exp, m_exp, c)
                if e2 < energy:
                    energy, relaxed = e2, p2
            except:
                pass
        
        if energy < best_energy:
            best_energy, best_positions = energy, relaxed
        current_positions = relaxed
    
    return best_energy, best_positions
>>>>>>> REPLACE

