    best_state = {'x': x0.copy(), 'val': objective(x0)}
    
    def callback(xk):
        val = objective(xk)
        if val < best_state['val']:
            best_state['x'] = xk.copy()
            best_state['val'] = val
        if time.time() > deadline:
            raise TimeoutError()

    try:
        res = scipy.optimize.minimize(
            objective, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True},
            callback=callback
        )
        # If it finished normally, res.x is the best
        if res.fun < best_state['val']:
             best_state['x'] = res.x
             best_state['val'] = res.fun
    except TimeoutError:
        pass
    except Exception:
        pass
