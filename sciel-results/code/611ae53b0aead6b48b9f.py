<<<<<<< SEARCH
            res = minimize(
                objective, 
                v0, 
                method='SLSQP', 
                constraints=cons, 
                options={'maxiter': 100, 'ftol': 1e-8}
            )
=======
            res = minimize(
                objective, 
                v0, 
                method='SLSQP', 
                constraints=cons, 
                options={'maxiter': 500, 'ftol': 1e-10}
            )
>>>>>>> REPLACE
