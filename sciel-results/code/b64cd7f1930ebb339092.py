<<<<<<< SEARCH
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={"presolve": "on", "dual_feasibility_tolerance": 1e-12,
                               "primal_feasibility_tolerance": 1e-12,
                               "time_limit": 40})
        
        if res.success:
            x = res.x
            allocations = np.zeros((n_types, n_goods))
            utilities = np.zeros(n_types)
            for i in range(n_types):
                for g in range(n_goods):
                    allocations[i, g] = x[var_x(i, g)]
                utilities[i] = x[var_U(i)]
            
            # Compute payments: p_i = v_i . x_i - U_i
            payments = np.zeros(n_types)
            for i in range(n_types):
                vi = np.array(types[i], dtype=float)
                xi = np.array(allocations[i], dtype=float)
                p_i = np.dot(vi, xi) - utilities[i]
                payments[i] = max(0.0, p_i)  # Ensure non-negative
            
            return allocations, payments
        else:
            return None
    except Exception:
        return None
=======
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={"presolve": "on", "dual_feasibility_tolerance": 1e-10,
                               "primal_feasibility_tolerance": 1e-10,
                               "time_limit": 40})
        
        if res.success:
            x = res.x
            allocations = np.zeros((n_types, n_goods))
            utilities = np.zeros(n_types)
            for i in range(n_types):
                for g in range(n_goods):
                    allocations[i, g] = x[var_x(i, g)]
                utilities[i] = x[var_U(i)]
            
            # Compute payments: p_i = v_i . x_i - U_i
            payments = np.zeros(n_types)
            for i in range(n_types):
                vi = np.array(types[i], dtype=float)
                xi = np.array(allocations[i], dtype=float)
                p_i = np.dot(vi, xi) - utilities[i]
                payments[i] = max(0.0, p_i)  # Ensure non-negative
            
            return allocations, payments
        else:
            return None
    except Exception:
        return None
>>>>>>> REPLACE

