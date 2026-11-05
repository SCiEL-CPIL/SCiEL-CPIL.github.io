<<<<<<< SEARCH
        # Solve
        # time_limit is in seconds
        milp_time_limit = max(1.0, deadline - time.time() - 5.0)
        
        res = milp(c=c_obj, constraints=constraint, integrality=integrality, bounds=bounds, options={"time_limit": milp_time_limit, "presolve": "default"})
=======
        # Solve
        # time_limit is in seconds
        milp_time_limit = max(1.0, deadline - time.time() - 5.0)
        
        res = milp(c=c_obj, constraints=constraint, integrality=integrality, bounds=bounds, 
                   options={"time_limit": milp_time_limit, "presolve": "default", "start": warm_start})
>>>>>>> REPLACE

