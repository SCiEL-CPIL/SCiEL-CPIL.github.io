<<<<<<< SEARCH
        # Local search
        sol = local_search(sol, seed)
        
        # Simulated annealing for deeper exploration
        if time_left() > 40 and attempts % 3 == 0:
            sol = simulated_annealing(sol, seed + 444)
        
        # Additional swap search
        if time_left() > 30:
            sol = swap_search(sol, seed + 777)
        
        # Final polish
        if time_left() > 20:
            sol = polish_pass(sol, seed + 888)
=======
        # Local search with 2-swap moves
        sol = local_search(sol, seed)
        
        # Simulated annealing for deeper exploration
        if time_left() > 40 and attempts % 3 == 0:
            sol = simulated_annealing(sol, seed + 444)
        
        # Additional swap search
        if time_left() > 30:
            sol = swap_search(sol, seed + 777)
        
        # Final polish with 2-swap
        if time_left() > 20:
            sol = polish_pass(sol, seed + 888)
        
        # Extra local search pass after polish
        if time_left() > 15:
            sol = local_search(sol, seed + 1234)
>>>>>>> REPLACE
