<<<<<<< SEARCH
            # Improve with 2-opt
            # Reduce passes if n is large to save time
            passes = 10 if n_unvisited <= 30 else 5
            sub_tour = _two_opt_improve(sub_tour, dm, first_node, dest, max_passes=passes)
=======
            # Improve with 2-opt - vectorized version allows more passes
            passes = 15 if n_unvisited <= 25 else 10
            sub_tour = _two_opt_improve(sub_tour, dm, first_node, dest, max_passes=passes)
>>>>>>> REPLACE

