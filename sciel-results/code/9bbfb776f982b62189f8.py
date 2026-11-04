<<<<<<< SEARCH
                # Penalty-based optimization (fast)
                if time.time() < deadline - 20.0:
                    optimized = _penalty_optimize(circles, ww, hh, deadline)
                    if optimized is not None:
                        circles = optimized
                        current_value = _total_radius(circles)
                
                # SLSQP polish (slower but more precise)
                if time.time() < deadline - 10.0:
                    polished = _slsqp_polish(circles, ww, hh, deadline)
                    if polished is not None:
                        circles = polished
                        current_value = _total_radius(circles)
=======
                # Penalty-based optimization (fast)
                if time.time() < deadline - 25.0:
                    optimized = _penalty_optimize(circles, ww, hh, deadline)
                    if optimized is not None:
                        circles = optimized
                        current_value = _total_radius(circles)
                
                # SLSQP polish (slower but more precise)
                if time.time() < deadline - 18.0:
                    polished = _slsqp_polish(circles, ww, hh, deadline)
                    if polished is not None:
                        circles = polished
                        current_value = _total_radius(circles)
>>>>>>> REPLACE
