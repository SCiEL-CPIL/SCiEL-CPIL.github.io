            # Periodic coordinate polish every 30 steps
            if es_iter % 30 == 29 and call_count < budget - 20:
                # Quick coordinate polish
                step_sizes = [0.5, 0.1, 0.01]
                x_polished, f_polished, calls_cd = coord_descent(x, fx, 3, step_sizes)
                call_count += calls_cd
                if f_polished < fx:
                    x, fx = x_polished, f_polished
                    if fx < best_phase2_f:
                        best_phase2_f = fx
                        best_phase2_x = x.copy()
