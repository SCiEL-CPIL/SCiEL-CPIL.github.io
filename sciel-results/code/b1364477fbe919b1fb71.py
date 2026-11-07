            x_opt, f_opt, calls_used = self._nelder_mead(
                func, x0, fx0, per_start_budget, rng, deadline
            )
            total_calls += calls_used
