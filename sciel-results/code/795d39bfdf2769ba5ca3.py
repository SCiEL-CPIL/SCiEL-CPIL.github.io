<<<<<<< SEARCH
        rng = np.random.default_rng(2024)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        evaluated = 0

        def eval_point(x):
            nonlocal evaluated, best_x, best_f
            x = clip(x)
            f = func(x)
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
=======
        rng = np.random.default_rng(2024)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        def eval_point(x):
            nonlocal evaluated, best_x, best_f
            x = clip(x)
            f = func(x)
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
>>>>>>> REPLACE

Finally, ensure the return statement is safe.

