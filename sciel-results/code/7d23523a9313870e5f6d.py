def _try_rank13():
    # ...
    for r_drop in range(14):
        # ...
        if res.cost < 1e-6:
            U_sol = res.x[:U_size * R].reshape(U_size, R)
            V_sol = res.x[U_size * R : U_size * R + V_size * R].reshape(V_size, R)
            W_sol = res.x[U_size * R + V_size * R :].reshape(W_size, R)
            
            res_rounded = _round_and_verify(U_sol, V_sol, W_sol, R)
            if res_rounded is not None:
                return res_rounded
    # ...
