        if np.allclose(T_check, T_exp, atol=1e-5):
            # Polish and round
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            T_round = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
            if np.allclose(T_round, T_exp, atol=1e-9):
                return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
            # Else, this candidate is invalid due to integrality constraint
            # Do not return it. Continue searching or fallback.
