<<<<<<< SEARCH
    for i in range(N):
        wi = w_vals[i]
        # Constraint 1: wi*(c0*M0_i + c4*M4_i + c8*M8_i) - t <= 0
        # c0*(wi*M0_i) + c4*(wi*M4_i) + c8*(wi*M8_i) - t <= 0
        A_ub.append([wi*M0_vals[i], wi*M4_vals[i], wi*M8_vals[i], -1.0])
        b_ub.append(0.0)
        
        # Constraint 2: -(wi*(c0*M0_i + c4*M4_i + c8*M8_i)) - t <= 0
        # -c0*(wi*M0_i) - c4*(wi*M4_i) - c8*(wi*M8_i) - t <= 0
        A_ub.append([-wi*M0_vals[i], -wi*M4_vals[i], -wi*M8_vals[i], -1.0])
        b_ub.append(0.0)
=======
    for i in range(N):
        wi = w_vals[i]
        # Constraint 1: wi*(c0*M0_i + c4*M4_i + c8*M8_i) - t <= 0
        # c0*(wi*M0_i) + c4*(wi*M4_i) + c8*(wi*M8_i) - t <= 0
        A_ub.append([wi*M0[i], wi*M4[i], wi*M8[i], -1.0])
        b_ub.append(0.0)
        
        # Constraint 2: -(wi*(c0*M0_i + c4*M4_i + c8*M8_i)) - t <= 0
        # -c0*(wi*M0_i) - c4*(wi*M4_i) - c8*(wi*M8_i) - t <= 0
        A_ub.append([-wi*M0[i], -wi*M4[i], -wi*M8[i], -1.0])
        b_ub.append(0.0)
>>>>>>> REPLACE
