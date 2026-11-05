        def ir_constraint(x, i=i, v1_i=v1_i, v2_i=v2_i):
            q1_i = x[i]
            q2_i = x[n_types + i]
            p_i = x[2 * n_types + i]
            return (v1_i * q1_i + v2_i * q2_i) - p_i
