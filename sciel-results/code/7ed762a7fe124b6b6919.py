        sum_w = 0.0
        for v in node_list:
            sum_w += weights[v]
        if r_weight + sum_w <= best_weight:
            return
