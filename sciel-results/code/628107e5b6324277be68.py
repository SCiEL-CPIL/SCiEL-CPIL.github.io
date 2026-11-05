    def _greedy_fill_value(remaining_capacity, remaining_weights, remaining_values, order, exclude_idx=None):
        if remaining_capacity <= 0:
            return 0.0
        total_val = 0.0
        cap = remaining_capacity
        for idx in order:
            if idx == exclude_idx:
                continue
            w = remaining_weights[idx]
            if w <= cap:
                cap -= w
                total_val += remaining_values[idx]
                if cap <= 0:
                    break
        return total_val
    