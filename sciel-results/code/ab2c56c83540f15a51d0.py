        max_rest_float = 0.0
        for i in range(terms_left - 1):
            max_rest_float += 1.0 / (d + 1 + i)
            
        if float(new_target) > max_rest_float:
            break
