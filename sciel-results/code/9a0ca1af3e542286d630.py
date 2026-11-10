    def try_poly(poly_list):
        nonlocal best_score, best_poly
        # Trim trailing zeros
        while len(poly_list) > 0 and poly_list[-1] == 0:
            poly_list.pop()
            
        if not validate(np.array(poly_list)):
            return
        score = evaluate(np.array(poly_list))
        if score < best_score:
            best_score = score
            best_poly = np.array(poly_list, dtype=np.int64).copy()
