            def strassen_weighted(a_pos, b_pos, c_pos_weights):
                # a_pos, b_pos: list of 4 lists of positions (weight 1 implied)
                # c_pos_weights: list of 4 lists of (pos, weight)
                result = []
                # ... generate 7 terms ...
                # For each term, construct u, v, w
                # u: sum of a_pos[i] with appropriate signs from Strassen formula
                # v: sum of b_pos[i] with appropriate signs
                # w: sum of c_pos_weights[i] with appropriate signs
            