    def run_greedy(order):
        chosen = []
        pairs = [] # list of (a, b, a&b)
        for c in order:
            is_valid = True
            for a, b, inter_ab in pairs:
                if (c & a == inter_ab) and (c & b == inter_ab):
                    is_valid = False
                    break
            if is_valid:
                chosen.append(c)
                # Update pairs with new element c
                for a in chosen[:-1]: # all previously chosen
                    pairs.append((c, a, c & a))
        return chosen
