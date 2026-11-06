            if s in targets_set_2:
                if a % b != 0 and b % a != 0:
                    best_rep[s] = (a, b)
                    targets_set_2.remove(s)
