            cost_float = 1.0 / p
            if float(new_budget) - cost_float < 0:
                continue
            cost = Fraction(1, p)
            if new_budget - cost < 0:
                continue
