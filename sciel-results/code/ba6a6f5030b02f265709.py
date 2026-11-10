<<<<<<< SEARCH
    def max_tail(lo, hi):
        # upper bound on sum_{d=lo}^{hi} 1/d
        if hi < lo:
            return Fraction(0, 1)
        if hi == lo:
            return Fraction(1, lo)
        return Fraction(1, lo) + Fraction(math.log(hi / lo), 1)  # approx; as float
=======
    def max_tail(lo, hi):
        # upper bound on sum_{d=lo}^{hi} 1/d
        if hi < lo:
            return 0.0
        if hi == lo:
            return 1.0 / lo
        return (1.0 / lo) + math.log(hi / lo)
>>>>>>> REPLACE
