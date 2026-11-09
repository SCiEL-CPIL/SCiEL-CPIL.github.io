    # Trim primes if necessary
    def trim_set(p_set):
        s = sum(Fraction(1, p) for p in p_set)
        if s > 1:
            # Sort by p descending to remove expensive? No, remove large p (cheap) first?
            # We want to keep small p (expensive) if possible?
            # Actually, to minimize loss of coverage, we might want to keep small p.
            # So remove large p first.
            p_set_sorted = sorted(p_set, reverse=True)
            new_set = []
            s = Fraction(0)
            # Add small primes first
            for p in sorted(p_set):
                c = Fraction(1, p)
                if s + c <= 1:
                    new_set.append(p)
                    s += c
            return new_set
        return p_set
