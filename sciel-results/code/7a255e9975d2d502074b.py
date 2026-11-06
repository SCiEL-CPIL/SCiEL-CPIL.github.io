    for p in primes_all:
        cost = Fraction(1, p)
        if current_sum + cost <= 1:
            selected_primes.append(p)
            current_sum += cost
