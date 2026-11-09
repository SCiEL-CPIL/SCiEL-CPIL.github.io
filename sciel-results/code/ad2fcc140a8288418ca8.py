    for p in primes:
        p2 = p * p
        if p2 >= limit:
            break
        power = p2
        while power < limit:
            next_power = power * p
            for m in range(power, limit, power):
                if m % next_power != 0:
                    P[m] *= power
            power = next_power
    