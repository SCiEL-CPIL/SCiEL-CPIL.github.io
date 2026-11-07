    for n in range(2, MAX_N + 1):
        p = spf_list[n]
        m = n // p
        power = p
        while m > 1 and m % p == 0:
            power *= p
            m //= p
        factor = (power * p - 1) // (p - 1)
        if m == 1:
            sigma_list[n] = factor
        else:
            sigma_list[n] = sigma_list[m] * factor
