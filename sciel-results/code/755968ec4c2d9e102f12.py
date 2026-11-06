"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    def count_divisors(n):
        count = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
            i += 1
        return count

    def get_divisors(n):
        divs = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
            i += 1
        return sorted(divs)

    best_m = 1
    max_divs = 0
    for m in range(1, 513):
        c = count_divisors(m)
        if c > max_divs:
            max_divs = c
            best_m = m
    
    return get_divisors(best_m)
# EVOLVE-BLOCK-END
