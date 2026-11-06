def construct():
    M = 360
    divisors = []
    for i in range(1, int(M**0.5) + 1):
        if M % i == 0:
            divisors.append(i)
            if i != M // i:
                divisors.append(M // i)
    return divisors
