for m in range(pk, N + 1, pk):
    if m % p_next != 0:
        P[m] *= pk
