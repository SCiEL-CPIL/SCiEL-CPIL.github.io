for i in range(4):
    for k in range(4):
        r = i * 4 + k
        U41[i * 4 + k, r] = 1
        V41[k, r] = 1
        W41[i, r] = 1
