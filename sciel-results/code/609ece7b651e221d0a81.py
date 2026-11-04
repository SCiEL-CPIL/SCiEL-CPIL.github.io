for a in range(4):
    for b in range(4):
        for c in range(4):
            val = sum(U6[a, r] * V6[b, r] * W6_rounded[c, r] for r in range(6))
            i_a = a // 2
            j_a = a % 2
            j_b = b // 2
            k_b = b % 2
            i_c = c // 2
            k_c = c % 2
            expected = 1.0 if (i_a == i_c and j_a == j_b and k_b == k_c) else 0.0
